"""Online training and checkpointing for the MorphIK flow model."""

import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .data import sample_training_batch
from .geometry import RobotBatch
from .model import ModelConfig, MorphIK, flow_matching_loss


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1_000_000
    batch_size: int = 2048
    flat_steps: int = 350_000
    lr: float = 5.4e-4
    final_lr: float = 8e-6
    ema_decay: float = 0.999
    check_collisions: bool = True
    amp: bool = False
    seed: int = 0
    log_every: int = 100
    save_every: int = 10_000

    def learning_rate(self, step: int) -> float:
        if step <= self.flat_steps:
            return self.lr
        fraction = (step - self.flat_steps) / (self.steps - self.flat_steps)
        return self.lr + fraction * (self.final_lr - self.lr)


def _store_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temp_path = Path(handle.name)
        try:
            torch.save(state, handle)
            handle.close()
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)


def load_model(path: Path, device: torch.device) -> MorphIK:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = MorphIK(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["ema"])
    return model.eval()


def train(
    bank: RobotBatch,
    path: Path,
    model_config: ModelConfig,
    options: TrainConfig,
    *,
    resume: Path | None = None,
) -> None:
    device = bank.points.device
    torch.manual_seed(options.seed)
    generator = torch.Generator(device=device).manual_seed(options.seed)
    state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if state is not None:
        model_config = ModelConfig(**state["model_config"])
    model = MorphIK(model_config).to(device)
    ema = deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.lr, weight_decay=1e-4, fused=device.type == "cuda",
    )
    start_step = 0
    if state is not None:
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        if state["rng_device"] == device.type:
            generator.set_state(state["rng"])
        else:
            generator.manual_seed(options.seed + start_step)
        del state
    if options.steps <= start_step:
        raise ValueError(f"--steps must exceed checkpoint step {start_step}")

    params, averaged_params = list(model.parameters()), list(ema.parameters())
    for step in range(start_step + 1, options.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = options.learning_rate(step)
        batch = sample_training_batch(
            bank, options.batch_size, generator, check_collisions=options.check_collisions
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=options.amp):
            loss, metrics = flow_matching_loss(model, batch, generator)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(averaged_params, params, 1 - options.ema_decay)

        if step == start_step + 1 or step == options.steps or step % options.log_every == 0:
            print(f"step={step}  loss={loss.item():.5f}  flow={metrics['flow'].item():.5f}  "
                  f"gated_fk={metrics['fk'].item():.5f}", flush=True)
        if step == options.steps or step % options.save_every == 0:
            _store_checkpoint(path, {
                "step": step,
                "model_config": asdict(model_config),
                "train_config": asdict(options),
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": generator.get_state(),
                "rng_device": device.type,
            })
