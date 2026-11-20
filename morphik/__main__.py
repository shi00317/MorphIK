"""Command line entry point for building a bank, training, and solving IK."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .data import generate_robot_bank, load_robot_bank, save_robot_bank
from .evaluation import evaluate
from .model import ModelConfig
from .solver import dls_step, pose_errors, sample_configurations, select_best
from .training import TrainConfig, load_model, train
from .urdf import load_urdf

DEFAULT_BANK = Path("artifacts/robots.pt")
DEFAULT_CHECKPOINT = Path("artifacts/morphik.pt")


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer")
    return number


def _generate(args: argparse.Namespace) -> None:
    training, holdout = generate_robot_bank(args.robots_per_dof, args.holdout_per_dof, args.seed)
    save_robot_bank(args.output, training, holdout, seed=args.seed)
    print(f"Saved {len(training):,} training and {len(holdout):,} held-out robots to {args.output}")


def _train(args: argparse.Namespace) -> None:
    config = ModelConfig(width=32, attention_heads=4, encoder_blocks=2, flow_width=64,
                         flow_blocks=2, time_width=32) if args.tiny else ModelConfig()
    options = asdict(TrainConfig())
    options.update({key: getattr(args, key) for key in options})
    if args.lr <= 0 or args.final_lr <= 0 or not 0 <= args.ema_decay < 1:
        raise ValueError("Expected positive learning rates and 0 <= EMA decay < 1")
    bank = load_robot_bank(args.bank, "train", _device(args.device))
    train(bank, args.output, config, TrainConfig(**options), resume=args.resume)
    print(f"Saved checkpoint to {args.output}")


def _evaluate(args: argparse.Namespace) -> None:
    device = _device(args.device)
    results = evaluate(
        load_model(args.checkpoint, device),
        load_robot_bank(args.bank, "holdout", device),
        torch.Generator(device=device).manual_seed(args.seed),
        poses_per_robot=args.poses_per_robot,
        batch_size=args.batch_size,
        samples=args.samples,
        flow_steps=args.flow_steps,
        check_collisions=args.check_collisions,
    )
    print(json.dumps(results, indent=2))


def _solve(args: argparse.Namespace) -> None:
    device = _device(args.device)
    chain = load_urdf(args.urdf, args.tip, base=args.base)
    robots = chain.robots.to(device)
    position = chain.midpoints.new_tensor(args.position)[None]
    orientation = chain.midpoints.new_tensor(args.quaternion)[None]
    if not (position.isfinite().all() and orientation.isfinite().all() and orientation.norm() > 0):
        raise ValueError("Expected a finite position and a finite, nonzero quaternion")
    position, orientation = chain.frame.to_canonical(position, orientation)
    position = position.to(robots.points)
    orientation = orientation.to(robots.points)
    candidates = sample_configurations(
        load_model(args.checkpoint, device),
        robots,
        position,
        orientation,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        samples=args.samples,
        steps=args.flow_steps,
    )
    joints = select_best(robots, candidates, position, orientation)
    raw_position, raw_orientation = pose_errors(robots, joints, position, orientation)
    for _ in range(args.dls_steps):
        joints = dls_step(robots, joints, position, orientation)
    final_position, final_orientation = pose_errors(robots, joints, position, orientation)
    angles = chain.to_urdf_angles(joints[0].cpu().to(chain.midpoints))
    # Float32 centered limits can round past the original URDF bounds.
    angles = angles.clamp(min=chain.lower_limits, max=chain.upper_limits)
    scale = float(chain.frame.scale.item())
    print(json.dumps({
        "base": chain.base_link,
        "tip": chain.tip_link,
        "joint_angles_rad": dict(zip(chain.joint_names, angles.tolist(), strict=True)),
        "raw_position_error_m": raw_position.item() * scale,
        "raw_orientation_error_deg": raw_orientation.rad2deg().item(),
        "dls_steps": args.dls_steps,
        "position_error_m": final_position.item() * scale,
        "orientation_error_deg": final_orientation.rad2deg().item(),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="MorphIK paper reimplementation")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate a fixed synthetic robot bank")
    generate.add_argument("--output", type=Path, default=DEFAULT_BANK)
    generate.add_argument("--robots-per-dof", type=_positive, default=100_000)
    generate.add_argument("--holdout-per-dof", type=_positive, default=200)
    generate.set_defaults(run=_generate)

    fit = commands.add_parser("train", help="Train on newly sampled configurations from the bank")
    fit.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    fit.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    fit.add_argument("--resume", type=Path)
    fit.add_argument("--tiny", action="store_true", help="Use the smaller network for smoke runs")
    fit.add_argument("--amp", action="store_true", help="Enable bfloat16 autocast")
    defaults = TrainConfig()
    for name in ("steps", "batch_size", "log_every", "save_every"):
        fit.add_argument("--" + name.replace("_", "-"), type=_positive, default=getattr(defaults, name))
    fit.add_argument("--flat-steps", type=_nonnegative, default=defaults.flat_steps)
    for name in ("lr", "final_lr", "ema_decay"):
        fit.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    fit.set_defaults(run=_train)

    test = commands.add_parser("evaluate", help="Evaluate raw and DLS-polished IK on held-out robots")
    test.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    test.add_argument("--poses-per-robot", type=_positive, default=8)
    test.add_argument("--batch-size", type=_positive, default=64)
    test.set_defaults(run=_evaluate)

    solve = commands.add_parser("solve", help="Solve IK for one target on a URDF robot")
    solve.add_argument("--urdf", type=Path, required=True)
    solve.add_argument("--base", help="Base link (default is the root of the tip's chain)")
    solve.add_argument("--tip", required=True, help="End-effector link name")
    solve.add_argument("--position", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    solve.add_argument("--quaternion", type=float, nargs=4, required=True, metavar=("W", "X", "Y", "Z"))
    solve.add_argument("--dls-steps", type=_nonnegative, default=3)
    solve.set_defaults(run=_solve)

    for command in (fit, test):
        command.add_argument("--no-collision-filter", action="store_false", dest="check_collisions")
    for command in (test, solve):
        command.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
        command.add_argument("--samples", type=_positive, default=16)
        command.add_argument("--flow-steps", type=_positive, default=10)
    for command in (fit, test, solve):
        command.add_argument("--device", default="auto", help="auto, cpu, cuda, or a specific CUDA device")
    for command in (generate, fit, test, solve):
        command.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    try:
        args.run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
