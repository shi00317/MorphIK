"""Evaluate best-of-16 pose error on held-out robot identities."""

import torch
from torch import Tensor

from .data import DOF_CLASSES, sample_reachable
from .geometry import RobotBatch
from .model import MorphIK
from .solver import dls_step, pose_errors, sample_configurations, select_best


def _statistics(error: Tensor) -> dict:
    median, p99 = torch.quantile(error, error.new_tensor([0.5, 0.99]), dim=0)
    return {
        name: {"position_pct_of_reach": float(value[0]), "orientation_deg": float(value[1])}
        for name, value in zip(("mean", "median", "p99"), (error.mean(dim=0), median, p99), strict=True)
    }


@torch.inference_mode()
def evaluate(
    model: MorphIK,
    holdout: RobotBatch,
    generator: torch.Generator,
    *,
    poses_per_robot: int,
    batch_size: int,
    samples: int = 16,
    flow_steps: int = 10,
    check_collisions: bool = True,
) -> dict:
    device = holdout.points.device
    indices = torch.arange(len(holdout), device=device).repeat_interleave(poses_per_robot)
    checkpoints = (0, 1, 3, 5)
    records = {step: [] for step in checkpoints}
    dof_values = []
    for ids in indices.split(batch_size):
        batch, valid = sample_reachable(holdout[ids], generator, check_collisions=check_collisions)
        if not valid.any():
            continue
        batch = batch[valid]
        robots, position, orientation = batch.robots, batch.position, batch.orientation
        dof_values.append(robots.joint_mask.sum(dim=1).cpu())
        candidates = sample_configurations(
            model, robots, position, orientation, generator=generator, samples=samples, steps=flow_steps,
        )
        joints = select_best(robots, candidates, position, orientation)
        for step in range(checkpoints[-1] + 1):
            if step:
                joints = dls_step(robots, joints, position, orientation)
            if step in checkpoints:
                pos, ori = pose_errors(robots, joints, position, orientation)
                records[step].append(torch.stack((pos * 100, ori.rad2deg()), dim=-1).cpu())

    if not dof_values:
        raise RuntimeError("No reachable targets were sampled from the holdout bank")
    dof = torch.cat(dof_values)
    errors = {step: torch.cat(batches) for step, batches in records.items()}
    result = {}
    for label, selected in (("all", dof > 0), *((str(n), dof == n) for n in DOF_CLASSES)):
        count = int(selected.sum())
        if count:
            result[label] = {"poses": count, **{
                "raw" if step == 0 else f"dls_{step}": _statistics(error[selected])
                for step, error in errors.items()
            }}
    return {"requested_poses": len(indices), "skipped_poses": len(indices) - len(dof), "results": result}
