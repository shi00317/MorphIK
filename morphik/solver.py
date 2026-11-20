"""Euler integration, best-of-N selection, and damped least-squares polishing."""

import torch
from torch import Tensor

from .geometry import MAX_DOF, RobotBatch
from .kinematics import (
    forward_kinematics,
    geometric_jacobian,
    joint_limit_projection,
    rotation_error,
)
from .model import MorphIK

ORIENTATION_WEIGHT = 0.25


@torch.inference_mode()
def sample_configurations(
    model: MorphIK,
    robots: RobotBatch,
    target_position: Tensor,
    target_orientation: Tensor,
    *,
    generator: torch.Generator,
    samples: int = 16,
    steps: int = 10,
) -> Tensor:
    """Return [batch, samples, 9] centered joint vectors."""
    if steps < 1 or samples < 1:
        raise ValueError("Expected at least one sample and one integration step")
    context = model.context_projection(model.encode(robots, target_position, target_orientation))
    context = context.repeat_interleave(samples, dim=0)
    mask = robots.joint_mask.repeat_interleave(samples, dim=0)
    state = torch.randn((len(context), MAX_DOF), device=context.device, dtype=context.dtype,
                        generator=generator)
    state.masked_fill_(~mask, 0)
    for step in range(steps):
        time = state.new_full((1, 1), step / steps)
        state = state + model.velocity(state, time, context, mask) / steps
    limits = robots.half_ranges[:, None]
    return state.reshape(len(robots), samples, MAX_DOF).clamp(min=-limits, max=limits)


def pose_errors(
    robots: RobotBatch, joints: Tensor, target_position: Tensor, target_orientation: Tensor
) -> tuple[Tensor, Tensor]:
    """Return pairwise position errors in normalized length and orientation in radians."""
    position, orientation = forward_kinematics(robots, joints)
    pos_error = torch.linalg.vector_norm(target_position - position, dim=-1)
    ori_error = torch.linalg.vector_norm(rotation_error(target_orientation, orientation), dim=-1)
    return pos_error, ori_error


@torch.inference_mode()
def select_best(
    robots: RobotBatch, candidates: Tensor, target_position: Tensor, target_orientation: Tensor
) -> Tensor:
    count = candidates.shape[1]
    rows = torch.arange(len(robots), device=candidates.device)
    pos, ori = pose_errors(
        robots[rows.repeat_interleave(count)],
        candidates.flatten(0, 1),
        target_position.repeat_interleave(count, dim=0),
        target_orientation.repeat_interleave(count, dim=0),
    )
    best = (pos + ORIENTATION_WEIGHT * ori).reshape(len(robots), count).argmin(dim=1)
    return candidates[rows, best]


@torch.inference_mode()
def dls_step(
    robots: RobotBatch,
    joints: Tensor,
    target_position: Tensor,
    target_orientation: Tensor,
    *,
    damping: float = 0.02,
) -> Tensor:
    position, orientation, jacobian = geometric_jacobian(robots, joints)
    error = torch.cat((target_position - position, rotation_error(target_orientation, orientation)),
                      dim=-1)
    weights = joints.new_tensor([1, 1, 1, ORIENTATION_WEIGHT, ORIENTATION_WEIGHT, ORIENTATION_WEIGHT])
    jacobian = jacobian * weights[:, None]
    error = error * weights
    transpose = jacobian.transpose(1, 2)
    regularization = damping**2 * torch.eye(6, dtype=joints.dtype, device=joints.device)
    update = torch.linalg.solve_ex(jacobian @ transpose + regularization, error[..., None]).result
    delta = (transpose @ update).squeeze(-1)
    # Bound the largest per-joint update before applying the step near singularities.
    max_update = delta.abs().amax(dim=-1, keepdim=True)
    delta = delta * (0.5 / max_update.clamp_min(0.5))
    return joint_limit_projection(robots, joints + delta)
