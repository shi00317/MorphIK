"""Batched product-of-exponentials kinematics for canonical revolute chains."""

import torch
from torch import Tensor
from torch.nn import functional as F

from .geometry import RobotBatch, axis_angle, conjugate, multiply, rotate


def _forward_chain(
    robots: RobotBatch, joints: Tensor, *, capture_frames: bool
) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
    joint_rotation = axis_angle(robots.axes, joints.masked_fill(~robots.joint_mask, 0))
    joint_translation = robots.points - rotate(joint_rotation, robots.points)
    position = torch.zeros_like(robots.tool_position)
    orientation = joints.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(len(robots), 4)
    frame_positions, frame_orientations = [], []
    for shift, rotation in zip(joint_translation.unbind(1), joint_rotation.unbind(1), strict=True):
        if capture_frames:
            frame_positions.append(position)
            frame_orientations.append(orientation)
        position = position + rotate(orientation, shift)
        orientation = multiply(orientation, rotation)
    position = position + rotate(orientation, robots.tool_position)
    orientation = F.normalize(multiply(orientation, robots.tool_quaternion), dim=-1)
    return position, orientation, frame_positions, frame_orientations


def forward_kinematics(robots: RobotBatch, joints: Tensor) -> tuple[Tensor, Tensor]:
    """Return TCP position and wxyz quaternion for centered angles in radians."""
    position, orientation, _, _ = _forward_chain(robots, joints, capture_frames=False)
    return position, orientation


def posed_chain(robots: RobotBatch, joints: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return TCP pose, joint axis points and joint directions at the given angles."""
    position, orientation, frame_positions, frame_orientations = _forward_chain(
        robots, joints, capture_frames=True
    )
    prefixes = torch.stack(frame_orientations, dim=1)
    points = torch.stack(frame_positions, dim=1) + rotate(prefixes, robots.points)
    axes = rotate(prefixes, robots.axes)
    return position, orientation, points, axes


def geometric_jacobian(robots: RobotBatch, joints: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return the TCP pose and its [linear; angular] spatial Jacobian."""
    position, orientation, points, axes = posed_chain(robots, joints)
    linear = torch.linalg.cross(axes, position[:, None] - points)
    return position, orientation, torch.cat((linear, axes), dim=-1).transpose(1, 2)


def rotation_error(target: Tensor, orientation: Tensor) -> Tensor:
    """Shortest rotation from orientation to target as a spatial rotation vector."""
    relative = multiply(target, conjugate(orientation))
    relative = torch.where(relative[..., :1] < 0, -relative, relative)
    vector = relative[..., 1:]
    length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, relative[..., :1])
    return vector * torch.where(length > 1e-8, angle / length.clamp_min(1e-8), 2.0)


def joint_limit_projection(robots: RobotBatch, joints: Tensor) -> Tensor:
    return joints.clamp(min=-robots.half_ranges, max=robots.half_ranges)
