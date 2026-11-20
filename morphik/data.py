"""Procedural robot bank and online, collision-filtered FK supervision."""

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .geometry import MAX_DOF, RobotBatch, canonicalize
from .kinematics import forward_kinematics, geometric_jacobian, posed_chain

DOF_CLASSES = (6, 7, 8, 9)


def _outgoing_endpoints(robots: RobotBatch, joint_points: Tensor, tool_position: Tensor) -> Tensor:
    following = torch.cat((joint_points[:, 1:], tool_position[:, None]), dim=1)
    next_joint = F.pad(robots.joint_mask[:, 1:], (0, 1), value=False)
    return torch.where(next_joint[..., None], following, tool_position[:, None])


def tapered_radii(robots: RobotBatch, base_radius: Tensor | float) -> Tensor:
    """Assign tapered radii in normalized units, omitting zero-length links."""
    end = _outgoing_endpoints(robots, robots.points, robots.tool_position)
    lengths = torch.linalg.vector_norm(end - robots.points, dim=-1)
    active = robots.joint_mask & (lengths > 1e-7)
    index = torch.arange(MAX_DOF, device=lengths.device)
    fraction = index / (robots.joint_mask.sum(dim=-1, keepdim=True) - 1).clamp_min(1)
    radius = base_radius * (1 - 0.4 * fraction)

    # Shrink capsules around short links so opposite ends of a bridge can clear.
    order = active.cumsum(dim=-1)
    nearby = (order[:, :, None] - order[:, None, :]).abs() <= 1
    nearest_length = lengths[:, None].masked_fill(~(active[:, None] & nearby), torch.inf)
    radius = torch.minimum(radius, 0.24 * nearest_length.amin(dim=-1))
    return radius.masked_fill(~active, 0)


def _procedural_robots(count: int, dof: int, generator: torch.Generator) -> RobotBatch:
    dtype = torch.float64

    def uniform(*shape: int) -> Tensor:
        return torch.rand(*shape, dtype=dtype, generator=generator)

    def normals(*shape: int) -> Tensor:
        return torch.randn(*shape, dtype=dtype, generator=generator)

    axes = [torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(count, 3)]
    modes = uniform(count, dof - 1)
    directions = F.normalize(normals(count, dof - 1, 3), dim=-1)
    for mode, direction in zip(modes.unbind(1), directions.unbind(1), strict=True):
        previous = axes[-1]
        orthogonal = direction - (direction * previous).sum(-1, keepdim=True) * previous
        axis = torch.where((mode < 0.8)[:, None], F.normalize(orthogonal, dim=-1), direction)
        axes.append(torch.where((mode < 0.25)[:, None], previous, axis))
    axes = torch.stack(axes, dim=1)

    # Normalized independent Gamma(2, 1) variates yield Dirichlet(alpha=2).
    lengths = -torch.log1p(-uniform(count, dof, 2)).sum(dim=-1)
    keep = uniform(count, dof) >= 0.35
    longest = F.one_hot(lengths.argmax(dim=1), num_classes=dof).bool()
    keep |= (~keep.any(dim=1, keepdim=True)) & longest
    lengths = lengths * keep

    wrist = (uniform(count, 1) < 0.30) & (torch.arange(dof) >= dof - 2) & (dof >= 7)
    lengths = lengths * torch.where(wrist, 0.2, 1.0)
    lengths = lengths / lengths.sum(dim=1, keepdim=True)
    limits = uniform(count, dof)
    half_ranges = torch.where(wrist, 0.15 + (0.9 - 0.15) * limits,
                              0.9 + (torch.pi - 0.9) * limits)

    # For an inter-joint link, choose a direction in the next axis's normal plane.
    link_directions = normals(count, dof, 3)
    inter_joint = link_directions[:, :-1]
    next_axes = axes[:, 1:]
    inter_joint -= (inter_joint * next_axes).sum(-1, keepdim=True) * next_axes
    offsets = F.normalize(link_directions, dim=-1) * lengths[..., None]
    chain = F.pad(offsets.cumsum(dim=1), (0, 0, 1, 0))
    tool_quaternion = F.normalize(normals(count, 4), dim=-1)
    robots, _ = canonicalize(chain[:, :-1], axes, half_ranges, chain[:, -1], tool_quaternion)
    radii = tapered_radii(robots, 0.012 + 0.012 * uniform(count, 1))
    return replace(robots, radii=radii)


def _has_full_task_rank(robots: RobotBatch, generator: torch.Generator) -> Tensor:
    """Keep chains with a locally six-dimensional task space at a sampled pose."""
    remaining = torch.arange(len(robots))
    accepted = torch.zeros(len(robots), dtype=torch.bool)
    for _ in range(3):
        pending = robots[remaining]
        _, _, jacobian = geometric_jacobian(pending, _sample_angles(pending, generator))
        singular_values = torch.linalg.svdvals(jacobian)
        full_rank = singular_values[:, -1] > 1e-7 * singular_values[:, 0]
        accepted[remaining[full_rank]] = True
        remaining = remaining[~full_rank]
        if not remaining.numel():
            break
    return accepted


def _sample_morphologies(count: int, dof: int, generator: torch.Generator) -> RobotBatch:
    """Reject degenerate proposals before assigning them to either split."""
    batches = []
    remaining = count
    for _ in range(64):
        robots = _procedural_robots(remaining, dof, generator)
        accepted = robots[_has_full_task_rank(robots, generator)]
        batches.append(accepted.to(dtype=torch.float32))
        remaining -= len(accepted)
        if not remaining:
            return RobotBatch.cat(batches)
    raise RuntimeError(f"Could not generate {count} full-rank {dof}-DoF chains")


def generate_robot_bank(
    robots_per_dof: int = 100_000, holdout_per_dof: int = 200, seed: int = 0
) -> tuple[RobotBatch, RobotBatch]:
    """Draw a fixed bank with disjoint robot identities for training and evaluation."""
    if not 0 < holdout_per_dof < robots_per_dof:
        raise ValueError("Expected 0 < holdout_per_dof < robots_per_dof")
    generator = torch.Generator().manual_seed(seed)
    train, holdout = [], []
    train_per_dof = robots_per_dof - holdout_per_dof
    for dof in DOF_CLASSES:
        for destination, size in ((train, train_per_dof), (holdout, holdout_per_dof)):
            for offset in range(0, size, 4096):
                destination.append(_sample_morphologies(min(4096, size - offset), dof, generator))
    return RobotBatch.cat(train), RobotBatch.cat(holdout)


def save_robot_bank(path: Path, train: RobotBatch, holdout: RobotBatch, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": seed, "train": vars(train), "holdout": vars(holdout)}, path)


def load_robot_bank(path: Path, split: str, device: torch.device) -> RobotBatch:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return RobotBatch(**payload[split]).to(device)


def _point_segment_distance_sq(point: Tensor, start: Tensor, end: Tensor) -> Tensor:
    direction = end - start
    offset = point - start
    denominator = direction.square().sum(-1).clamp_min(1e-20)
    fraction = ((offset * direction).sum(-1) / denominator).clamp(0, 1)
    return (offset - fraction[..., None] * direction).square().sum(-1)


def segment_distance_sq(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
    """Squared separation of each pair of closed 3D line segments."""
    u, v, offset = b - a, d - c, a - c
    normal = torch.linalg.cross(u, v)
    denominator = normal.square().sum(-1)
    safe_denominator = denominator.clamp_min(1e-30)
    s = (torch.linalg.cross(v, offset) * normal).sum(-1) / safe_denominator
    t = (torch.linalg.cross(u, offset) * normal).sum(-1) / safe_denominator
    inside = (denominator > 0) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    interior = offset + s.clamp(0, 1)[..., None] * u - t.clamp(0, 1)[..., None] * v
    interior = interior.square().sum(-1).masked_fill(~inside, torch.inf)
    return torch.stack(
        (interior, _point_segment_distance_sq(a, c, d), _point_segment_distance_sq(b, c, d),
         _point_segment_distance_sq(c, a, b), _point_segment_distance_sq(d, a, b))
    ).amin(dim=0)


def capsule_collisions(robots: RobotBatch, points: Tensor, tool_position: Tensor) -> Tensor:
    """Check nonadjacent capsule pairs, skipping empty segments and shared joints."""
    i, j = torch.triu_indices(MAX_DOF, MAX_DOF, offset=2, device=points.device)
    ends = _outgoing_endpoints(robots, points, tool_position)
    active = robots.radii > 0
    order = active.cumsum(dim=1)
    pairs = active[:, i] & active[:, j] & ((order[:, j] - order[:, i]) > 1)
    distance_sq = segment_distance_sq(points[:, i], ends[:, i], points[:, j], ends[:, j])
    radii_sq = (robots.radii[:, i] + robots.radii[:, j]).square()
    return (pairs & (distance_sq < radii_sq)).any(dim=1)


@dataclass(frozen=True)
class IKBatch:
    robots: RobotBatch
    joints: Tensor
    position: Tensor
    orientation: Tensor

    def __getitem__(self, index: Tensor) -> "IKBatch":
        return IKBatch(self.robots[index], self.joints[index], self.position[index], self.orientation[index])


def _sample_angles(robots: RobotBatch, generator: torch.Generator) -> Tensor:
    limits = robots.half_ranges
    unit_values = torch.rand(limits.shape, dtype=limits.dtype, device=limits.device, generator=generator)
    return (2 * unit_values - 1) * limits


def _draw_poses(
    robots: RobotBatch, generator: torch.Generator, check_collisions: bool
) -> tuple[IKBatch, Tensor]:
    joints = _sample_angles(robots, generator)
    if check_collisions:
        position, orientation, points, _ = posed_chain(robots, joints)
        valid = ~capsule_collisions(robots, points, position)
    else:
        position, orientation = forward_kinematics(robots, joints)
        valid = torch.ones(len(robots), dtype=torch.bool, device=joints.device)
    return IKBatch(robots, joints, position, orientation), valid


@torch.no_grad()
def sample_training_batch(
    bank: RobotBatch, size: int, generator: torch.Generator, *, check_collisions: bool = True
) -> IKBatch:
    device = bank.points.device
    remaining = size
    batches = []
    for _ in range(32):
        count = int(1.5 * remaining) if check_collisions else remaining
        ids = torch.randint(len(bank), (count,), device=device, generator=generator)
        batch, valid = _draw_poses(bank[ids], generator, check_collisions)
        if not check_collisions:
            return batch
        accepted = valid.nonzero(as_tuple=True)[0][:remaining]
        batches.append(batch[accepted])
        remaining -= accepted.numel()
        if not remaining:
            if len(batches) == 1:
                return batches[0]
            return IKBatch(
                RobotBatch.cat([b.robots for b in batches]),
                torch.cat([b.joints for b in batches]),
                torch.cat([b.position for b in batches]),
                torch.cat([b.orientation for b in batches]),
            )
    raise RuntimeError(f"Could not fill training batch after 32 draws ({remaining} missing)")


@torch.no_grad()
def sample_reachable(
    robots: RobotBatch, generator: torch.Generator, *, check_collisions: bool = True
) -> tuple[IKBatch, Tensor]:
    """Sample one target per supplied robot, returning a mask of successful draws."""
    batch, valid = _draw_poses(robots, generator, check_collisions)
    if check_collisions:
        for _ in range(63):
            rejected = (~valid).nonzero(as_tuple=True)[0]
            if not rejected.numel():
                break
            new, success = _draw_poses(robots[rejected], generator, True)
            batch.joints[rejected] = new.joints
            batch.position[rejected] = new.position
            batch.orientation[rejected] = new.orientation
            valid[rejected] = success
    return batch, valid
