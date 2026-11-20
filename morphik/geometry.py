"""Common-frame robot geometry and conversion to the paper's canonical frame."""

from dataclasses import dataclass, fields

import torch
from torch import Tensor
from torch.nn import functional as F

MAX_DOF = 9


@dataclass(frozen=True)
class RobotBatch:
    """Padded geometry at the joint-limit midpoint, in unit-length coordinates."""

    points: Tensor           # [batch, 9, 3], a point on each axis
    axes: Tensor             # [batch, 9, 3], unit directions on real joints
    half_ranges: Tensor      # [batch, 9], radians; zero means padding
    tool_position: Tensor    # [batch, 3], TCP with all centered joints at zero
    tool_quaternion: Tensor  # [batch, 4], wxyz
    radii: Tensor            # [batch, 9], radius of each outgoing capsule

    @property
    def joint_mask(self) -> Tensor:
        return self.half_ranges > 0

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, index) -> "RobotBatch":
        if isinstance(index, int):
            index = [index]
        return RobotBatch(*(getattr(self, field.name)[index] for field in fields(self)))

    def to(self, *args, **kwargs) -> "RobotBatch":
        return RobotBatch(*(getattr(self, f.name).to(*args, **kwargs) for f in fields(self)))

    @classmethod
    def cat(cls, batches: list["RobotBatch"]) -> "RobotBatch":
        return cls(*(torch.cat([getattr(batch, f.name) for batch in batches]) for f in fields(cls)))


def axis_angle(axis: Tensor, angle: Tensor) -> Tensor:
    half = angle[..., None] / 2
    return torch.cat((half.cos(), half.sin() * axis), dim=-1)


def conjugate(quaternion: Tensor) -> Tensor:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def multiply(a: Tensor, b: Tensor) -> Tensor:
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    scalar = aw * bw - (av * bv).sum(dim=-1, keepdim=True)
    vector = aw * bv + bw * av + torch.linalg.cross(av, bv)
    return torch.cat((scalar, vector), dim=-1)


def rotate(quaternion: Tensor, vector: Tensor) -> Tensor:
    qv = quaternion[..., 1:]
    doubled = 2 * torch.linalg.cross(qv, vector)
    return vector + quaternion[..., :1] * doubled + torch.linalg.cross(qv, doubled)


def _first_component_sign(vector: Tensor, tolerance: float) -> Tensor:
    first = (vector.abs() > tolerance).to(torch.int64).argmax(dim=-1, keepdim=True)
    return torch.where(vector.gather(-1, first) < 0, -1.0, 1.0).to(vector.dtype)


def canonical_quaternion(quaternion: Tensor) -> Tensor:
    """Pick a consistent representative of the two unit quaternions for a rotation."""
    quaternion = F.normalize(quaternion, dim=-1)
    return quaternion * _first_component_sign(quaternion, 1e-8)


def _basis_quaternion(matrix: Tensor) -> Tensor:
    """Convert an orthonormal frame to wxyz using the largest quaternion component."""
    xx, xy, xz = matrix[..., 0, :].unbind(-1)
    yx, yy, yz = matrix[..., 1, :].unbind(-1)
    zx, zy, zz = matrix[..., 2, :].unbind(-1)
    w = 1 + xx + yy + zz
    x = 1 + xx - yy - zz
    y = 1 - xx + yy - zz
    z = 1 - xx - yy + zz
    candidates = torch.stack(
        (
            torch.stack((w, zy - yz, xz - zx, yx - xy), dim=-1),
            torch.stack((zy - yz, x, yx + xy, xz + zx), dim=-1),
            torch.stack((xz - zx, yx + xy, y, yz + zy), dim=-1),
            torch.stack((yx - xy, xz + zx, yz + zy, z), dim=-1),
        ),
        dim=-2,
    )
    best = torch.stack((w, x, y, z), dim=-1).argmax(dim=-1)
    quaternion = candidates.gather(-2, best[..., None, None].expand(*best.shape, 1, 4))
    return canonical_quaternion(quaternion.squeeze(-2))


@dataclass(frozen=True)
class CanonicalFrame:
    origin: Tensor
    orientation: Tensor  # canonical basis expressed in the original frame
    scale: Tensor        # original chain length, with trailing singleton dimension
    joint_signs: Tensor  # active joints only, maps centered angles in either direction

    def to_canonical(self, position: Tensor, quaternion: Tensor) -> tuple[Tensor, Tensor]:
        inverse = conjugate(self.orientation)
        return (
            rotate(inverse, (position - self.origin) / self.scale),
            canonical_quaternion(multiply(inverse, quaternion)),
        )

    def to_original(self, position: Tensor, quaternion: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self.origin + rotate(self.orientation, position * self.scale),
            canonical_quaternion(multiply(self.orientation, quaternion)),
        )


def canonicalize(
    points: Tensor,
    axes: Tensor,
    half_ranges: Tensor,
    tool_position: Tensor,
    tool_quaternion: Tensor,
    *,
    tolerance: float = 1e-7,
) -> tuple[RobotBatch, CanonicalFrame]:
    """Canonicalize a batch with the same number of active joints in each row."""
    batch_size, dof, _ = points.shape
    if not 1 <= dof <= MAX_DOF:
        raise ValueError(f"Expected between 1 and {MAX_DOF} revolute joints, got {dof}")
    axes = F.normalize(axes, dim=-1)
    tool_quaternion = canonical_quaternion(tool_quaternion)
    first_axis = axes[:, 0]
    first_point = points[:, 0]

    # Locate the point on the first axis closest to the first nonparallel axis.
    cross_squared = torch.linalg.cross(first_axis[:, None], axes).square().sum(-1)
    nonparallel = cross_squared > tolerance**2
    indices = torch.arange(batch_size, device=points.device)
    next_axis_index = nonparallel.to(torch.int64).argmax(dim=1)
    next_axis = axes[indices, next_axis_index]
    offset = points[indices, next_axis_index] - first_point
    cosine = (first_axis * next_axis).sum(-1)
    denominator = cross_squared[indices, next_axis_index].clamp_min(tolerance**2)
    along_first = ((offset * first_axis).sum(-1) - cosine * (offset * next_axis).sum(-1))
    along_first = along_first / denominator
    # With only parallel axes, use the TCP's projection to fix the axial origin.
    tcp_projection = ((tool_position - first_point) * first_axis).sum(-1)
    along_first = torch.where(nonparallel.any(1), along_first, tcp_projection)
    origin = first_point + along_first[:, None] * first_axis

    # Each subsequent point is the projection of the preceding point onto its axis.
    projected = [origin]
    for point, axis in zip(points.unbind(1)[1:], axes.unbind(1)[1:], strict=True):
        projected.append(point + ((projected[-1] - point) * axis).sum(-1, keepdim=True) * axis)
    projected = torch.stack(projected, dim=1)
    chain = torch.cat((projected, tool_position[:, None]), dim=1)
    scale = torch.linalg.vector_norm(chain.diff(dim=1), dim=-1).sum(dim=1, keepdim=True)
    if (scale <= torch.finfo(scale.dtype).eps).any():
        raise ValueError("Cannot normalize a robot with zero canonical chain length")

    # The first offset from the first axis sets +x. The TCP frame breaks axial ties.
    offsets = (chain[:, 1:] - origin[:, None]) / scale[:, None]
    tool_x = rotate(tool_quaternion, points.new_tensor([1.0, 0.0, 0.0]).expand(batch_size, 3))
    tool_y = rotate(tool_quaternion, points.new_tensor([0.0, 1.0, 0.0]).expand(batch_size, 3))
    offsets = torch.cat((offsets, tool_x[:, None], tool_y[:, None]), dim=1)
    radial = offsets - (offsets * first_axis[:, None]).sum(-1, keepdim=True) * first_axis[:, None]
    first_radial = (torch.linalg.vector_norm(radial, dim=-1) > tolerance).long().argmax(dim=1)
    x_axis = F.normalize(radial[indices, first_radial], dim=-1)

    # Canonicalize both choices of first-axis sign and compare their descriptors.
    z_axes = torch.stack((first_axis, -first_axis), dim=1)
    x_axes = x_axis[:, None].expand_as(z_axes)
    basis = torch.stack((x_axes, torch.linalg.cross(z_axes, x_axes), z_axes), dim=-1)
    basis_quaternions = _basis_quaternion(basis)
    candidates_points = ((projected - origin[:, None]) / scale[:, None])[:, None] @ basis
    candidates_axes = axes[:, None] @ basis
    signs = _first_component_sign(candidates_axes, tolerance)
    candidates_axes = candidates_axes * signs
    candidates_tool = (((tool_position - origin) / scale)[:, None, None] @ basis).squeeze(-2)
    candidates_quat = canonical_quaternion(
        multiply(conjugate(basis_quaternions), tool_quaternion[:, None])
    )
    description = torch.cat(
        (
            torch.cat((candidates_points, candidates_axes), dim=-1).flatten(2),
            candidates_tool,
            candidates_quat,
        ),
        dim=-1,
    )
    difference = description[:, 0] - description[:, 1]
    variant = (_first_component_sign(difference, tolerance).squeeze(-1) > 0).long()

    padding = MAX_DOF - dof
    robots = RobotBatch(
        F.pad(candidates_points[indices, variant], (0, 0, 0, padding)),
        F.pad(candidates_axes[indices, variant], (0, 0, 0, padding)),
        F.pad(half_ranges, (0, padding)),
        candidates_tool[indices, variant],
        candidates_quat[indices, variant],
        points.new_zeros(batch_size, MAX_DOF),
    )
    frame = CanonicalFrame(origin, basis_quaternions[indices, variant], scale,
                           signs[indices, variant].squeeze(-1))
    return robots, frame
