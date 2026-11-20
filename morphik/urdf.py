"""Extract and canonicalize the revolute path from a URDF base to a tool link."""

from dataclasses import dataclass, replace
from math import isfinite, pi
from pathlib import Path
from xml.etree import ElementTree as ET

import torch
from torch import Tensor
from torch.nn import functional as F

from .data import tapered_radii
from .geometry import MAX_DOF, CanonicalFrame, RobotBatch, axis_angle, canonicalize, multiply, rotate


@dataclass(frozen=True)
class URDFChain:
    robots: RobotBatch
    frame: CanonicalFrame
    base_link: str
    tip_link: str
    joint_names: tuple[str, ...]
    midpoints: Tensor

    def to_canonical_angles(self, original: Tensor) -> Tensor:
        centered = (original - self.midpoints.to(original)) * self.frame.joint_signs[0].to(original)
        return F.pad(centered, (0, MAX_DOF - len(self.joint_names)))

    def to_urdf_angles(self, canonical: Tensor) -> Tensor:
        active = canonical[..., :len(self.joint_names)]
        return active * self.frame.joint_signs[0].to(active) + self.midpoints.to(active)


def _vector(value: str) -> Tensor:
    numbers = tuple(float(item) for item in value.split())
    if len(numbers) != 3 or not all(map(isfinite, numbers)):
        raise ValueError(f"Expected three finite coordinates, got {value!r}")
    return torch.tensor(numbers, dtype=torch.float64)


def _origin(joint: ET.Element) -> tuple[Tensor, Tensor]:
    origin = joint.find("origin")
    attributes = origin.attrib if origin is not None else {}
    xyz = _vector(attributes.get("xyz", "0 0 0"))
    angles = _vector(attributes.get("rpy", "0 0 0"))
    roll, pitch, yaw = axis_angle(torch.eye(3, dtype=torch.float64), angles)
    return xyz, multiply(yaw, multiply(pitch, roll))


def _bounds(joint: ET.Element) -> tuple[float, float]:
    if joint.get("type") == "continuous":
        return -pi, pi
    limit = joint.find("limit")
    if limit is None or not {"lower", "upper"} <= limit.attrib.keys():
        raise ValueError(f"Revolute joint {joint.attrib['name']!r} needs lower and upper limits")
    lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
    if not (isfinite(lower) and isfinite(upper) and lower <= upper):
        raise ValueError(f"Invalid limits on joint {joint.attrib['name']!r}")
    return lower, upper


def load_urdf(path: str | Path, tip: str, *, base: str | None = None) -> URDFChain:
    """Import a root-to-tip or named base-to-tip path from an expanded URDF file."""
    root = ET.parse(path).getroot()
    links = {link.attrib["name"] for link in root.findall("link")}
    if tip not in links or (base is not None and base not in links):
        raise ValueError(f"Unknown base or tip link in {path}: {base!r}, {tip!r}")
    by_child = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint.attrib['name']!r} needs a parent and child")
        name = child.attrib["link"]
        if name in by_child:
            raise ValueError(f"Multiple URDF joints have the child link {name!r}")
        by_child[name] = joint, parent.attrib["link"]

    path_joints = []
    cursor = tip
    visited = set()
    while cursor != base and cursor in by_child:
        if cursor in visited:
            raise ValueError("Cycle in URDF kinematic chain")
        visited.add(cursor)
        joint, cursor = by_child[cursor]
        path_joints.append(joint)
    if base is not None and cursor != base:
        raise ValueError(f"Tip {tip!r} is not a descendant of base {base!r}")

    position = torch.zeros(3, dtype=torch.float64)
    orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    points, axes, limits, midpoints, names = [], [], [], [], []
    for joint in reversed(path_joints):
        translation, rotation = _origin(joint)
        position = position + rotate(orientation, translation)
        orientation = multiply(orientation, rotation)

        name, kind = joint.attrib["name"], joint.attrib["type"]
        if kind == "fixed":
            continue
        if kind not in {"revolute", "continuous"} or joint.find("mimic") is not None:
            raise ValueError(f"Unsupported joint {name!r} ({kind}) on the selected path")
        lower, upper = _bounds(joint)
        center = (lower + upper) / 2
        axis_element = joint.find("axis")
        local_axis = _vector(axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0")
        if torch.linalg.vector_norm(local_axis) == 0:
            raise ValueError(f"Joint {name!r} has a zero rotation axis")
        local_axis = F.normalize(local_axis, dim=-1)
        if upper > lower:
            points.append(position)
            axes.append(rotate(orientation, local_axis))
            limits.append((upper - lower) / 2)
            midpoints.append(center)
            names.append(name)
        orientation = multiply(orientation, axis_angle(local_axis, position.new_tensor(center)))

    if not 1 <= len(names) <= MAX_DOF:
        raise ValueError(f"Expected 1–{MAX_DOF} active joints on path, got {len(names)}")
    robots, frame = canonicalize(
        torch.stack(points)[None], torch.stack(axes)[None], position.new_tensor(limits)[None],
        position[None], orientation[None],
    )
    robots = replace(robots, radii=tapered_radii(robots, 0.018)).to(dtype=torch.float32)
    return URDFChain(robots, frame, cursor, tip, tuple(names), position.new_tensor(midpoints))
