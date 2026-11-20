import unittest

import torch

from morphik.data import generate_robot_bank
from morphik.geometry import RobotBatch
from morphik.kinematics import geometric_jacobian


class RobotBankTests(unittest.TestCase):
    def test_bank_poses_can_span_the_full_task_space(self):
        train, holdout = generate_robot_bank(24, 4, seed=2026)
        robots = RobotBatch.cat([train, holdout]).to(dtype=torch.float64)
        generator = torch.Generator().manual_seed(47)
        full_rank = torch.zeros(len(robots), dtype=torch.bool)
        for _ in range(8):
            unit_angles = torch.rand(robots.half_ranges.shape, dtype=torch.float64, generator=generator)
            angles = (2 * unit_angles - 1) * robots.half_ranges
            _, _, jacobian = geometric_jacobian(robots, angles)
            sigma = torch.linalg.svdvals(jacobian)
            full_rank |= sigma[:, -1] > 1e-7 * sigma[:, 0]

        self.assertTrue(full_rank.all().item(), f"{(~full_rank).sum().item()} rank-deficient chains")
