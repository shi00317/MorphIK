# MorphIK

An independent PyTorch reimplementation of morphology-conditioned inverse kinematics for serial robotic arms with 6–9 revolute joints. One model takes a robot description and a target end-effector pose and samples candidate joint configurations. [MorphIK paper by Clasmeier et al.](https://arxiv.org/abs/2609.29908)

> [!NOTE]
> I couldn't find an official open-source implementation from the paper's authors, so I used AI tools to create this unofficial version for testing.

## How it works

1. **Generate a robot bank.** Sample link lengths, joint axes, joint limits, and tool transforms. Express the axes in a common canonical frame and scale the chain to unit length.
2. **Generate training pairs online.** Draw joint angles within the limits, reject capsule self-collisions, and compute the corresponding target pose with forward kinematics (FK). The bank stores robot descriptions, and each training batch draws fresh configurations.
3. **Train the flow model.** A four-block Transformer encodes the robot's joint tokens, tool center point, and target pose. A six-block AdaLN head predicts a velocity in joint space. Training combines flow matching with a time-gated differentiable FK loss.
4. **Solve IK.** Integrate from noise with 10 Euler steps, score 16 candidate configurations by pose error, and optionally refine the best candidate with damped least squares (DLS).

## Setup

Use Python 3.10+ and PyTorch 2.5+. GPU training uses a CUDA-enabled PyTorch build. Run these commands from the repository root.

```sh
python -m pip install -e .
python -m morphik --help
```

## Quick start on CPU

Generate 80 training robots and 16 holdout robots, train a small model for 20 updates, and evaluate it on 32 held-out poses.

```sh
python -m morphik generate --robots-per-dof 24 --holdout-per-dof 4 \
  --output artifacts/smoke/robots.pt

python -m morphik train --bank artifacts/smoke/robots.pt \
  --output artifacts/smoke/model.pt --tiny --steps 20 --batch-size 16 --device cpu

python -m morphik evaluate --bank artifacts/smoke/robots.pt \
  --checkpoint artifacts/smoke/model.pt --poses-per-robot 2 --device cpu
```

The evaluator reports position error as a percentage of chain length and orientation error in degrees. It measures the best candidate before refinement and after 1, 3, and 5 DLS steps. This small run verifies the pipeline.

## Train the paper-sized model

```sh
python -m morphik generate
python -m morphik train --device cuda --amp
python -m morphik evaluate --device cuda
```

The defaults generate 400,000 robot descriptions, with 200 holdouts for each of the four DoF classes. Training uses batches of 2,048 for 1,000,000 steps. The learning rate is `5.4e-4` for the first 350,000 steps and then decreases linearly to `8e-6`. The bank and checkpoint are saved to `artifacts/robots.pt` and `artifacts/morphik.pt`, respectively. Evaluation and inference load the exponential moving average (EMA) weights.

To resume an interrupted run, pass `--resume artifacts/morphik.pt` with the same bank and training settings. See `python -m morphik train --help` for checkpoint intervals and schedule options.

## Solve IK for a URDF

Use an expanded URDF, the name of its end-effector link, and a reachable target. Pass position in meters and orientation as a quaternion in `W X Y Z` order. The target is relative to the root of the selected chain by default. Use `--base` to select another base link.

```sh
python -m morphik solve --urdf path/to/robot.urdf --tip tool0 \
  --checkpoint artifacts/morphik.pt \
  --position 0.45 0.10 0.35 --quaternion 1 0 0 0 --dls-steps 3
```

The command returns JSON with joint angles in radians under the original URDF joint names, plus pose errors before and after refinement. The capsule filter applies to synthetic data sampling. The solver ranks candidates by pose error and enforces joint limits.

## Code and verification

| File | Role |
| --- | --- |
| [morphik/data.py](morphik/data.py) | Robot bank and online training samples |
| [morphik/geometry.py](morphik/geometry.py), [morphik/kinematics.py](morphik/kinematics.py) | Canonical representation, differentiable FK, and Jacobians |
| [morphik/model.py](morphik/model.py), [morphik/training.py](morphik/training.py) | Transformer, flow head, loss, and checkpointing |
| [morphik/solver.py](morphik/solver.py), [morphik/urdf.py](morphik/urdf.py) | Sampling, DLS refinement, and URDF conversion |

The review verified full bank generation, a paper-sized CUDA update, checkpoint resume, and FK against an independent Franka Panda calculation. The generator's rank filter and capsule dimensions are reconstruction choices. The comparison with the paper's Table II remains open. [Implementation checklist and review findings](CHECKLIST.md)

```sh
python -m unittest discover -s tests -v
```
