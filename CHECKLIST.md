# MorphIK implementation checklist

This checklist follows Section III and Figures 3–4 of Clasmeier et al., *MorphIK: Morphology-Conditioned Neural Inverse Kinematics for Unknown Robots*. I found no public author implementation in searches of arXiv, GitHub, and the authors' lab publications. The code in this workspace is an independent PyTorch implementation. [Paper](https://arxiv.org/abs/2609.29908) · [Lab publications](https://wtmbib.informatik.uni-hamburg.de/Basilic/)

The implementation checks passed on Python 3.11 and PyTorch 2.8. The full-training comparison with Table II is still open.

## 1. Review the training data

- [x] Confirm that the training and holdout splits contain different robot identities. The generated bank contained 399,200 training and 800 held-out robots, with no duplicate descriptors. Each training step draws fresh joint configurations and poses. [Section III-B](https://arxiv.org/html/2609.29908v1#S3.SS2)
- [x] Review the procedural recipe in `morphik/data.py`. Proposals draw link lengths from Dirichlet(2), zero segments with probability 35%, and use the 25/55/20 parallel, perpendicular, and uniform axis mixture. About 30% of 7–9 DoF chains get compact two-joint wrists. Arm joint half-ranges are uniform on [0.9, π], wrist half-ranges on [0.15, 0.9], and the TCP rotation is uniform on SO(3). The rank filter changes the final link and axis proportions, tabulated below.
- [x] Check the conventions in `morphik/geometry.py`. The first axis becomes +z, subsequent axis points are projected from the previous point onto each joint axis, the first off-axis point fixes +x, and the chain including the TCP has unit length. Verified invariance to axis-point shifts, axis sign reversals, and rigid base transforms, including the parallel-axis and axial-offset fallback cases.
- [x] Check the online triplets in `morphik/data.py`. Sampled joint angles stayed within their limits, padded angles were zero, accepted configurations passed the capsule check, and recomputing FK recovered each target pose.
- [x] Check `morphik/kinematics.py` against independent matrix FK on a Franka Panda URDF with fixed transforms and asymmetric limits. The pose agreed within numerical tolerance, both Jacobian components matched finite differences, and FK passed autograd gradcheck.

## 2. Review the model and loss

- [x] Check the inputs in `morphik/model.py`. Each joint token has 10 features (3 for the axis point, 3 for direction, 2 for limits, 1 for capsule radius, and 1 for chain position). The TCP and query tokens each have 7 features (position and a wxyz quaternion).
- [x] Check the paper-sized model using a batch with mixed DoFs. The encoder receives 11 token slots, masks unused joint slots, and runs four bidirectional ReZero Transformer blocks of width 256. The flattened output and raw 7D query give `11 × 256 + 7 = 2,823` conditioning values. The flow head has six 512-wide AdaLN blocks and a shared readout to nine velocities.
- [x] Check padding invariance with nonzero attention and conditioning weights. Changing padded geometry or flow state preserved all active velocities. Padded outputs stayed zero.
- [x] Check the equations. The flow uses `x_t = (1 - t)x_0 + tq` and regresses `q - x_0`. The FK auxiliary loss uses `q_hat = x_t + (1 - t)v`. A numerical check confirmed the Smooth L1 terms, quaternion sign invariance, the `t >= 0.5` gate and `t²` scaling, and the 6:1 and 5:1 weights. [Sections III-C and III-D](https://arxiv.org/html/2609.29908v1#S3.SS3)
- [x] Check inference in `morphik/solver.py`. Verified one encoder call, 10 Euler steps, ranking of 16 candidates, and bounded outputs before and after 1, 3, and 5 DLS steps.

## 3. Run the small end-to-end check

Run from the repository root with Python 3.10+ and PyTorch 2.5+. The workstation's `dinov3` Conda environment has Python 3.11 and PyTorch 2.8. You can use `conda run --no-capture-output -n dinov3 python` in place of `python`.

```sh
python -m unittest discover -s tests -v

python -m morphik generate --robots-per-dof 24 --holdout-per-dof 4 \
  --output artifacts/check/robots.pt

python -m morphik train --bank artifacts/check/robots.pt \
  --output artifacts/check/model.pt --tiny --steps 20 --batch-size 16 --device cpu

python -m morphik evaluate --bank artifacts/check/robots.pt \
  --checkpoint artifacts/check/model.pt --poses-per-robot 2 --device cpu
```

- [x] The generator reported 80 training and 16 held-out robots. The checkpoint saved the model, EMA weights, optimizer state, and batch RNG state.
- [x] Training reported finite flow and gated FK losses. Evaluation ran all 32 requested held-out targets with zero skips and produced mean, median, and p99 errors overall and for each DoF.
- [x] Restarted at step 21 with `--resume artifacts/check/model.pt --steps 25`. The model, EMA, optimizer, and batch RNG matched an uninterrupted 25-step run bit for bit.
- [x] Verified `solve --help` and solved a known reachable pose on a Panda URDF. The CLI returned the original joint names and angles, respected asymmetric bounds even for forced saturated predictions, and correctly reversed centering and axis flips. Inputs use `--position X Y Z` in meters and `--quaternion W X Y Z`.

## 4. Check reproduction fidelity

- [x] Review the reconstruction choices where the paper leaves details open. The generator samples inter-joint link directions in the next axis's normal plane, scales the last two wrist segments by 0.2, and chooses normalized, tapered capsule radii capped near short links. It ignores adjacent capsules and collapses zero-length segments for collision checks. Training resamples rejected robot/configuration draws, so the accepted distribution favors robots with more collision-free joint space.
- [x] Review the network and training choices. This implementation uses eight attention heads, a 4× Transformer FFN, sinusoidal time features, standard Gaussian noise in radians, AdamW, and EMA decay 0.999. DLS uses 0.02 damping, a 0.25 orientation weight, and a 0.5 rad maximum joint increment per step. The URDF importer approximates capsule radii from canonical link lengths.
- [x] Verify the paper-scale setup. `python -m morphik generate` uses 400,000 geometries. `python -m morphik train --device cuda --amp` defaults to 1,000,000 steps at batch size 2,048. The learning rate stays at 5.4e-4 for 350,000 steps and anneals to 8e-6. The full bank was generated, a single full-size update on an RTX 5090 succeeded with finite losses, and its checkpoint evaluated all 16 CUDA smoke targets. [Section III-E](https://arxiv.org/html/2609.29908v1#S3.SS5)
- [ ] Compare a full-size trained checkpoint to Table II on held-out robots. The paper reports mean raw errors of 3.671% of reach and 2.84° across synthetic holdouts, and 0.136% and 0.07° after three DLS steps. The short run above checks execution. The paper's separate real-robot results use 44 chains, including Shadow Hand wrist variants. [Table II](https://arxiv.org/html/2609.29908v1#S5.T2)

## Review findings

The original generator gave 42.7% of 2,398 sampled 6-DoF chains a deficient Jacobian at every one of eight sampled poses. A zero-length link between parallel axes can make two revolute joints share the same axis. The generator now requires the smallest Jacobian singular value to exceed `1e-7` times the largest at one of up to three sampled poses. This is a reconstruction choice based on Table I's finite condition numbers. In an independent audit of 2,398 chains per DoF from the new bank, every chain attained full rank at one of 32 sampled poses.

The filter shifts the accepted distributions, especially for 6 DoF. Three-consecutive-parallel-axis patterns are still more frequent than in the paper's Table I. The bank should be regenerated for training with the rank filter. [Table I](https://arxiv.org/pdf/2609.29908)

| DoF | Zero-length links | Parallel transitions | Chains with three consecutive parallel axes | Paper, three parallel |
| --- | ---: | ---: | ---: | ---: |
| 6 | 26.1% | 17.6% | 9.8% | 5% |
| 7 | 30.6% | 22.6% | 19.7% | 9% |
| 8 | 32.6% | 24.3% | 26.9% | 12% |
| 9 | 33.4% | 24.8% | 31.6% | 13% |

The Panda check also exposed a float32 rounding error of roughly `5e-8` rad at a joint limit. The solve command now clamps converted output to the original URDF bounds.

Mean errors from the 20-step CPU smoke run show that the refinement path works on the held-out targets.

| Refinement | Position (% of reach) | Orientation (degrees) |
| --- | ---: | ---: |
| Raw | 35.14 | 70.37 |
| 1 DLS step | 29.15 | 49.67 |
| 3 DLS steps | 17.71 | 25.98 |
| 5 DLS steps | 15.05 | 22.79 |
