# MorphIK implementation checklist

This checklist follows Section III and Figures 3–4 of Clasmeier et al., *MorphIK: Morphology-Conditioned Neural Inverse Kinematics for Unknown Robots*. I found no public author implementation in searches of arXiv, GitHub, and the authors' lab publications. The code in this workspace is an independent PyTorch implementation. [Paper](https://arxiv.org/abs/2609.29908) · [Lab publications](https://wtmbib.informatik.uni-hamburg.de/Basilic/)

The boxes below are for your verification.

## 1. Review the training data

- [ ] Confirm that the training and holdout splits contain different robot identities. The paper's counts are 100,000 robots per DoF class (6, 7, 8, and 9), with 200 per class held out. The bank contains geometry and joint limits. Each training step draws fresh joint configurations and poses. [Section III-B](https://arxiv.org/html/2609.29908v1#S3.SS2)
- [ ] Review the procedural recipe in `morphik/data.py`. Link lengths start from Dirichlet(2) draws with a 35% chance of zeroing each segment. Successive joint axes are 25% parallel, 55% perpendicular, and 20% uniform on the sphere. About 30% of 7–9 DoF chains get compact two-joint wrists. Arm joint half-ranges are uniform on [0.9, π], wrist half-ranges on [0.15, 0.9], and the TCP rotation is uniform on SO(3).
- [ ] Check the conventions in `morphik/geometry.py`. The first axis becomes +z, subsequent axis points are projected from the previous point onto each joint axis, the first off-axis point fixes +x, and the chain including the TCP has unit length. Verify that shifting an axis point along its own axis or rigidly moving the entire robot gives the same canonical geometry within tolerance.
- [ ] Check the online triplets in `morphik/data.py`. Every active joint angle lies within its limits, padded angles are zero, accepted configurations pass the capsule self-collision check, and recomputing FK gives the stored target position and orientation. The query is therefore reachable by construction.
- [ ] Check `morphik/kinematics.py` against an independently computed FK pose. Include a chain with fixed transforms and a nonzero joint-limit midpoint. Compare the geometric Jacobian with finite differences and check that gradients through FK are finite.

## 2. Review the model and loss

- [ ] Check the inputs in `morphik/model.py`. Each joint token has 10 features (3 for the axis point, 3 for direction, 2 for limits, 1 for capsule radius, and 1 for chain position). The TCP and query tokens each have 7 features (position and a wxyz quaternion).
- [ ] Check the paper-sized model using a batch with mixed DoFs. The encoder receives 11 token slots, masks unused joint slots, and runs four bidirectional ReZero Transformer blocks of width 256. The flattened output and raw 7D query give `11 × 256 + 7 = 2,823` conditioning values. The flow head has six 512-wide AdaLN blocks and a shared readout to nine velocities.
- [ ] Check padding invariance. Changing the values in a masked joint slot leaves the active output unchanged, flow-state noise in padded joints is ignored, and padded output velocities are zero.
- [ ] Check the equations. The flow uses `x_t = (1 - t)x_0 + tq` and regresses `q - x_0`. The FK auxiliary loss uses `q_hat = x_t + (1 - t)v`. It combines position Smooth L1 with the smaller of the two quaternion-sign Smooth L1 errors. The FK term switches on at `t >= 0.5` and is scaled by `t²`, with FK-to-flow weights of 6:1 and position-to-orientation weights of 5:1. [Sections III-C and III-D](https://arxiv.org/html/2609.29908v1#S3.SS3)
- [ ] Check inference in `morphik/solver.py`. Encode each query once, integrate noise with 10 explicit Euler steps, and rank 16 candidates by pose error. Compare the winning candidate before and after 1, 3, and 5 DLS steps. Check returned angles against the joint limits.

## 3. Run the small end-to-end check

Run from the repository root with Python 3.10+ and PyTorch 2.5+. The workstation's `dinov3` Conda environment has Python 3.11 and PyTorch 2.8. You can use `conda run --no-capture-output -n dinov3 python` in place of `python`.

```sh
python -m morphik generate --robots-per-dof 24 --holdout-per-dof 4 \
  --output artifacts/check/robots.pt

python -m morphik train --bank artifacts/check/robots.pt \
  --output artifacts/check/model.pt --tiny --steps 20 --batch-size 16 --device cpu

python -m morphik evaluate --bank artifacts/check/robots.pt \
  --checkpoint artifacts/check/model.pt --poses-per-robot 2 --device cpu
```

- [ ] The generator reports 80 training and 16 held-out robots. The checkpoint saves the model, EMA weights, optimizer state, and batch RNG state.
- [ ] Training reports finite flow and gated FK losses. The evaluator loads the EMA weights and reports mean, median, and p99 position (% of reach) and orientation (degrees), grouped by DoF and pooled. Check the skipped-pose count alongside the 32 requested targets.
- [ ] If checking continuation, run the same training command with `--resume artifacts/check/model.pt --steps 25` in place of `--steps 20`. Training picks up at step 21.
- [ ] Try `python -m morphik solve --help`, then supply an expanded URDF, its tip link, a checkpoint, and a known reachable base-frame target using `--position X Y Z` (meters) and `--quaternion W X Y Z`. Check joint names and original URDF angles in the output. For a URDF with asymmetric limits, verify the centered coordinates and joint-axis flips are correctly reversed.

## 4. Check reproduction fidelity

- [ ] Review the reconstruction choices where the paper leaves details open. The generator samples inter-joint link directions in the next axis's normal plane, scales the last two wrist segments by 0.2, and chooses normalized, tapered capsule radii capped near short links. It ignores adjacent capsules and collapses zero-length segments for collision checks. Training resamples rejected robot/configuration draws, so the accepted distribution favors robots with more collision-free joint space.
- [ ] Review the network and training choices. This implementation uses eight attention heads, a 4× Transformer FFN, sinusoidal time features, standard Gaussian noise in radians, AdamW, and EMA decay 0.999. DLS uses 0.02 damping, a 0.25 orientation weight, and a 0.5 rad maximum joint increment per step. The URDF importer approximates capsule radii from canonical link lengths.
- [ ] For paper-scale training, use `python -m morphik generate` followed by `python -m morphik train --device cuda --amp`. The defaults use the paper-sized bank and model, 1,000,000 steps, and batches of 2,048. The learning rate stays at 5.4e-4 for 350,000 steps then anneals to 8e-6. [Section III-E](https://arxiv.org/html/2609.29908v1#S3.SS5)
- [ ] Compare a full-size trained checkpoint to Table II on held-out robots. The paper reports mean raw errors of 3.671% of reach and 2.84° across synthetic holdouts, and 0.136% and 0.07° after three DLS steps. The short run above checks execution. The paper's separate real-robot results use 44 chains, including Shadow Hand wrist variants. [Table II](https://arxiv.org/html/2609.29908v1#S5.T2)
