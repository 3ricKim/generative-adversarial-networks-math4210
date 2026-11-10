# Findings from the completed local experiment matrix

All 16 baselines and the single learning-rate follow-up completed their configured budgets. Point models trained on CPU and image models on Apple MPS. Every dataset has a final checkpoint, per-step logs, fixed-latent previews, comparison/progress/loss figures, and source/environment records. The runner regenerated the figures from saved checkpoints without repeating baseline training.

## Simulated distributions

- Gaussian runs recover a broad centered cloud. Final SWD values are approximately 0.140, 0.082, and 0.085, compared with a real-real reference of 0.036. This is approximate agreement rather than exact density recovery.
- Circle seeds 42 and 44 learn an approximate ring, with SWD near 0.030. Seed 43 remains concentrated in a narrow region, with final SWD 0.729. Its saved stages show persistent failure rather than successful recovery followed by a late collapse.
- The Gaussian mixture meets the coverage threshold in 8, 6, and 8 modes across the three seeds. However, in-mode fractions are 86.8%, 9.9%, and 81.3%. Seed 43 produces a diffuse cloud, demonstrating why coverage alone is insufficient.
- Sine runs follow the nonlinear curve. Their generated bands appear thinner than the reference noise band. Learning the curve's shape does not establish recovery of its full noise distribution.

Distances use 2,048 samples and 128 fixed projection directions. Compare them within each dataset. Three seeds illustrate variability without establishing a statistical reliability guarantee.

## Image datasets

MNIST samples show recognizable digits with varied strokes and some ambiguous glyphs. Fashion-MNIST samples show clothing and footwear silhouettes with noisy textures and imperfect boundaries. KMNIST reproduces broad stroke patterns, while correct character identity and legibility remain unvalidated. CIFAR-10 develops varied colors, backgrounds, and object-like forms, with blurred or distorted structures in many samples.

These are qualitative observations from fixed sample grids, not validated measures of class coverage, fidelity, or memorization. Each image experiment uses one seed and 20 epochs. The final checkpoint remains the primary comparison, regardless of whether an earlier checkpoint looks preferable.

## Controlled follow-up

For Gaussian-mixture seed 42, reducing learning rate from 0.0002 to 0.0001 gives final SWD 0.0931 versus 0.0873, with all eight modes meeting the threshold in both runs. In-mode fraction falls from 86.8% to 80.9%. This one paired result does not support a general claim that either learning rate is superior.

## Verification and review

The 19-test suite passes, including finite and isolated gradients, dataset preprocessing, metric checks, exact CPU continuation, MPS continuation within tolerance, durable-copy recovery, and incomplete/stale-result handling. Source-archive installation was also checked as a normal wheel. Every dataset passed a smoke experiment before its baseline training job.

Group members should review the implementation and evidence, reproduce selected runs, and distinguish observed behavior from hypotheses about its causes. The workflow and research responsibilities are in docs/PROJECT_GUIDE.md.
