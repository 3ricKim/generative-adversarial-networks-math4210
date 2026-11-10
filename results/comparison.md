# Experiment comparison

Baseline complete: **True**

Smoke tests and partial or stale evaluations do not count toward completion.

| Dataset | LR | Seeds | Device | SWD mean ± sample SD | Real-real SWD | Runtime (min) |
|---|---:|---|---|---:|---:|---:|
| cifar10 | 0.0002 | [42] | mps | Qualitative review | — | 14.3 |
| circle | 0.0002 | [42, 43, 44] | cpu | 0.2630 ± 0.4038 | 0.0235 | 0.3 |
| fashion_mnist | 0.0002 | [42] | mps | Qualitative review | — | 17.4 |
| gaussian | 0.0002 | [42, 43, 44] | cpu | 0.1020 ± 0.0328 | 0.0356 | 0.3 |
| gaussian_mixture | 0.0001 | [42] | cpu | 0.0931 | 0.0214 | 0.1 |
| gaussian_mixture | 0.0002 | [42, 43, 44] | cpu | 0.1520 ± 0.0971 | 0.0214 | 0.3 |
| kmnist | 0.0002 | [42] | mps | Qualitative review | — | 14.8 |
| mnist | 0.0002 | [42] | mps | Qualitative review | — | 14.7 |
| sine | 0.0002 | [42, 43, 44] | cpu | 0.0628 ± 0.0128 | 0.0350 | 0.3 |

## Remaining baselines


## Interpretation

Compare SWD within a dataset. Read observations.md alongside image grids. Loss magnitude alone does not establish sample quality. Three synthetic seeds and one image seed limit generalization. Runtime includes previews and checkpoint writes, excluding downloads and post-training evaluation. Hardware and concurrent workloads differ, so runtimes do not isolate dataset complexity.
