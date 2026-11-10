# Experiment workflow

The three-person project uses four simulated distributions and four real datasets. The objective is to understand adversarial training, adapt an existing implementation, run reproducible experiments, and analyze the resulting samples and training behavior.

## Research responsibilities

Member A owns the shared pipeline and theoretical explanation. Member B owns simulated experiments and numerical evaluation. Member C owns image experiments and qualitative comparisons. Each member reviews another member's code and evidence.

## Reproduce the experiments

1. Read docs/THEORY.md and the implementation references in docs/SOURCES.md. Connect the mathematical objectives to the generator/discriminator updates in the code.
2. Install the package and run the tests. Use the tested dependency versions when reproducing the local environment.
3. Run the eight-dataset smoke matrix before starting new full experiments.
4. Run the 16 baseline experiments: three seeds per simulated dataset and one seed per image dataset. Keep configurations fixed when comparing random seeds.
5. After the baseline matrix is complete, repeat Gaussian-mixture seed 42 at learning rate 0.0001. Keep other modeling settings fixed.
6. Regenerate metrics and figures from saved checkpoints, then build the aggregate comparison. Training does not need to repeat to rebuild these outputs.

The local experiment matrix is complete. Use the Colab launcher to reproduce the pipeline with CUDA and save durable checkpoints to Drive.

## Review the evidence

Use final checkpoints for primary comparisons. Label partial, intermediate, and smoke results accurately. Check reported numbers against saved metrics and inspect image grids before making quality claims. Record observations separately from hypotheses about their causes in each run's observations.md.

For simulated datasets, inspect distribution fit, sliced Wasserstein distance, and variation across seeds. For the Gaussian mixture, consider mode occupancy and in-mode fraction together with coverage. For image datasets, discuss visible structure, diversity, and artifacts while acknowledging the limits of qualitative evaluation and single-seed runs.

Review challenges actually encountered, document any configuration changes, and state the limitations of the resulting comparisons. Additional architectures and broader hyperparameter studies are optional extensions.
