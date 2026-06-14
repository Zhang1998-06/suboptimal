# Suboptimal Bootstrap Training Code

This repository contains training source code and run instructions only. It does
not include evaluation scripts, checkpoints, CSV/JSON result files, plots,
videos, notebooks, or generated caches.

## Contents

- `scripts/continuous_bootstrap.py`: trains the continuous Suboptimal Bootstrap model.
- `scripts/continuous_vanilla_sac.py`: trains the continuous vanilla SAC backbone.
- `scripts/continuous_dagger.py`: trains the continuous DAgger baseline.
- `scripts/Dscontroller.py`: rule/controller used by Suboptimal Bootstrap and DAgger.
- `scripts/highway_env/`: local environment package used by the training scripts.
- `scripts/agents/hdqn_mdp.py`: discrete SAC-style model class used by the discrete backbone.
- `scripts/dqnmodel.py`: vanilla discrete SAC-style model class.

## Environment

The experiments were run with Python 3.8. A minimal setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from `scripts/` so the local `highway_env` package is imported:

```bash
cd scripts
export MPLCONFIGDIR=/tmp/mplconfig
```

## Train Continuous Suboptimal Bootstrap

This is the 5 Hz, 75 s horizon setting:

```bash
python continuous_bootstrap.py \
  --duration 375 \
  --offline-episodes 50 \
  --bc-steps 6000 \
  --train-episodes 2000 \
  --log-interval 150 \
  --seed 41 \
  --model-path continuous_results/suboptimal_bootstrap_controller.pth \
  --stats-path continuous_results/suboptimal_bootstrap_train_stats.csv \
  --device cpu
```

## Train Continuous Vanilla SAC Backbone

Matched backbone setting, without offline data, BC, KL/bootstrap terms, expert
blend, reward KL penalty, or random warm-up:

```bash
python continuous_vanilla_sac.py \
  --duration 375 \
  --train-episodes 400 \
  --seed 41 \
  --random-steps 0 \
  --log-interval 150 \
  --checkpoint-selection final \
  --model-path continuous_results/vanilla_sac_matched400_controller.pth \
  --stats-path continuous_results/vanilla_sac_matched400_train_stats.csv \
  --device cpu
```

For a longer fixed-budget SAC run, change `--train-episodes` while keeping
`--checkpoint-selection final` if SAC should not be selected by its own best
checkpoint.

## Train Continuous DAgger Baseline

```bash
python continuous_dagger.py \
  --offline-episodes 50 \
  --initial-bc-steps 6000 \
  --dagger-iterations 6 \
  --dagger-episodes 15 \
  --dagger-bc-steps 2500 \
  --seed 41 \
  --device cpu
```

## Artifacts

Training writes checkpoints and optional training statistics under ignored
paths such as `scripts/continuous_results/`. These files are intentionally not
part of the repository and should not be committed.

## Notes

- This PR is source-only. Reproduce checkpoints by running the training scripts.
- CUDA is optional. Use `--device cpu` for portability if GPU drivers are not
  configured.
- The internal accepted Suboptimal checkpoint was selected after episode 300 and
  before episode 450. If an exact matched-budget comparison is required, rerun
  with episode-indexed checkpoint saving enabled and train the SAC backbone for
  the same number of episodes.
