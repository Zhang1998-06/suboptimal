# Suboptimal Highway Training Code

This repository contains source-only training code for the suboptimal highway
experiments and baseline methods. It intentionally excludes evaluation scripts,
checkpoints, CSV/JSON result files, plots, videos, notebooks, and cache files.

## Included Models

- `suboptimal_controller`: Suboptimal Bootstrap
- `SACend`: vanilla discrete SAC backbone
- `SAC_continuous`: continuous SAC variant
- `sqil`: SQIL
- `dapg`: DAPG
- `ddqgfd`: DDPGfD
- `dqfd`: DQfD
- `TD3_BC`: TD3+BC
- `CQL`: CQL
- `AWAC`: AWAC
- `IQL`: IQL
- `BCQ`: BCQ
- `GAIL`: GAIL
- `BC`: behavior cloning
- `IDMboostrap`, `IDM_MOBIL`, `warmSAC`: additional training variants used in
  the ablation codebase

Each model folder keeps its own copy of the environment and model source under
`highway-env/scripts` because the baseline implementations were developed as
separate experiment directories.

## Setup

Use Python 3.8 if possible.

```bash
pip install -r requirements.txt
```

For headless machines, set:

```bash
export MPLCONFIGDIR=/tmp/mplconfig
```

## Training

Run commands from the target model's `highway-env/scripts` directory.

Suboptimal Bootstrap:

```bash
cd suboptimal_controller/highway-env/scripts
python collect_trajectories.py
python run_hdqn.py
```

Vanilla SAC:

```bash
cd SACend/highway-env/scripts
python run_dqn.py
```

Baseline methods:

```bash
cd sqil/highway-env/scripts && python collect_trajectories.py && python run_hdqn.py
cd dapg/highway-env/scripts && python collect_trajectories.py && python run_hdqn.py
cd ddqgfd/highway-env/scripts && python collect_trajectories.py && python run_hdqn.py
cd dqfd/highway-env/scripts && python collect_trajectories.py && python run_hdqn.py
cd TD3_BC/highway-env/scripts && python collect_trajectories.py && python run_hdqn.py
cd CQL/highway-env/scripts && python collect_trajectories.py && python run_hdqn.py
cd AWAC/highway-env/scripts && python collect_trajectories.py && python run_hdqn.py
cd IQL/highway-env/scripts && python collect_trajectories.py && python run_hdqn.py
cd BCQ/highway-env/scripts && python collect_trajectories.py && python run_hdqn.py
cd GAIL/highway-env/scripts && python collect_trajectories.py && python GAIL.py
cd BC/highway-env/scripts && python collect_trajectories.py && python run_BC.py
```

Training may write checkpoints, logs, or stats locally depending on the script.
Those generated files are ignored by Git and should not be committed.
