# MAHALO: Unifying Offline Reinforcement Learning and Imitation Learning from Observations
Source code for the paper [MAHALO: Unifying Offline Reinforcement Learning and Imitation Learning from Observations](https://arxiv.org/abs/2303.17156) from ICML 2023.

### Installation

Installing MuJoCo. Our repo uses MuJoCo 210. If you don't have it installed on your machine, this can be done by running the following script.
```
. install.sh
```

Installing this repo.
```
conda create -n mahalo python=3.7
conda activate mahalo
pip install -e .
```

### Running
To run MAHALO:
```
python main.py --env {ENV} --scenario {SCENARIO} --log_dir mahalo_results/ --beta {BETA} --alpha_beta_ratio 100000 --clip_v --remove_terminals
```
Options
- `--env`: `hopper-all-v2`, `walker2d-all-v2`, `halfcheetah-all-v2`
- `--scenario`: `il`, `ilfo`, `rl_expert`, `rlfo`, `rl_sample`

*Note: our Meta-World experiments rely on data and code that are not yet open-sourced. We will update this repo once that process is finished.

To run baseline algorithms:
```
python baselines.py --env {ENV} --scenario {SCENARIO} --algo {ALGO} --log_dir baseline_results/ --beta {BETA} --clip_v --remove_terminals
```
Options
- `--env`: `hopper-all-v2`, `walker2d-all-v2`, `halfcheetah-all-v2`
- `--scenario`: `il`, `ilfo`, `rl_expert`, `rlfo`, `rl_sample`
- `--algo`:
  - `rp`: reward prediction (RP as in paper)
  - `ap`: action prediction (AP as in paper)
  - `arp`: reward-action prediction
  - `uds`: [UDS](https://proceedings.mlr.press/v162/yu22c.html) implemented by ATAC (UDS as in paper)
  - `uds-a`: variant of [UDS](https://proceedings.mlr.press/v162/yu22c.html) without knowing the common transitions between dynamics and reward datasets (UDS-A as in paper)
  - `common`: running ATAC on common transitions (ATAC as in paper)
  - `oracle`: running ATAC with full action and reward information (Oracle as in paper)

*Note: BC and BCO results can be obtained by pretraining of `common` and `ap`, respectively.

### Citing
If you use this repo, please cite:
```
@inproceedings{li2023mahalo,
  title={MAHALO: Unifying Offline Reinforcement Learning and Imitation Learning from Observations},
  author={Li, Anqi and Boots, Byron and Cheng, Ching-An},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```