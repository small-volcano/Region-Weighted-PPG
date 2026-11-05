# Region-Weighted PPG

Training code for **Regional Risk Allocation for PPG-Based Blood Pressure Estimation under Shifting Evaluation Distributions**.

## Method

The primary experiment trains a lightweight one-dimensional CNN to predict systolic and diastolic blood pressure from photoplethysmography (PPG). A normalized Smooth-L1 objective gives additional weight to examples in a label-defined priority region. The union region is defined by systolic pressure at least 140 mmHg **or** diastolic pressure at least 90 mmHg. The primary configuration uses weight 5 inside this region and weight 1 elsewhere. Training labels determine the weights; the model input is a single PPG channel.

The same model also supports ordinary empirical risk minimization (`erm`), systolic-only and diastolic-only region definitions. The original primary comparison uses the same architecture and training budget for each objective.

## Installation

Python 3.10 or later. Install PyTorch for the desired CPU or CUDA environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data

Obtain the PulseDB VitalDB training subset from the [PulseDB repository](https://github.com/pulselabteam/PulseDB) and its original providers. The input is the MATLAB v7.3 training file `VitalDB_Train_Subset.mat`, with `Subset/Signals`, `Subset/SBP`, and `Subset/DBP`. The recorded primary setup uses 10-second windows at 125 Hz and PPG signal channel index 1. Each PPG window is standardized independently; target mean and standard deviation are calculated from the supplied training file.

The data remain subject to the original providers' access terms and licenses.

## Train

```bash
python scripts/train.py --train /path/to/VitalDB_Train_Subset.mat --out runs/union_s42 --region union --seed 42 --device cuda
python scripts/train.py --train /path/to/VitalDB_Train_Subset.mat --out runs/erm_s42 --region erm --seed 42 --device cuda
```

Replace `--seed 42` with `43` or `44` for the original three-seed primary runs. For region-definition experiments, use `--region sbp` or `--region dbp`. Set the priority weight with `--weight`; edit a copy of `configs/primary.json` and pass `--config` to change the training thresholds or budget.

The output directory receives a final-epoch `model.pt`, `config.json`, and `history.json`. The checkpoint includes target normalization statistics and the training configuration. Each run uses a new output directory.

## Experiment scope

The implementation preserves the primary lightweight CNN architecture, label standardization, per-window PPG standardization, normalized weighted Smooth-L1 objective, AdamW settings, and final-epoch training budget. See [training protocol](docs/TRAINING.md) for configuration details and correspondence with the archived experiments. Hardware, framework version, and GPU kernels can affect numerical results.

## Citation

```bibtex
@misc{bu_region_weighted_ppg,
  title={Regional Risk Allocation for PPG-Based Blood Pressure Estimation under Shifting Evaluation Distributions},
  author={Bu, Can and Yang, Zhiyuan and Fan, Peiyu and Sun, Zhiyuan and Huang, YongMing},
  note={Manuscript}
}
```

## License

Code is released under the MIT License. Dataset licenses remain with the respective providers.
