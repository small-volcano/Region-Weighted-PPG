"""Train the primary PPG regressor from a PulseDB MATLAB v7.3 training file."""

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import PPGEncoder
from models.losses import weighted_loss


class PPGDataset(Dataset):
    def __init__(self, signals, targets):
        self.signals = signals
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.from_numpy(self.signals[index]), torch.from_numpy(self.targets[index])


def read_training_data(path, channel, samples):
    with h5py.File(path, "r") as f:
        subset = f["Subset"]
        shape = subset["Signals"].shape
        if len(shape) != 3 or shape[0] != samples or not 0 <= channel < shape[1]:
            raise ValueError(f"Expected Signals shape ({samples}, channels, windows); got {shape}")
        sbp = np.asarray(subset["SBP"][0, :], dtype=np.float32)
        dbp = np.asarray(subset["DBP"][0, :], dtype=np.float32)
        keep = np.isfinite(sbp) & np.isfinite(dbp)
        if not keep.any():
            raise ValueError("Training file contains no finite target labels")
        signals = np.asarray(subset["Signals"][:, channel, :], dtype=np.float32).T[keep]
    if not np.isfinite(signals).all():
        raise ValueError("Training waveforms contain non-finite values")
    labels = np.stack((sbp[keep], dbp[keep]), axis=1)
    mean, std = labels.mean(axis=0), labels.std(axis=0)
    if np.any(std <= 0):
        raise ValueError("Both target standard deviations must be positive")
    signals = signals[:, None, :]
    signals = (signals - signals.mean(axis=2, keepdims=True)) / (signals.std(axis=2, keepdims=True) + 1e-6)
    return signals.astype(np.float32), ((labels - mean) / std).astype(np.float32), mean, std


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, help="PulseDB VitalDB training MAT file")
    ap.add_argument("--out", required=True, help="Output directory for a new run")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/primary.json"))
    ap.add_argument("--region", choices=["erm", "sbp", "dbp", "union"])
    ap.add_argument("--weight", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.region is not None:
        cfg["region"] = args.region
    if args.weight is not None:
        cfg["priority_weight"] = args.weight
    if args.seed is not None:
        cfg["seed"] = args.seed
    if cfg["priority_weight"] <= 0:
        raise ValueError("Priority weight must be positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    x, y, mu, sd = read_training_data(args.train, cfg["ppg_channel"], cfg["samples_per_window"])
    dl = DataLoader(PPGDataset(x, y), batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    model = PPGEncoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    tmu = torch.as_tensor(mu, device=device); tsd = torch.as_tensor(sd, device=device)
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            prediction = model(xb)[1]
            loss = weighted_loss(prediction, yb, tmu, tsd, cfg["region"], cfg["priority_weight"],
                                 cfg["sbp_threshold_mmhg"], cfg["dbp_threshold_mmhg"])
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss.detach())
        row = {"epoch": epoch, "mean_batch_loss": total / len(dl)}
        history.append(row); print(json.dumps(row), flush=True)
    torch.save({"state_dict": model.state_dict(), "target_mean": mu.tolist(), "target_std": sd.tolist(),
                "config": cfg, "training_windows": len(y)}, output / "model.pt")
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
