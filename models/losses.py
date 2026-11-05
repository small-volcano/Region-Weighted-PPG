"""Label-defined normalized weighting for two-output regression."""

import torch
import torch.nn.functional as F


def region_mask(y_raw, region, sbp_threshold, dbp_threshold):
    sbp = y_raw[:, 0] >= sbp_threshold
    dbp = y_raw[:, 1] >= dbp_threshold
    if region == "sbp":
        return sbp
    if region == "dbp":
        return dbp
    if region == "union":
        return sbp | dbp
    if region == "erm":
        return torch.zeros_like(sbp, dtype=torch.bool)
    raise ValueError(f"Unknown region: {region}")


def weighted_loss(prediction, target, target_mean, target_std, region, weight,
                  sbp_threshold=140.0, dbp_threshold=90.0):
    per_example = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=1)
    labels = target * target_std + target_mean
    high = region_mask(labels, region, sbp_threshold, dbp_threshold)
    weights = torch.where(high, weight, 1.0)
    return (per_example * weights).sum() / weights.sum()
