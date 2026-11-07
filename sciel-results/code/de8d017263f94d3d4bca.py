import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = 300.0  # Safety margin before 330s kill
    eps = 1e-9
    capacity = 25.0

    # Handle potential shape issues
    if prizes.ndim == 2:
        P = prizes.sum(axis=0)          # (100,)
    else:
        P = prizes

    if weights.ndim == 2:
        W = weights
    else:
        W = weights.reshape(1, -1)

    n_constraints, n_items = W.shape

    # ---- Component 1: per-constraint efficiency (current program) ----
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = P[None, :] / (W + eps)    # (5, 100)
    eff = np.where(np.isfinite(eff), eff, 0.0)

    # ---- Component 2: geometric-mean bottleneck (inspiration 1) ----
    # Smooth min: an item is limited by its worst constraint.
    with np.errstate(divide='ignore', invalid='ignore'):
        log_eff = np.log(eff + eps)
    log_eff = np.where(np.isfinite(log_eff), log_eff, np.log(eps))
    geo_mean = np.exp(log_eff.mean(axis=0))  # (100,)
    geo_mean = np.where(np.isfinite(geo_mean), geo_mean, eps)

    # ---- Component 3: most-binding-constraint scarcity premium (inspiration 3) ----
    # The constraint where this item consumes the largest fraction of total weight.
    total_weight_per_item = W.sum(axis=0)  # (100,)
    weight_fractions = W / (total_weight_per_item + eps)  # (5, 100)
    most_binding_idx = np.argmax(weight_fractions, axis=0)  # (100,)
    item_indices = np.arange(n_items)
    binding_weights = W[most_binding_idx, item_indices]  # (100,)
    remaining_fraction = (capacity - binding_weights) / capacity
    remaining_fraction = np.maximum(remaining_fraction, eps)
    scarcity_premium = 1.0 / remaining_fraction
    binding_ratios = eff[most_binding_idx, item_indices]  # (100,)
    scarcity_adjusted = binding_ratios * scarcity_premium  # (100,)
    scarcity_adjusted = np.where(np.isfinite(scarcity_adjusted), scarcity_adjusted, 0.0)

    # ---- Component 4: constraint-level scarcity weighting (current program) ----
    k = max(1, n_items // 5)  # top 20% of items
    top_k_eff = np.zeros(n_constraints)
    for c in range(n_constraints):
        top_k_eff[c] = np.partition(eff[c], -k)[-k].mean()
    scarcity = top_k_eff / (top_k_eff.sum() + eps)
    # Marginal utility: value weighted by how scarce each constraint is.
    marginal_util = (scarcity[:, None] * eff).sum(axis=0)  # (100,)
    marginal_util = np.where(np.isfinite(marginal_util), marginal_util, 0.0)

    # Base efficiency ratio (general prize-per-total-weight)
    W_sum = W.sum(axis=0)
    base_eff = P / (W_sum + eps)
    base_eff = np.where(np.isfinite(base_eff), base_eff, 0.0)

    # ---- Blend all components coherently ----
    # Normalize each to mean 1 so blending weights are meaningful.
    def norm(x):
        m = np.mean(x)
        x = np.where(np.isfinite(x), x, 0.0)
        return x / (m + eps)

    n_marginal = norm(marginal_util)
    n_geo = norm(geo_mean)
    n_scarcity = norm(scarcity_adjusted)
    n_base = norm(base_eff)

    # Weights: emphasize marginal utility (scarce-constraint value) and the
    # bottleneck geometric mean (robustness), with scarcity premium as a
    # tie-breaker and base efficiency as a stable floor.
    H = (0.40 * n_marginal + 0.30 * n_geo + 0.20 * n_scarcity + 0.10 * n_base)

    # Ensure positive and finite
    H = np.where(np.isfinite(H), H, eps)
    H = np.maximum(H, eps)

    # Deadline guard
    if time.time() > deadline - 5.0:
        return H

    return H
