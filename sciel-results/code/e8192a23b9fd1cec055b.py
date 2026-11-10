"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0  # seconds wall-clock limit


def heuristics(prizes, weights):
    t0 = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Ensure positive values to avoid division issues
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)

    capacity = 25.0
    n_constraints, n_items = weights.shape

    # Per-constraint efficiency: prize/weight for each constraint
    ratios = prizes / weights

    # Log of ratios for smooth operations
    log_ratios = np.log(ratios)

    # Soft-min across constraints (smooth surrogate for min-ratio)
    neg_log_ratios = -log_ratios
    max_neg = neg_log_ratios.max(axis=0)
    log_sum_exp_neg = max_neg + np.log(np.sum(np.exp(neg_log_ratios - max_neg), axis=0))
    softmin_log = -log_sum_exp_neg

    # Geometric mean of ratios
    log_gm = log_ratios.mean(axis=0)

    # Total prize per item, normalized
    total_prize = prizes.sum(axis=0)
    max_prize = total_prize.max()
    norm_prize = total_prize / (max_prize + 1e-9)

    # Softmin and GM on ratio scale
    softmin_ratio = np.exp(softmin_log)
    gm_ratio = np.exp(log_gm)

    # Bottleneck slack: minimum remaining capacity ratio across all constraints
    slack = (capacity - weights) / weights
    bottleneck_slack = slack.min(axis=0)
    min_slack = bottleneck_slack.min()
    max_slack = bottleneck_slack.max()
    if max_slack > min_slack:
        norm_slack = (bottleneck_slack - min_slack) / (max_slack - min_slack)
    else:
        norm_slack = np.ones_like(bottleneck_slack)

    # Per-constraint "how much of capacity this item uses"
    frac_used = weights / capacity
    max_frac = frac_used.max(axis=0)
    min_frac = max_frac.min()
    max_frac_max = max_frac.max()
    if max_frac_max > min_frac:
        norm_frac_inv = 1.0 - (max_frac - min_frac) / (max_frac_max - min_frac)
    else:
        norm_frac_inv = np.ones_like(max_frac)

    # Feasibility margin
    feasibility_margin = norm_slack * norm_frac_inv

    # Greedy regret score: for each item, compute the drop in total prize
    # of a baseline greedy solution if that item is forced to be excluded.
    # This explicitly models blocking interactions.
    regret = np.zeros(n_items, dtype=np.float64)
    
    # Baseline greedy solution: repeatedly pick item with best total ratio that fits
    remaining_cap_base = np.full(n_constraints, capacity, dtype=np.float64)
    selected_base = np.zeros(n_items, dtype=bool)
    total_prize_base = 0.0
    
    # Pre-compute greedy scores (sum of ratios across constraints)
    greedy_scores = ratios.sum(axis=0)
    
    for _ in range(n_items):
        if time.time() - t0 > _DEADLINE - 15.0:
            break
        best_idx = -1
        best_val = -1
        for i in range(n_items):
            if selected_base[i]:
                continue
            if np.all(weights[:, i] <= remaining_cap_base + 1e-9):
                val = greedy_scores[i]
                if val > best_val:
                    best_val = val
                    best_idx = i
        if best_idx == -1:
            break
        selected_base[best_idx] = True
        remaining_cap_base -= weights[:, best_idx]
        total_prize_base += total_prize[best_idx]
    
    # For each selected item, compute regret: run greedy with that item excluded
    # and measure the drop in total prize
    for k in range(n_items):
        if time.time() - t0 > _DEADLINE - 10.0:
            break
        if not selected_base[k]:
            continue
        
        remaining_cap_k = np.full(n_constraints, capacity, dtype=np.float64)
        selected_k = np.zeros(n_items, dtype=bool)
        selected_k[k] = True  # Force exclusion
        total_prize_k = 0.0
        
        for _ in range(n_items):
            if time.time() - t0 > _DEADLINE - 10.0:
                break
            best_idx = -1
            best_val = -1
            for i in range(n_items):
                if selected_k[i]:
                    continue
                if np.all(weights[:, i] <= remaining_cap_k + 1e-9):
                    val = greedy_scores[i]
                    if val > best_val:
                        best_val = val
                        best_idx = i
            if best_idx == -1:
                break
            selected_k[best_idx] = True
            remaining_cap_k -= weights[:, best_idx]
            total_prize_k += total_prize[best_idx]
        
        regret[k] = total_prize_base - total_prize_k
    
    # Normalize regret to [0, 1]
    max_regret = regret.max()
    if max_regret > 0:
        norm_regret = regret / max_regret
    else:
        norm_regret = np.zeros_like(regret)
    
    # Combine all components into the final heuristic
    # Emphasize feasibility margin, ratio efficiency, prize, and regret
    heuristic = (
        (0.5 + 0.5 * norm_prize)
        * (0.3 + 0.7 * feasibility_margin)
        * (softmin_ratio * gm_ratio)
        * (0.8 + 0.4 * norm_regret)
    )

    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() - t0 > _DEADLINE - 5.0:
        return prizes.sum(axis=0) / (weights.sum(axis=0) + 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
