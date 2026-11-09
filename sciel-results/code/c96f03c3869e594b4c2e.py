"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

CAPACITY = 25.0
N_ITEMS = 100


def _simulate_ant(heuristic, prizes, weights, capacity, n_items, n_cons):
    """Simulate a single ant: add items that still fit in every constraint,
    choosing the highest heuristic among the feasible ones (greedy).
    Returns the total prize of the produced solution."""
    remaining = capacity - weights[:, 0]
    for c in range(1, n_cons):
        remaining = np.minimum(remaining, capacity - weights[:, c])
    remaining = np.maximum(remaining, 0.0)
    feasible = remaining >= 0
    if not np.any(feasible):
        return 0.0
    best = np.argmax(np.where(feasible, heuristic, -np.inf))
    total = 0.0
    while True:
        if remaining[best] < 0:
            return total
        total += prizes[:, best].sum()
        remaining = np.minimum(remaining, capacity - weights[:, best])
        feasible = remaining >= 0
        if not np.any(feasible):
            return total
        best = int(np.argmax(np.where(feasible, heuristic, -np.inf)))


def _evaluate_heuristic(heuristic, prizes, weights, capacity, n_items, n_cons):
    """Expected total prize of a heuristic = mean prize over several simulated ants."""
    s = 0.0
    for _ in range(8):
        s += _simulate_ant(heuristic, prizes, weights, capacity, n_items, n_cons)
    return s / 8.0


def _pareto_front(prizes, weights):
    """
    Compute Pareto non-dominated items in the prize-weight space.
    An item i is dominated by j if:
      - prize[j] >= prize[i] (in all constraints)
      - weight[j] <= weight[i] (in all constraints)
      - at least one strict inequality holds
    
    Returns a boolean array where True means non-dominated.
    """
    n_cons, n_items = prizes.shape
    total_prize = prizes.sum(axis=0)
    total_weight = weights.sum(axis=0)
    
    # Sort by total_prize descending, then total_weight ascending
    order = np.lexsort((total_weight, -total_prize))
    
    non_dominated = np.zeros(n_items, dtype=bool)
    max_prize = np.zeros(n_cons)
    min_weight = np.full(n_cons, np.inf)
    
    for idx in order:
        p = prizes[:, idx]
        w = weights[:, idx]
        
        # Check if dominated by any previously seen item
        dominated = False
        if np.all(p <= max_prize) and np.all(w >= min_weight):
            # Check if strictly better in at least one dimension
            if np.any(p < max_prize) or np.any(w > min_weight):
                dominated = True
        
        if not dominated:
            non_dominated[idx] = True
            max_prize = np.maximum(max_prize, p)
            min_weight = np.minimum(min_weight, w)
    
    return non_dominated


def _compute_heuristics(prizes, weights, deadline):
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n_cons, n_items = prizes.shape

    total_prize = prizes.sum(axis=0)          # (N,)
    sum_w = weights.sum(axis=0)               # (N,)
    max_w = weights.max(axis=0)               # (N,)
    eps = 1e-9

    # Compute Pareto front
    pareto_mask = _pareto_front(prizes, weights)
    
    # Build candidate heuristics focusing on max-weight ratio (binding constraint)
    base_ratio_max = total_prize / (max_w + eps)
    base_ratio_sum = total_prize / (sum_w + eps)
    
    # Pareto-boosted variants
    pareto_boost = np.where(pareto_mask, 1.0, 0.2)
    
    # Additional features: per-constraint efficiency
    # For each constraint c, compute prize_c / weight_c
    per_cons_ratio = np.ones(n_items)
    for c in range(n_cons):
        r = prizes[c] / (weights[c] + eps)
        per_cons_ratio = np.minimum(per_cons_ratio, r)
    
    pool = [
        # Max-weight ratio variants (primary focus)
        base_ratio_max,
        base_ratio_max * pareto_boost,
        base_ratio_max ** 1.5,
        base_ratio_max ** 2,
        base_ratio_max * np.exp(-0.2 * max_w),
        base_ratio_max * np.exp(-0.5 * max_w),
        
        # Sum-weight ratio variants
        base_ratio_sum,
        base_ratio_sum * pareto_boost,
        base_ratio_sum ** 1.5,
        
        # Per-constraint bottleneck ratio
        per_cons_ratio,
        per_cons_ratio * pareto_boost,
        per_cons_ratio ** 1.5,
        
        # Combined: max-weight ratio with per-constraint penalty
        base_ratio_max * (per_cons_ratio / (per_cons_ratio.max() + eps)),
        
        # Prize-weighted by inverse max weight
        total_prize * (1.0 / (max_w + eps)) ** 2,
        
        # Geometric mean of ratios
        np.sqrt(base_ratio_max * base_ratio_sum),
    ]
    
    # Score every candidate by the prize it actually yields in the ACO process.
    best_h = pool[0].copy()
    best_score = -np.inf
    for h in pool:
        if time.time() > deadline:
            break
        hh = np.asarray(h, dtype=np.float64)
        hh = np.where(np.isfinite(hh), hh, 1e-9)
        hh = np.maximum(hh, 1e-9)
        sc = _evaluate_heuristic(hh, prizes, weights, CAPACITY, n_items, n_cons)
        if sc > best_score:
            best_score = sc
            best_h = hh.copy()

    # Local refinement: try scaling and mixing with other good candidates
    # First, try simple scaling
    for mult in (0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 4.0):
        if time.time() > deadline:
            break
        cand = best_h * mult
        cand = np.maximum(cand, 1e-9)
        sc = _evaluate_heuristic(cand, prizes, weights, CAPACITY, n_items, n_cons)
        if sc > best_score:
            best_score = sc
            best_h = cand

    # Try mixing best_h with other pool candidates
    for h in pool:
        if time.time() > deadline:
            break
        hh = np.asarray(h, dtype=np.float64)
        hh = np.where(np.isfinite(hh), hh, 1e-9)
        hh = np.maximum(hh, 1e-9)
        for alpha in (0.2, 0.4, 0.6, 0.8):
            if time.time() > deadline:
                break
            cand = alpha * best_h + (1 - alpha) * hh
            cand = np.maximum(cand, 1e-9)
            sc = _evaluate_heuristic(cand, prizes, weights, CAPACITY, n_items, n_cons)
            if sc > best_score:
                best_score = sc
                best_h = cand

    best_h = np.where(np.isfinite(best_h), best_h, 1e-9)
    best_h = np.maximum(best_h, 1e-9)
    return best_h


def heuristics(prizes, weights):
    start_time = time.time()
    try:
        p = np.asarray(prizes, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if p.ndim != 2 or w.ndim != 2:
            raise ValueError("bad shape")
        if p.shape != w.shape:
            raise ValueError("shape mismatch")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(w)):
            raise ValueError("non-finite")
        n_cons, n_items = p.shape

        deadline = start_time + 290.0  # safety margin before the 330s kill
        result = _compute_heuristics(p, w, deadline)
        result = np.asarray(result, dtype=np.float64)
        if result.ndim != 1 or result.shape[0] != n_items:
            raise ValueError("bad result shape")
        if not np.all(np.isfinite(result)):
            raise ValueError("non-finite result")
        return result
    except Exception:
        try:
            p = np.asarray(prizes, dtype=np.float64)
            n_items = p.shape[1] if p.ndim == 2 else len(p)
        except Exception:
            n_items = N_ITEMS
        return np.ones(n_items, dtype=np.float64)
# EVOLVE-BLOCK-END
