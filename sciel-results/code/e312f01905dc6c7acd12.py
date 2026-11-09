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
    best = heuristic[feasible]
    if best.size == 0:
        return 0.0
    best = np.where(feasible, heuristic, -np.inf)
    best = np.argmax(best)
    total = 0.0
    while True:
        if remaining[best] < 0:
            return total
        total += prizes[:, best].sum()
        remaining = np.minimum(remaining, capacity - weights[:, best])
        feasible = remaining >= 0
        b = np.where(feasible, heuristic, -np.inf)
        if not np.any(np.isfinite(b)):
            return total
        best = int(np.argmax(b))


def _evaluate_heuristic(heuristic, prizes, weights, capacity, n_items, n_cons):
    """Expected total prize of a heuristic = mean prize over several simulated ants."""
    s = 0.0
    for _ in range(4):
        s += _simulate_ant(heuristic, prizes, weights, capacity, n_items, n_cons)
    return s / 4.0


def _compute_heuristics(prizes, weights, deadline):
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n_cons, n_items = prizes.shape

    total_prize = prizes.sum(axis=0)          # (N,)
    sum_w = weights.sum(axis=0)               # (N,)
    max_w = weights.max(axis=0)               # (N,)
    eps = 1e-9

    # Build a diverse pool of candidate heuristics (different modelling families).
    pool = [
        total_prize / (sum_w + eps),
        total_prize / (max_w + eps),
        total_prize / (sum_w + eps) * (1.0 / (1.0 + 1.0 * max_w)),
        (total_prize / (sum_w + eps)) ** 1.5,
        (total_prize / (max_w + eps)) ** 1.5,
        total_prize ** 2 / (sum_w + eps),
        total_prize / (sum_w + eps) * (total_prize + eps),
        1.0 / (sum_w + eps) * total_prize * total_prize,
        total_prize / (sum_w + eps) * np.exp(-0.5 * max_w),
        total_prize / (sum_w + eps) * np.exp(-0.5 * sum_w),
        total_prize / (max_w + eps) * np.exp(-0.5 * sum_w),
        (total_prize / (sum_w + eps)) * (total_prize / (sum_w + eps)) / (total_prize / (sum_w + eps) + eps),
    ]

    # Also add a couple of dominance-boosted variants.
    dom = np.zeros(n_items, dtype=np.int32)
    for i in range(n_items):
        if time.time() > deadline:
            break
        wi = weights[:, i]
        prize_ok = total_prize <= total_prize[i]
        weight_ok = np.all(wi <= weights.T, axis=1)
        cands = np.where(prize_ok & weight_ok)[0]
        for j in cands:
            if i == j:
                continue
            if total_prize[i] > total_prize[j] or np.any(wi < weights[:, j]):
                dom[j] += 1
    dom_factor = 1.0 / (1.0 + 2.0 * dom)
    pool.append(pool[0] * dom_factor)
    pool.append(pool[1] * dom_factor)

    # Score every candidate by the prize it actually yields in the ACO process.
    best_h = pool[0]
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
            best_h = hh

    # Light local refinement: try scaling the best heuristic by a few factors.
    for mult in (0.5, 0.75, 1.25, 2.0, 3.0):
        if time.time() > deadline:
            break
        cand = best_h * mult
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

        deadline = start_time + 295.0  # safety margin before the 330s kill
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
