# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 145.0  # wall-clock safety margin under the 150s evaluator limit


def _makespan(pt, order):
    """Makespan of a job permutation for a flow shop (vectorized)."""
    return int(np.cumsum(pt[order], axis=0).max())


def _neh(pt, rng):
    """Nawaz-Ener-Ham constructive heuristic (optionally randomized)."""
    n = len(pt)
    m = pt.shape[1]
    totals = pt.sum(axis=1)
    means = pt.mean(axis=1)
    variances = pt.var(axis=1)
    spread = (m - 1) * (totals - variances)

    order = []
    remaining = list(range(n))
    while remaining:
        if rng is None:
            idx = max(remaining, key=lambda i: spread[i])
        else:
            vals = np.array([spread[i] for i in remaining])
            # soft sampling: higher spread -> higher probability
            w = (vals - vals.min() + 1.0)
            w = w ** 2
            p = w / w.sum()
            idx = remaining[int(rng.choice(len(remaining), p=p))]
        # insert idx at best position
        best_cost = None
        best_pos = 0
        for pos in range(len(order) + 1):
            cand = order[:pos] + [idx] + order[pos:]
            c = _makespan(pt, cand)
            if best_cost is None or c < best_cost:
                best_cost = c
                best_pos = pos
        order = order[:best_pos] + [idx] + order[best_pos:]
        remaining.remove(idx)
    return order


def _insertion_pass(pt, order):
    """Full best-improvement insertion moves. Returns (order, improved)."""
    n = len(order)
    cur = _makespan(pt, order)
    improved = False
    for i in range(n):
        j = order.pop(i)
        best_cost = cur
        best_pos = i
        for pos in range(n):
            cand = order[:pos] + [j] + order[pos:]
            c = _makespan(pt, cand)
            if c < best_cost:
                best_cost = c
                best_pos = pos
        order = order[:best_pos] + [j] + order[best_pos:]
        if best_cost < cur:
            cur = best_cost
            improved = True
    return order, improved


def _twopass(pt, order):
    """Best-improvement 2-opt (reversal) moves. Returns (order, improved)."""
    n = len(order)
    cur = _makespan(pt, order)
    improved = False
    for i in range(n - 1):
        for j in range(i + 1, n):
            cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
            c = _makespan(pt, cand)
            if c < cur:
                order = cand
                cur = c
                improved = True
    return order, improved


def order_jobs(processing_times):
    pt = np.asarray(processing_times)
    n = len(pt)
    if n <= 1:
        return list(range(n))

    t0 = time.time()
    deadline = t0 + _DEADLINE

    rng = np.random.default_rng()

    # Start from NEH
    best = _neh(pt, None)
    best_cost = _makespan(pt, best)

    # Diversifying restarts: a few randomized NEH seeds
    n_restarts = 3
    for _ in range(n_restarts):
        if time.time() > deadline:
            break
        cand = _neh(pt, rng)
        c = _makespan(pt, cand)
        if c < best_cost:
            best = cand
            best_cost = c

    # Local search: alternate insertion and 2-opt until no improvement or time up
    while time.time() < deadline:
        improved = False
        if time.time() < deadline:
            best, imp = _insertion_pass(pt, best)
            improved = improved or imp
            best_cost = _makespan(pt, best)
        if time.time() < deadline:
            best, imp = _twopass(pt, best)
            improved = improved or imp
            best_cost = _makespan(pt, best)
        if not improved:
            break

    return list(best)
# EVOLVE-BLOCK-END
