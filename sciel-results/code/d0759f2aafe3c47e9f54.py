"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import deque


def _compute_ranks(times, preds):
    """Compute bottom-level (RPV), top-level, successor counts, and weighted successor RPV."""
    n = len(times)
    succs = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            indeg[i] += 1

    # Topological sort (Kahn's)
    q = deque([i for i in range(n) if indeg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(topo) != n:
        topo = list(range(n))

    # Bottom-level (RPV): longest path from node to a sink, including node's time
    rpv = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in succs[u]:
            if rpv[v] > best:
                best = rpv[v]
        rpv[u] = times[u] + best

    # Top-level: longest path from a source to node, including node's time
    top = np.zeros(n, dtype=float)
    for u in topo:
        best = 0.0
        for p in preds[u]:
            if top[p] > best:
                best = top[p]
        top[u] = times[u] + best

    succ_count = np.array([len(s) for s in succs], dtype=float)

    # Weighted successor RPV: sum of RPV of immediate successors (captures "width" impact)
    # A task with many high-RPV successors is more valuable to schedule early.
    w_succ_rpv = np.zeros(n, dtype=float)
    for u in range(n):
        if succs[u]:
            w_succ_rpv[u] = float(np.sum(rpv[succs[u]]))

    return rpv, top, succ_count, w_succ_rpv


def priority(instance):
    t0 = _time.time()
    DEADLINE = 140.0  # safety margin before the 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    rpv, top, succ_count, w_succ_rpv = _compute_ranks(times, preds)

    max_rpv = float(np.max(rpv)) if n else 1.0
    max_time = float(np.max(times)) if n else 1.0
    max_wsr = float(np.max(w_succ_rpv)) if n and np.max(w_succ_rpv) > 0 else 1.0

    # Improved weighting:
    # 1. RPV is primary (critical path urgency)
    # 2. Weighted successor RPV captures "unlocking" value - tasks that enable many
    #    critical-path tasks should be prioritized
    # 3. Task time as LPT tie-breaker
    # 4. Successor count as secondary unlock signal
    #
    # Use log-sum-exp-inspired soft combination to avoid hard threshold issues.
    # Normalize each component to [0, 1] then combine with learned-like weights.

    rpv_norm = rpv / max_rpv if max_rpv > 0 else rpv
    wsr_norm = w_succ_rpv / max_wsr if max_wsr > 0 else w_succ_rpv
    time_norm = times / max_time if max_time > 0 else times
    succ_norm = succ_count / (np.max(succ_count) if np.max(succ_count) > 0 else 1.0)

    # Weights: RPV dominant, weighted successor RPV significant (captures width),
    # time moderate, successor count small
    w1 = 1.0      # RPV
    w2 = 0.35     # weighted successor RPV (new: captures unlock value)
    w3 = 0.15     # task time (LPT)
    w4 = 0.05     # successor count

    scores = w1 * rpv_norm + w2 * wsr_norm + w3 * time_norm + w4 * succ_norm

    # Scale back to original magnitude for stable comparison
    scores = scores * max_rpv

    if _time.time() - t0 > DEADLINE:
        return times

    return scores
# EVOLVE-BLOCK-END
