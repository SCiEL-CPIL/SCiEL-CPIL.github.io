"""Priority function for resource-constrained project scheduling (serial SGS)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s wall-clock kill


def priority(instance):
    """Return a priority score per activity (higher = schedule earlier).

    Combines:
      - Critical path length (longest path from activity to project end)
      - Resource peak contention (max demand/capacity ratio per activity)
      - Weighted successor critical path (bottleneck indicator)
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    # Build successor lists
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)

    # Compute critical path length (longest path to end) via reverse topological order
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        for i in predecessors[j]:
            in_deg[j] += 1

    # Topological order (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    # If topo is incomplete (cycle), fall back to index order
    if len(topo) < n:
        topo = list(range(n))

    # Reverse topological order for backward pass
    rtopo = list(reversed(topo))

    # Longest path from each node to end (including its own duration)
    cp_len = np.zeros(n, dtype=float)
    for u in rtopo:
        cp_len[u] = durations[u]
        for v in successors[u]:
            if cp_len[v] + durations[u] > cp_len[u]:
                cp_len[u] = durations[u] + cp_len[v]

    # Resource peak contention: max(demand[r] / capacity[r]) across resources
    # This captures the tightest single-resource constraint for each activity
    if demands.ndim == 1:
        # Single resource case
        peak_contention = demands / max(capacities, 1e-9)
    else:
        cap_safe = np.where(capacities > 0, capacities, 1e-9)
        # Per-activity, per-resource ratio, then take max across resources
        ratios = demands / cap_safe  # shape (n, R)
        peak_contention = np.max(ratios, axis=1)

    # Weighted successor critical path: sum of successor cp_lens
    # This captures how much "downstream work" depends on this activity
    w_succ_cp = np.zeros(n, dtype=float)
    for u in range(n):
        if successors[u]:
            w_succ_cp[u] = np.max(cp_len[successors[u]])
        else:
            w_succ_cp[u] = 0.0

    # Normalize each component to [0, 1]
    def norm(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    cp_norm = norm(cp_len)
    contention_norm = norm(peak_contention)
    w_succ_norm = norm(w_succ_cp)

    # Weighted combination: critical path most important, then contention, then successor work
    score = 0.55 * cp_norm + 0.25 * contention_norm + 0.20 * w_succ_norm

    # Time guard: if we're running long, return immediately
    if time.time() - _START_TIME > _TIME_LIMIT:
        return score

    return score
# EVOLVE-BLOCK-END
