"""Priority function for resource-constrained project scheduling (serial SGS)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s wall-clock kill


def _norm(x):
    """Min-max normalize to [0,1]; return zeros if degenerate."""
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def priority(instance):
    """Return a priority score per activity (higher = schedule earlier).

    Combines (same design family as the seed):
      - Critical path length (longest path from activity to project end)
      - Schedule-aware resource contention (bottleneck-resource pressure)
      - Successor critical path (downstream work / bottleneck indicator)
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    # Build successor lists
    successors = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
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

    # Reverse topological order for the backward pass
    rtopo = list(reversed(topo))

    # Longest path from each node to end (including its own duration)
    cp_len = np.zeros(n, dtype=float)
    for u in rtopo:
        best = 0.0
        for v in successors[u]:
            if cp_len[v] > best:
                best = cp_len[v]
        cp_len[u] = durations[u] + best

    # ---- Schedule-aware resource contention proxy -----------------------
    # util[r] = total time resource r is busy = sum_a demand[a][r] / duration[a]
    # A resource with high util (relative to a nominal project horizon) is the
    # real bottleneck; activities drawing on it should be prioritized.
    if demands.ndim == 1:
        R = 1
        cap_safe = np.where(capacities > 0, capacities, 1e-9)
        # per-activity pressure on the single scarce resource
        util = np.sum(demands / np.maximum(durations, 1e-9))
        pressure = demands / (cap_safe * max(util, 1e-9))
    else:
        cap_safe = np.where(capacities > 0, capacities, 1e-9)  # (R,)
        dur_safe = np.maximum(durations, 1e-9)                 # (n,)
        # util[r] = sum_a demand[a][r] / duration[a]
        util = np.sum(demands / dur_safe[:, None], axis=0)      # (R,)
        # pressure[a] = sum_r demand[a][r] / (capacity[r] * util[r])
        denom = cap_safe * np.maximum(util, 1e-9)               # (R,)
        pressure = np.sum(demands / denom[None, :], axis=1)     # (n,)

    # Successor critical path: max cp_len among successors (downstream criticality)
    w_succ_cp = np.zeros(n, dtype=float)
    for u in range(n):
        if successors[u]:
            m = 0.0
            for v in successors[u]:
                if cp_len[v] > m:
                    m = cp_len[v]
            w_succ_cp[u] = m

    # Normalize each component to [0, 1]
    cp_norm = _norm(cp_len)
    pressure_norm = _norm(pressure)
    w_succ_norm = _norm(w_succ_cp)

    # Weighted combination: critical path dominant, then schedule-aware
    # resource pressure, then downstream criticality.
    score = 0.60 * cp_norm + 0.25 * pressure_norm + 0.15 * w_succ_norm

    # Time guard: if we're running long, return immediately
    if time.time() - _START_TIME > _TIME_LIMIT:
        return score

    return score
# EVOLVE-BLOCK-END
