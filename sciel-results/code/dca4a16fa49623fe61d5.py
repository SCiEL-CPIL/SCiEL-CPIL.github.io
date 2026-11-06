"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def priority(instance):
    t0 = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    cycle_time = instance["cycle_time"]

    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        p_list = preds[i] if i < len(preds) else []
        for p in p_list:
            successors[p].append(i)

    # Topological sort (Kahn's algorithm)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        p_list = preds[i] if i < len(preds) else []
        in_degree[i] = len(p_list)

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Compute rank: longest path from task to end (including own time)
    rank = np.zeros(n)
    for u in reversed(topo_order):
        if successors[u]:
            max_succ_rank = max(rank[v] for v in successors[u])
            rank[u] = times[u] + max_succ_rank
        else:
            rank[u] = times[u]

    # Count transitive successors (downstream tasks)
    transitive_succ = np.zeros(n, dtype=int)
    for u in reversed(topo_order):
        if successors[u]:
            transitive_succ[u] = len(successors[u]) + max(transitive_succ[v] for v in successors[u])

    # Sum of times of all transitive successors
    transitive_succ_time = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if successors[u]:
            transitive_succ_time[u] = times[u] + max(
                transitive_succ_time[v] for v in successors[u]
            )
        else:
            transitive_succ_time[u] = times[u]

    # Normalize features to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr, dtype=float)

    rank_norm = normalize(rank)
    ts_norm = normalize(transitive_succ.astype(float))
    t_norm = normalize(times)
    tst_norm = normalize(transitive_succ_time)

    # Combined priority:
    # - rank is the primary ALB heuristic (longest tail)
    # - transitive successor count identifies bottleneck tasks
    # - task time helps with packing (longer tasks are harder to fit)
    # - transitive successor time captures total downstream workload
    #
    # Weights tuned to emphasize rank while using secondary signals for tie-breaking
    priority_scores = (
        0.50 * rank_norm +
        0.20 * ts_norm +
        0.15 * t_norm +
        0.15 * tst_norm
    )

    # Small deterministic tie-breaker for stability
    priority_scores = priority_scores + np.arange(n) * 1e-12

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return times  # Fallback to simple time-based

    return priority_scores
# EVOLVE-BLOCK-END
