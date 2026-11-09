"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]  # list of lists: preds[i] = list of predecessor indices
    n = len(times)
    cycle_time = instance["cycle_time"]

    # Build successor list
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)

    # Compute RPW: number of all successors (transitive)
    # Use topological order (reverse topological for successors)
    in_degree = [len(preds[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_indeg = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            temp_indeg[v] -= 1
            if temp_indeg[v] == 0:
                queue.append(v)

    # RPW: count all transitive successors
    rpw = np.zeros(n)
    # Process in reverse topological order
    for u in reversed(topo_order):
        rpw[u] = sum(1 + rpw[v] for v in succs[u])

    # Critical path length from task to end (longest path through successors)
    # cp[u] = times[u] + max(cp[v] for v in succs[u], default=0)
    cp = np.zeros(n)
    for u in reversed(topo_order):
        max_succ = 0.0
        for v in succs[u]:
            if cp[v] > max_succ:
                max_succ = cp[v]
        cp[u] = times[u] + max_succ

    # Immediate successor count
    imm_succ = np.array([len(succs[i]) for i in range(n)], dtype=float)

    # Normalize each component to [0, 1]
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    norm_times = normalize(times)
    norm_rpw = normalize(rpw)
    norm_cp = normalize(cp)
    norm_imm = normalize(imm_succ)

    # Composite priority: weight the components
    # Critical path is most important for leaving room
    # RPW captures "how much work depends on this"
    # LPT captures difficulty of fitting
    # Immediate successors capture urgency
    score = (
        0.35 * norm_cp +
        0.25 * norm_rpw +
        0.25 * norm_times +
        0.15 * norm_imm
    )

    return score
# EVOLVE-BLOCK-END
