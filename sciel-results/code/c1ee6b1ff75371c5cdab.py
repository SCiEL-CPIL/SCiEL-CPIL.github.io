"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

_START_TIME = time.time()
_DEADLINE = 140.0


def priority(instance):
    if time.time() - _START_TIME > _DEADLINE:
        return np.asarray(instance["durations"], dtype=float)

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    m = len(capacities)

    # Build adjacency
    successors = [[] for _ in range(n)]
    indegree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            indegree[i] += 1

    # Topological order
    topo = []
    queue = deque([i for i in range(n) if indegree[i] == 0])
    temp_indeg = indegree.copy()
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            temp_indeg[v] -= 1
            if temp_indeg[v] == 0:
                queue.append(v)

    if len(topo) != n:
        # Fallback: cycle detected, return durations
        return durations

    # Forward pass: earliest start (ES)
    es = np.zeros(n, dtype=float)
    for u in topo:
        if predecessors[u]:
            es[u] = max(es[p] + durations[p] for p in predecessors[u])

    # Backward pass: latest start (LS)
    project_end = max(es[i] + durations[i] for i in range(n))
    ls = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if successors[u]:
            ls[u] = min(ls[v] - durations[u] for v in successors[u])
        else:
            ls[u] = project_end - durations[u]

    # Total float / slack
    slack = ls - es

    # Smooth Critical Path Length (backward longest path to any sink)
    # Use a softmax-like aggregation to avoid cliff effects
    cp_length = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if successors[u]:
            # Use max as proxy for critical path; smooth by taking weighted max
            # Simple approach: max
            cp_length[u] = max(cp_length[v] + durations[v] for v in successors[u])
        else:
            cp_length[u] = 0.0

    # Normalize each component to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    cp_norm = normalize(cp_length)
    slack_norm = normalize(slack)
    crit_norm = 1.0 - slack_norm  # low slack -> high criticality

    # Resource intensity: sum of demand/capacity ratios
    res_int = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(m):
            if capacities[j] > 0:
                res_int[i] += demands[i, j] / capacities[j]
    res_norm = normalize(res_int)

    # Duration norm
    dur_norm = normalize(durations)

    # Combined score
    # Primary: smooth CPL
    # Secondary: criticality (low slack)
    # Tertiary: resource intensity
    # Quaternary: duration (break ties)
    weights_cp = 0.45
    weights_crit = 0.25
    weights_res = 0.20
    weights_dur = 0.10

    scores = (weights_cp * cp_norm +
              weights_crit * crit_norm +
              weights_res * res_norm +
              weights_dur * dur_norm)

    return scores
# EVOLVE-BLOCK-END
