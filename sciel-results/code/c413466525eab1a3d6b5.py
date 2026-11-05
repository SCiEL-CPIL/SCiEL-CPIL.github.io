"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    demands = instance["demands"]
    capacities = np.asarray(instance["capacities"], dtype=float)
    n_resources = len(capacities)

    # Build successor lists and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    # If not all nodes are in topo_order, there's a cycle; fall back to 0..n-1
    if len(topo_order) != n:
        topo_order = list(range(n))

    # CPM Forward pass
    ES = np.zeros(n)
    EF = np.zeros(n)
    for i in topo_order:
        if predecessors[i]:
            ES[i] = max(EF[p] for p in predecessors[i])
        EF[i] = ES[i] + durations[i]

    project_duration = max(EF)

    # CPM Backward pass (reverse topo order)
    LF = np.full(n, project_duration)
    LS = np.full(n, project_duration)
    for i in reversed(topo_order):
        if successors[i]:
            LF[i] = min(LS[s] for s in successors[i])
        LS[i] = LF[i] - durations[i]

    # Total float
    total_float = LS - ES

    # Number of immediate successors
    num_successors = np.array([len(s) for s in successors], dtype=float)

    # Resource pressure: sum of (demand / capacity) for each activity
    resource_pressure = np.zeros(n)
    for i in range(n):
        for r in range(n_resources):
            if capacities[r] > 0:
                resource_pressure[i] += demands[i][r] / capacities[r]

    # Combine: primary is -total_float (lower float = higher priority)
    # Secondary: more successors = higher priority (tiebreaker)
    # Tertiary: higher resource pressure = slightly higher priority
    float_range = total_float.max() - total_float.min() + 1e-9
    succ_range = num_successors.max() - num_successors.min() + 1e-9
    rp_range = resource_pressure.max() - resource_pressure.min() + 1e-9

    priority_score = (-total_float / float_range) * 1.0 + (num_successors / succ_range) * 0.1 + (resource_pressure / rp_range) * 0.05

    return priority_score
# EVOLVE-BLOCK-END
