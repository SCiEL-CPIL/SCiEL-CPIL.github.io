import numpy as np
import time

_START_TIME = None
_DEADLINE = 115.0  # safety margin before 120s kill


def priority(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        durations = np.asarray(instance["durations"], dtype=float)
        return durations.copy()

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    n_res = len(capacities)

    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)

    # Compute topological order (Kahn's algorithm)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)

    # Calculate CPM Rank (successor depth - longest path from i to end)
    cpm_rank = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            cpm_rank[u] = durations[u]
        else:
            max_succ = max(cpm_rank[v] for v in successors[u])
            cpm_rank[u] = durations[u] + max_succ

    # Calculate Resource Pressure
    safe_caps = np.where(capacities == 0, 1, capacities)
    ratios = demands / safe_caps
    resource_pressure = np.max(ratios, axis=1)

    # Calculate Successor Count
    succ_count = np.array([len(s) for s in successors], dtype=float)

    # Calculate Predecessor Count (activities with many predecessors are more constrained)
    pred_count = np.array([len(p) for p in predecessors], dtype=float)

    # Calculate average demand per resource (higher means more resource-intensive)
    avg_demand = np.mean(demands, axis=1)

    # Normalize scores to be comparable
    # CPM rank is in time units, typically the dominant factor
    cpm_min, cpm_max = np.min(cpm_rank), np.max(cpm_rank)
    cpm_range = cpm_max - cpm_min if cpm_max > cpm_min else 1.0
    cpm_norm = (cpm_rank - cpm_min) / cpm_range

    # Resource pressure is already 0-1 roughly
    # Succ count and pred count are small integers

    # Weighted combination:
    # CPM rank is the most important (drives critical path)
    # Resource pressure: higher pressure activities should be scheduled earlier to free up resources
    # Succ count: activities with more successors have more impact on the rest of the schedule
    # Pred count: activities with more predecessors are more constrained, schedule them when ready
    # Avg demand: higher demand activities consume more resources, schedule earlier if they fit

    w_cpm = 1.0
    w_res = 3.0
    w_succ = 0.8
    w_pred = 0.3
    w_demand = 0.5

    # Normalize succ_count, pred_count, avg_demand to 0-1 range
    succ_max = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    succ_norm = succ_count / succ_max

    pred_max = np.max(pred_count) if np.max(pred_count) > 0 else 1.0
    pred_norm = pred_count / pred_max

    demand_max = np.max(avg_demand) if np.max(avg_demand) > 0 else 1.0
    demand_norm = avg_demand / demand_max

    scores = (w_cpm * cpm_norm * cpm_range +
              w_res * resource_pressure * cpm_range * 0.1 +
              w_succ * succ_norm * cpm_range * 0.05 +
              w_pred * pred_norm * cpm_range * 0.02 +
              w_demand * demand_norm * cpm_range * 0.03)

    return scores
