import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher priority (score) is scheduled first among eligible activities.

    Design (CPM + resource-aware family, refined):
    1. Compute Critical Path Length (CPL): longest path from each activity to project end.
       Primary, dominant priority signal.
    2. Compute Earliest Start (ES) and Latest Start (LS) to get Total Float (Slack).
       Low slack = high criticality, complementary to CPL.
    3. Compute a smooth resource-contention signal using log-sum-exp soft-max over
       demand/capacity ratios. This accounts for multiple near-saturated resources
       rather than just the single worst one, giving a smoother, more informative
       pressure signal.
    4. Combine: CPL (dominant) + criticality (from slack) + smooth resource pressure.

    The serial SGS benefits most from scheduling critical-path activities first;
    the smooth resource term helps avoid wasting scarce capacity on non-critical tasks
    while capturing the cumulative pressure from activities that stress multiple resources.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]

    # Build successor list and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological sort (Kahn's algorithm)
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(topo_order) != n:
        # Fallback: just use durations as priority
        return durations.copy()

    # Compute Critical Path Length (CPL) - longest path from activity to project end
    cpl = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            cpl[u] = durations[u]
        else:
            cpl[u] = durations[u] + max(cpl[v] for v in successors[u])

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute Earliest Start times
    es = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            es[u] = 0.0
        else:
            es[u] = max(es[p] + durations[p] for p in preds[u])

    # Project makespan estimate (CPM length)
    project_length = max(es[i] + durations[i] for i in range(n))

    # Compute Latest Finish and Latest Start times
    lf = np.full(n, project_length, dtype=float)
    ls = np.full(n, project_length, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            lf[u] = project_length
        else:
            lf[u] = min(ls[v] for v in successors[u])
        ls[u] = lf[u] - durations[u]

    # Total Float (Slack) = LS - ES
    # Lower slack means more critical
    slack = ls - es

    # Check deadline
    if time.time() - start_time > DEADLINE:
        cpl_norm = (cpl - cpl.min()) / (cpl.max() - cpl.min() + 1e-10)
        return cpl_norm

    # Compute smooth resource-contention signal using log-sum-exp soft-max.
    # For each activity, compute demand/capacity ratios for all resources,
    # then use a temperature-scaled log-sum-exp to get a smooth max-like value
    # that accounts for multiple near-saturated resources.
    n_resources = len(capacities)
    # Build ratio matrix: shape (n, n_resources)
    ratios = np.zeros((n, n_resources), dtype=float)
    for i in range(n):
        for j in range(n_resources):
            if capacities[j] > 0 and demands[i][j] > 0:
                ratios[i][j] = demands[i][j] / capacities[j]

    # Temperature-scaled log-sum-exp for smooth max
    # As temperature -> 0, this approaches the hard max
    # As temperature -> inf, this approaches the mean
    # We use a moderate temperature to capture multiple near-saturated resources
    temperature = 0.1  # Small temperature = close to hard max but smooth
    # Scale ratios by 1/temperature for log-sum-exp
    scaled = ratios / temperature
    # Subtract max for numerical stability
    scaled_max = np.max(scaled, axis=1, keepdims=True)
    # Handle case where all ratios are zero
    scaled_safe = np.where(scaled_max > 0, scaled - scaled_max, 0.0)
    exp_scaled = np.exp(scaled_safe)
    log_sum_exp = np.log(np.sum(exp_scaled, axis=1)) + scaled_max.squeeze()
    res_pressure = log_sum_exp * temperature  # Scale back

    # Check deadline
    if time.time() - start_time > DEADLINE:
        cpl_norm = (cpl - cpl.min()) / (cpl.max() - cpl.min() + 1e-10)
        return cpl_norm

    # Normalize components
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    cpl_norm = normalize(cpl)
    slack_norm = normalize(slack)  # Low slack -> low norm
    res_norm = normalize(res_pressure)

    # Criticality: inverse of slack (low slack = high criticality)
    crit = 1.0 - slack_norm

    # Combine scores:
    # Primary: Critical Path Length (dominant signal)
    # Secondary: Criticality from slack (complementary to CPL)
    # Tertiary: Smooth resource pressure (captures multi-resource contention)
    scores = 1.0 * cpl_norm + 0.25 * crit + 0.15 * res_norm

    return scores
