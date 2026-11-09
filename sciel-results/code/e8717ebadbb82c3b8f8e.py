import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher priority (score) is scheduled first among eligible activities.

    Design (refined within the CPM + resource-aware family):
    1. Compute Critical Path Length (CPL): longest path from each activity to project end.
       This is the primary, dominant priority signal.
    2. Compute a resource-contention signal: max over resources of demand/capacity.
       This favors activities that use scarce resources, reducing later blocking.
    3. Small tie-break by duration for determinism.

    The serial SGS benefits most from scheduling critical-path activities first;
    the resource term is a small secondary signal to avoid wasting scarce capacity.
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

    # Compute resource-contention signal using a smooth log-sum-exp surrogate for max.
    # This captures the "resource tail": how much an activity constrains the overall
    # resource profile, rather than just its single worst ratio. Activities with
    # multiple moderately high ratios get appropriately high scores (unlike hard max
    # which ignores all but the single worst).
    # tau controls smoothness: small tau -> approaches hard max; larger tau -> smoother.
    tau = 0.1  # small enough to approximate max but smooth enough to capture tail
    res_ratio = np.zeros(n, dtype=float)
    for i in range(n):
        ratios = []
        for j in range(len(capacities)):
            if capacities[j] > 0 and demands[i][j] > 0:
                ratios.append(demands[i][j] / capacities[j])
        if ratios:
            arr = np.asarray(ratios, dtype=float)
            # Log-sum-exp: tau * log(sum(exp(x/tau))) approximates max(x) as tau->0
            m = arr.max()
            res_ratio[i] = tau * np.log(np.sum(np.exp((arr - m) / tau))) + m
        else:
            res_ratio[i] = 0.0

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute Latest Start Time (LST) to derive slack
    # LST[u] = min over successors v of (LST[v] - durations[u])
    # For sinks, LST = project_duration - durations[u]
    project_duration = max(cpl)
    lst = np.zeros(n, dtype=float)
    for u in topo_order:
        if not successors[u]:
            lst[u] = project_duration - durations[u]
        else:
            lst[u] = min(lst[v] - durations[u] for v in successors[u])

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute Earliest Start Time (EST) to derive slack
    est = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            est[u] = 0
        else:
            est[u] = max(est[p] + durations[p] for p in preds[u])

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Total Slack = LST - EST (higher slack = more flexibility)
    slack = lst - est

    # Normalize components
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    cpl_norm = normalize(cpl)
    res_norm = normalize(res_ratio)
    dur_norm = normalize(durations)
    slack_norm = normalize(slack)

    # Combine scores:
    # Primary: Critical Path Length (dominant signal)
    # Secondary: Resource contention (small boost for scarce-resource users)
    # Tertiary: Low slack (activities with less flexibility should be prioritized)
    # Quaternary: Duration (small tie-break for determinism)
    # Negative slack_norm because lower slack should increase priority
    scores = 1.0 * cpl_norm + 0.12 * res_norm - 0.15 * slack_norm + 0.05 * dur_norm

    return scores
