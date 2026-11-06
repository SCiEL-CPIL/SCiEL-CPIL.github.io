"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graph(instance):
    """Build adjacency and in-degree structures from predecessors."""
    n = len(instance["durations"])
    pred = instance["predecessors"]
    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        p_list = pred[i] if pred[i] is not None else []
        for p in p_list:
            succ[p].append(i)
            indeg[i] += 1
    return succ, indeg, n


def _topo_order(succ, indeg, n):
    """Kahn's algorithm for topological ordering."""
    from collections import deque
    q = deque()
    for i in range(n):
        if indeg[i] == 0:
            q.append(i)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < n:
        remaining = set(range(n)) - set(order)
        order.extend(sorted(remaining))
    return order


def _forward_pass(instance, succ, order):
    """Compute earliest start times (EST) ignoring resources."""
    n = len(instance["durations"])
    dur = np.asarray(instance["durations"], dtype=float)
    est = np.zeros(n, dtype=float)
    for u in order:
        for v in succ[u]:
            if est[v] < est[u] + dur[u]:
                est[v] = est[u] + dur[u]
    return est


def _backward_pass(instance, succ, order, n, horizon):
    """Compute latest start times (LST) given a horizon."""
    dur = np.asarray(instance["durations"], dtype=float)
    lst = np.full(n, horizon, dtype=float)
    for u in reversed(order):
        for v in succ[u]:
            if lst[u] > lst[v] - dur[u]:
                lst[u] = lst[v] - dur[u]
    return lst


def _compute_slack(instance, succ, order, n):
    """Compute total slack for each activity."""
    est = _forward_pass(instance, succ, order)
    makespan = np.max(est + np.asarray(instance["durations"], dtype=float))
    lst = _backward_pass(instance, succ, order, n, makespan)
    slack = lst - est
    return slack, est, lst, makespan


def _resource_pressure(instance, succ, order, n):
    """Compute a resource pressure score."""
    demands = instance["demands"]
    capacities = instance["capacities"]
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    R = len(capacities)
    if R > 0:
        ratio = demands / np.maximum(capacities, 1e-9)
        pressure = np.sum(ratio * demands, axis=1)
    else:
        pressure = np.zeros(n)
    return pressure


def _soft_min_critical_path(instance, succ, order, n, temp=2.0):
    """Compute a smooth estimate of the longest path from each node to the end.
    Uses log-sum-exp soft-min over successor durations as a surrogate for
    the exact critical path length, which helps differentiate activities
    on near-critical paths."""
    dur = np.asarray(instance["durations"], dtype=float)
    # path_len[i] = smooth estimate of longest path starting at i
    path_len = np.zeros(n, dtype=float)
    for u in reversed(order):
        if not succ[u]:
            path_len[u] = dur[u]
        else:
            # soft-min: -log(sum(exp(-x/temp))) * temp approximates min
            succ_vals = dur[u] + path_len[succ[u]]
            # Use log-sum-exp for numerical stability
            m = np.min(succ_vals)
            if m < -1e6:
                path_len[u] = dur[u]
            else:
                log_sum = np.log(np.sum(np.exp(-(succ_vals - m) / temp)))
                path_len[u] = dur[u] - temp * log_sum + m - dur[u] + dur[u]
                # Simplify: path_len[u] = dur[u] + soft_min(path_len[succ])
                # soft_min(x) = -temp * log(mean(exp(-x/temp)))
                # Let's redo this more carefully
                soft_min_val = -temp * np.log(np.mean(np.exp(-(succ_vals - dur[u]) / temp))) + dur[u]
                path_len[u] = dur[u] + soft_min_val
    return path_len


def _criticality_score(instance):
    """Compute a combined priority score: low slack + high resource pressure + duration."""
    succ, indeg, n = _build_graph(instance)
    order = _topo_order(succ, indeg.copy(), n)
    slack, est, lst, makespan = _compute_slack(instance, succ, order, n)
    pressure = _resource_pressure(instance, succ, order, n)
    dur = np.asarray(instance["durations"], dtype=float)
    
    # Normalize slack relative to makespan for scale invariance
    if makespan > 0:
        norm_slack = slack / makespan
    else:
        norm_slack = np.zeros(n)
    
    # Normalize pressure
    max_pressure = np.max(pressure) if np.max(pressure) > 0 else 1.0
    norm_pressure = pressure / max_pressure
    
    # Normalize duration
    max_dur = np.max(dur) if np.max(dur) > 0 else 1.0
    norm_dur = dur / max_dur
    
    # Priority: emphasize low slack (critical path) most, then resource pressure, then duration
    # Weighting: 0.6 criticality + 0.25 pressure + 0.15 duration
    score = (1.0 - norm_slack) * 0.6 + norm_pressure * 0.25 + norm_dur * 0.15
    return score


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Returns a numpy array of scores; higher score = higher priority.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    try:
        score = _criticality_score(instance)
        score = np.asarray(score, dtype=float)
        # Verify no NaN/Inf
        if not np.all(np.isfinite(score)):
            score = np.where(np.isfinite(score), score, 0.0)
        return score
    except Exception:
        # Fallback: longest duration first
        return np.asarray(instance["durations"], dtype=float)
# EVOLVE-BLOCK-END
