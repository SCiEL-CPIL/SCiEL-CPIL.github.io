"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graph(instance):
    """Build adjacency and in-degree structures from predecessors."""
    n = len(instance["durations"])
    pred = instance["predecessors"]
    # pred[i] is a list/set of indices that must precede i
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
    # If not all nodes are in order, there's a cycle; append remaining
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
    """Compute a resource pressure score: for each activity, the sum of demands
    weighted by how tight the resources are."""
    demands = instance["demands"]  # shape (n, R) or list of lists
    capacities = instance["capacities"]  # shape (R,)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    R = len(capacities)
    
    # For each activity, compute how much it "pressures" each resource
    # pressure_i = sum_r (demand[i,r] / capacity[r])^2 * demand[i,r]
    # This emphasizes activities that use scarce resources heavily.
    if R > 0:
        ratio = demands / np.maximum(capacities, 1e-9)  # (n, R)
        pressure = np.sum(ratio * demands, axis=1)  # (n,)
    else:
        pressure = np.zeros(n)
    return pressure


def _criticality_score(instance):
    """Compute a combined priority score: low slack + high resource pressure."""
    succ, indeg, n = _build_graph(instance)
    order = _topo_order(succ, indeg.copy(), n)
    slack, est, lst, makespan = _compute_slack(instance, succ, order, n)
    pressure = _resource_pressure(instance, succ, order, n)
    
    # Normalize slack to [0, 1]
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0
    norm_slack = slack / max_slack
    
    # Normalize pressure to [0, 1]
    max_pressure = np.max(pressure) if np.max(pressure) > 0 else 1.0
    norm_pressure = pressure / max_pressure
    
    # Priority: we want high priority for low slack (critical) and high pressure
    # score = (1 - norm_slack) * 0.7 + norm_pressure * 0.3
    # Higher score = higher priority
    score = (1.0 - norm_slack) * 0.7 + norm_pressure * 0.3
    return score


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Returns a numpy array of scores; higher score = higher priority.
    """
    start_time = time.time()
    
    try:
        score = _criticality_score(instance)
        return np.asarray(score, dtype=float)
    except Exception:
        # Fallback: longest duration first
        return np.asarray(instance["durations"], dtype=float)
# EVOLVE-BLOCK-END
