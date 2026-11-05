import numpy as np
from collections import deque
import time

_DEADLINE = 118.0
_START = None


def _now():
    global _START
    if _START is None:
        _START = time.perf_counter()
    return time.perf_counter() - _START


def _compute_slack(instance):
    """Compute slack (LS - ES) for each activity."""
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    preds = instance.get("predecessors", [])
    
    # Build successors
    succs = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        p = preds[i] if i < len(preds) else []
        for j in p:
            succs[j].append(i)
            in_degree[i] += 1
    
    # Topological sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    if len(topo_order) != n:
        # Fallback: no valid topo order
        return np.zeros(n), durations
    
    # Forward pass: ES, EF
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        if preds[u]:
            ES[u] = max(EF[p] for p in preds[u])
        else:
            ES[u] = 0.0
        EF[u] = ES[u] + durations[u]
    
    T = max(EF) if n > 0 else 0.0
    
    # Backward pass: LF, LS
    LF = np.full(n, T)
    LS = np.full(n, T)
    for u in reversed(topo_order):
        if succs[u]:
            LF[u] = min(LS[v] for v in succs[u])
        else:
            LF[u] = T
        LS[u] = LF[u] - durations[u]
    
    slack = LS - ES
    return slack, durations


def _compute_resource_opportunity_cost(instance):
    """
    Compute resource opportunity cost for each activity.
    
    For each activity, compute:
      - For each resource r: scarcity_r = demand[activity, r] / capacity[r]
      - max_scarcity = max over all resources of scarcity_r
      - opportunity_cost = duration * max_scarcity
    
    This penalizes activities that consume scarce resources for long periods.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is None or capacities is None:
        return np.zeros(n)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    
    # Handle edge cases
    if demands.ndim == 1:
        demands = demands.reshape(-1, 1)
    if capacities.ndim == 0:
        capacities = capacities.reshape(1)
    
    # Safe division
    safe_cap = np.where(capacities > 0, capacities, 1.0)
    
    # For each activity, compute scarcity per resource: demand[i, r] / capacity[r]
    # Shape: (n, num_resources)
    scarcity = demands / safe_cap  # broadcasting: (n, k) / (k,) -> (n, k)
    
    # Max scarcity across resources for each activity
    max_scarcity = np.max(scarcity, axis=1)
    
    # Opportunity cost: duration * max_scarcity
    opp_cost = durations * max_scarcity
    
    return opp_cost


def priority(instance):
    global _START
    if _START is None:
        _START = time.perf_counter()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Compute slack and critical path info
    slack, durations_arr = _compute_slack(instance)
    
    # If slack computation failed, fall back to duration
    if slack is None or (np.all(slack == 0) and np.all(durations_arr == 0)):
        return durations
    
    # Compute resource opportunity cost
    opp_cost = _compute_resource_opportunity_cost(instance)
    
    # Normalize slack: lower slack = higher priority
    slack_min = slack.min()
    slack_max = slack.max()
    
    if slack_max > slack_min:
        # Invert: low slack -> high value
        slack_norm = 1.0 - (slack - slack_min) / (slack_max - slack_min)
    else:
        slack_norm = np.ones(n)
    
    # Normalize opportunity cost: higher cost = higher priority
    oc_min = opp_cost.min()
    oc_max = opp_cost.max()
    if oc_max > oc_min:
        oc_norm = (opp_cost - oc_min) / (oc_max - oc_min)
    else:
        oc_norm = np.zeros(n)
    
    # Combine: slack is dominant (critical path), opportunity cost is secondary
    # Weight: 0.6 slack + 0.4 opportunity cost
    priority_scores = 0.6 * slack_norm + 0.4 * oc_norm
    
    # Check deadline
    if _now() > _DEADLINE:
        return priority_scores
    
    return priority_scores
