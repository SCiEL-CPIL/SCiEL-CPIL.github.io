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


def priority(instance):
    global _START
    if _START is None:
        _START = time.perf_counter()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Compute slack and critical path info
    slack, durations_arr = _compute_slack(instance)
    
    # If slack computation failed, fall back to duration
    if slack is None or np.all(slack == 0) and np.all(durations_arr == 0):
        return durations
    
    # Resource demand factor
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_score = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        if demands.ndim == 2:
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            resource_score = np.sum(demands / safe_cap, axis=1)
        else:
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            resource_score = demands / safe_cap
    
    # Normalize slack: lower slack = higher priority
    # Slack ranges from 0 to some max
    slack_min = slack.min()
    slack_max = slack.max()
    
    if slack_max > slack_min:
        # Invert: low slack -> high value
        slack_norm = 1.0 - (slack - slack_min) / (slack_max - slack_min)
    else:
        slack_norm = np.ones(n)
    
    # Normalize duration: longer duration = higher priority
    dur_min = durations.min()
    dur_max = durations.max()
    if dur_max > dur_min:
        dur_norm = (durations - dur_min) / (dur_max - dur_min)
    else:
        dur_norm = np.ones(n)
    
    # Normalize resource score
    rs_min = resource_score.min()
    rs_max = resource_score.max()
    if rs_max > rs_min:
        rs_norm = (resource_score - rs_min) / (rs_max - rs_min)
    else:
        rs_norm = np.zeros(n)
    
    # Combine: slack is dominant (critical path), then duration, then resource
    # Weight: 0.7 slack + 0.2 duration + 0.1 resource
    priority_scores = 0.7 * slack_norm + 0.2 * dur_norm + 0.1 * rs_norm
    
    # Check deadline
    if _now() > _DEADLINE:
        return priority_scores
    
    return priority_scores
