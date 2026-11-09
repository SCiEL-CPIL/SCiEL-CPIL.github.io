"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graph(instance):
    """Build predecessor/successor lists from instance."""
    n = len(instance["durations"])
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = instance.get("preds", None)
    if preds is None:
        # Try other keys
        for key in ["predecessor", "pred", "prec"]:
            if key in instance:
                preds = instance[key]
                break
    if preds is None:
        preds = [[] for _ in range(n)]
    else:
        # Normalize to list of lists
        if isinstance(preds, (list, tuple)):
            if len(preds) > 0 and isinstance(preds[0], (list, tuple, np.ndarray)):
                preds = [list(p) for p in preds]
            else:
                # Maybe it's a flat list or dict
                preds = [list(p) for p in preds]
        elif isinstance(preds, dict):
            preds = [list(preds.get(i, [])) for i in range(n)]
    
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            if 0 <= p < n:
                succs[p].append(i)
    return preds, succs


def _critical_path_lengths(durations, preds, succs):
    """Compute longest path from each node to any sink (successor-free node)."""
    n = len(durations)
    # topological order via DFS
    in_degree = [len(preds[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    # If not all nodes in topo (cycle?), handle gracefully
    if len(topo) < n:
        # Fallback: just use reverse index order
        topo = list(range(n))
    
    # dist[i] = longest path from i to any sink
    dist = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for i in reversed(topo):
        if not succs[i]:
            dist[i] = durations[i]
        else:
            dist[i] = durations[i] + max(dist[j] for j in succs[i])
    return dist


def _resource_pressure(instance, durations, preds, succs, dist):
    """
    Estimate resource pressure for each activity.
    For each resource, compute the maximum concurrent demand if all activities
    were scheduled as early as possible (ignoring resource constraints).
    Then score each activity by how much it contributes to peak resource usage.
    """
    n = len(durations)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    if demands is None:
        demands = instance.get("resource_demands", None)
    if capacities is None:
        capacities = instance.get("resource_capacities", None)
    if demands is None:
        return np.ones(n, dtype=float)
    
    demands = np.asarray(demands, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(-1, 1)
    n_res = demands.shape[1] if demands.ndim > 1 else 1
    if capacities is None:
        capacities = np.ones(n_res, dtype=float)
    else:
        capacities = np.asarray(capacities, dtype=float)
        if capacities.ndim == 0:
            capacities = np.ones(n_res) * float(capacities)
    
    # Compute earliest start times (ignoring resource constraints)
    # ES[i] = max over predecessors p of (ES[p] + dur[p])
    es = np.zeros(n, dtype=float)
    # Topological order
    in_degree = [len(preds[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(topo) < n:
        topo = list(range(n))
    
    for u in topo:
        if preds[u]:
            es[u] = max(es[p] + durations[p] for p in preds[u])
        else:
            es[u] = 0.0
    
    # For each resource, compute peak load and which activities contribute
    # We'll estimate each activity's contribution to the peak as:
    #   demand[i,r] * (1 if i is active at the peak time for resource r)
    # To find peak times, we do a sweep
    pressure = np.zeros(n, dtype=float)
    max_capacity = max(float(np.max(capacities)), 1e-9)
    
    for r in range(n_res):
        cap_r = float(capacities[r]) if r < len(capacities) else 1.0
        if cap_r <= 0:
            cap_r = 1e-9
        # Collect intervals
        intervals = []
        for i in range(n):
            d = float(demands[i, r]) if demands.ndim > 1 else float(demands[i])
            if d > 0:
                intervals.append((es[i], es[i] + durations[i], d, i))
        if not intervals:
            continue
        # Sweep to find peak
        events = []
        for start, end, d, idx in intervals:
            events.append((start, 1, d))
            events.append((end, -1, d))
        events.sort(key=lambda x: (x[0], x[1]))
        current_load = 0.0
        peak_load = 0.0
        peak_active = set()
        active_set = set()
        for t, delta, d in events:
            if delta == 1:
                active_set.add(d)  # This is approximate; better to track by index
            # Actually let's track properly
        # Simpler: find max load
        active = set()
        peak = 0.0
        for start, end, d, idx in intervals:
            events2 = [(start, idx, 1), (end, idx, -1)]
        # Let's just do a simple sweep
        all_events = []
        for start, end, d, idx in intervals:
            all_events.append((start, 1, idx, d))
            all_events.append((end, -1, idx, d))
        all_events.sort(key=lambda x: (x[0], x[1]))
        load = 0.0
        active_idx = set()
        for t, delta, idx, d in all_events:
            if delta == 1:
                active_idx.add(idx)
                load += d
            else:
                active_idx.discard(idx)
                load -= d
            if load > peak:
                peak = load
                peak_active = set(active_idx)
        
        # Distribute pressure: activities active at peak get credit
        if peak > 0:
            for idx in peak_active:
                d = float(demands[idx, r]) if demands.ndim > 1 else float(demands[idx])
                pressure[idx] += d / (peak + 1e-9)
    
    # Normalize pressure to [0, 1]
    max_p = np.max(pressure)
    if max_p > 0:
        pressure /= max_p
    return pressure


def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Combines:
    1. Critical path length (successor-free distance)
    2. Duration
    3. Resource pressure (contribution to peak resource usage)
    4. Number of successors (breadth of impact)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([], dtype=float)
    
    preds, succs = _build_graph(instance)
    dist = _critical_path_lengths(durations, preds, succs)
    
    # Check deadline
    if time.time() > deadline:
        return dist.copy()
    
    # Resource pressure
    pressure = _resource_pressure(instance, durations, preds, succs, dist)
    
    # Check deadline
    if time.time() > deadline:
        return (dist + pressure).copy()
    
    # Successor count (normalized)
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    max_sc = np.max(succ_count)
    if max_sc > 0:
        succ_count /= max_sc
    
    # Predecessor count (normalized) - activities with many predecessors are "later"
    pred_count = np.array([len(preds[i]) for i in range(n)], dtype=float)
    max_pc = np.max(pred_count)
    if max_pc > 0:
        pred_count /= max_pc
    
    # Normalize each component to [0, 1]
    max_dist = np.max(dist)
    if max_dist > 0:
        dist_norm = dist / max_dist
    else:
        dist_norm = np.zeros(n)
    
    max_dur = np.max(durations)
    if max_dur > 0:
        dur_norm = durations / max_dur
    else:
        dur_norm = np.zeros(n)
    
    # Combine with weights
    # Critical path is most important, then resource pressure, then duration
    score = 0.45 * dist_norm + 0.25 * pressure + 0.20 * dur_norm + 0.10 * succ_count
    
    # Final check
    if time.time() > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
