"""Direct approach for Assembly Line Balancing using explicit structure-derived priorities."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import heapq

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0  # safety margin before 150s kill


def _build_graph(instance):
    """Build adjacency and in-degree from predecessors."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    adj = defaultdict(list)
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                adj[i].append(j)
                in_deg[j] += 1
    return adj, in_deg, n


def _topo_order(n, in_deg, adj):
    """Kahn's algorithm for topological order."""
    in_deg = in_deg.copy()
    queue = [i for i in range(n) if in_deg[i] == 0]
    heapq.heapify(queue)
    topo = []
    while queue:
        u = heapq.heappop(queue)
        topo.append(u)
        for v in adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                heapq.heappush(queue, v)
    return topo


def _forward_longest(instance):
    """Longest path from start to each node."""
    adj, in_deg, n = _build_graph(instance)
    times = np.asarray(instance["times"], dtype=float)
    topo = _topo_order(n, in_deg, adj)
    fwd = np.zeros(n, dtype=float)
    for u in topo:
        fwd[u] = times[u]
        for v in adj[u]:
            if fwd[u] + times[v] > fwd[v]:
                fwd[v] = fwd[u] + times[v]
    return fwd


def _backward_longest(instance):
    """Longest path from each node to end."""
    adj, in_deg, n = _build_graph(instance)
    times = np.asarray(instance["times"], dtype=float)
    topo = _topo_order(n, in_deg, adj)
    bwd = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ = 0.0
        for v in adj[u]:
            if bwd[v] > max_succ:
                max_succ = bwd[v]
        bwd[u] = times[u] + max_succ
    return bwd


def _successor_counts(instance):
    """Count all (direct and indirect) successors for each task."""
    adj, in_deg, n = _build_graph(instance)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        visited = set()
        stack = list(adj[i])
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(adj[v])
        counts[i] = len(visited)
    return counts


def _predecessor_counts(instance):
    """Count all (direct and indirect) predecessors for each task."""
    preds = instance["predecessors"]
    n = len(instance["times"])
    counts = np.zeros(n, dtype=int)
    # Build reverse graph
    radj = defaultdict(list)
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                radj[j].append(i)
    for i in range(n):
        visited = set()
        stack = list(radj[i])
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(radj[v])
        counts[i] = len(visited)
    return counts


def _assign_stations(instance, priority_scores, cycle_time):
    """
    Greedy assignment: for each station, repeatedly pick the highest-priority
    eligible task that fits. Returns number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Precompute
    pred_counts = np.zeros(n, dtype=int)
    for i in range(n):
        if i in preds:
            pred_counts[i] = len(preds[i])
    
    succ_map = defaultdict(list)
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                succ_map[i].append(j)
    
    assigned = np.zeros(n, dtype=bool)
    rem_preds = pred_counts.copy()
    
    station_time = 0.0
    num_stations = 1
    
    # Pre-sort indices by priority descending for faster max lookup
    # But we need dynamic eligibility, so we just scan
    
    while assigned.sum() < n:
        # Find eligible tasks
        eligible = [i for i in range(n) if not assigned[i] and rem_preds[i] == 0]
        
        if not eligible:
            # Should not happen for valid input, but break to avoid infinite loop
            break
        
        # Among eligible, find those that fit in current station
        remaining_capacity = cycle_time - station_time
        fit = [i for i in eligible if times[i] <= remaining_capacity + 1e-9]
        
        if fit:
            # Pick highest priority
            best = max(fit, key=lambda i: priority_scores[i])
        else:
            # Close current station, open new one
            station_time = 0.0
            num_stations += 1
            # In a fresh station, all eligible should fit (assuming times <= cycle_time)
            fit = eligible
            best = max(fit, key=lambda i: priority_scores[i])
        
        assigned[best] = True
        station_time += times[best]
        
        # Update successors
        for s in succ_map[best]:
            rem_preds[s] -= 1
    
    return num_stations


def _priority_critical_path(instance):
    """Priority = longest path from start through task to end (total criticality)."""
    fwd = _forward_longest(instance)
    bwd = _backward_longest(instance)
    times = np.asarray(instance["times"], dtype=float)
    return fwd + bwd - times


def _priority_rank(instance):
    """Priority = longest path from task to end (rank)."""
    return _backward_longest(instance)


def _priority_time(instance):
    """Priority = task time (LPT)."""
    return np.asarray(instance["times"], dtype=float)


def _priority_successor_weighted(instance):
    """Priority = time * (1 + successor_count)."""
    times = np.asarray(instance["times"], dtype=float)
    succ = _successor_counts(instance).astype(float)
    return times * (1.0 + succ)


def _priority_composite(instance):
    """
    Composite priority:
    - 0.4 * normalized critical path
    - 0.3 * normalized time
    - 0.2 * normalized successor count
    - 0.1 * normalized (1 / (1 + pred_count))
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    def norm(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    crit = norm(_priority_critical_path(instance))
    time_norm = norm(times)
    succ = norm(_successor_counts(instance).astype(float))
    pred = _predecessor_counts(instance).astype(float)
    pred_inv = norm(1.0 / (1.0 + pred))
    
    return 0.4 * crit + 0.3 * time_norm + 0.2 * succ + 0.1 * pred_inv


def _priority_time_rank_product(instance):
    """Priority = time * rank."""
    times = np.asarray(instance["times"], dtype=float)
    rank = _backward_longest(instance)
    return times * rank


def _priority_min_pos(instance):
    """
    Minimum position: prioritize tasks that have many successors relative to their time.
    This tends to open paths early.
    """
    times = np.asarray(instance["times"], dtype=float)
    succ = _successor_counts(instance).astype(float)
    # Higher successors and lower time = higher priority
    # Use: succ / (1 + time)
    return succ / (1.0 + times)


def _priority_fwd_rank_weighted(instance):
    """
    Priority = forward_rank * (1 + transitive_successor_count).
    This weights critical-path position by downstream urgency.
    """
    fwd = _forward_longest(instance)
    succ = _successor_counts(instance).astype(float)
    # Normalize fwd to [0,1] to make the product well-scaled
    fwd_min, fwd_max = fwd.min(), fwd.max()
    if fwd_max - fwd_min < 1e-9:
        fwd_norm = np.zeros_like(fwd)
    else:
        fwd_norm = (fwd - fwd_min) / (fwd_max - fwd_min)
    return fwd_norm * (1.0 + succ)


def _priority_soft_critical(instance):
    """
    Smooth surrogate: use log-sum-exp over successor paths instead of hard max.
    Priority = time + logsumexp(over successors: bwd[v] - time[v])
    This gives a softer critical path that accounts for multiple near-critical branches.
    Uses adaptive temperature based on the spread of successor values.
    """
    adj, in_deg, n = _build_graph(instance)
    times = np.asarray(instance["times"], dtype=float)
    topo = _topo_order(n, in_deg, adj)
    
    bwd = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if adj[u]:
            succ_vals = np.array([bwd[v] - times[v] for v in adj[u]], dtype=float)
            # Adaptive temperature: larger spread -> larger temp for smoother aggregation
            spread = succ_vals.max() - succ_vals.min()
            temp = max(1.0, spread * 0.5)
            max_val = succ_vals.max()
            if max_val > -1e15:
                lse = max_val + np.log(np.sum(np.exp((succ_vals - max_val) / temp))) * temp
            else:
                lse = 0.0
            bwd[u] = times[u] + lse
        else:
            bwd[u] = times[u]
    
    return bwd


def _priority_rank_plus_time(instance):
    """Priority = rank + 0.5 * time. Balances path length and individual task size."""
    rank = _backward_longest(instance)
    times = np.asarray(instance["times"], dtype=float)
    return rank + 0.5 * times


def _priority_critical_time_ratio(instance):
    """
    Priority = critical_path / (1 + time). 
    Favors tasks on long paths that aren't too large individually.
    """
    crit = _priority_critical_path(instance)
    times = np.asarray(instance["times"], dtype=float)
    return crit / (1.0 + times)


def _priority_critical_path_density(instance):
    """
    Critical Path Density: priority = time / (1 + slack),
    where slack = cycle_time - (time + longest_downstream_path).
    This prioritizes tasks that have the least flexibility in scheduling
    relative to their duration. Tasks with high time and low slack get
    the highest priority, forcing the greedy algorithm to place them early.
    """
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Longest downstream path from each task (excluding itself)
    adj, in_deg, n = _build_graph(instance)
    topo = _topo_order(n, in_deg, adj)
    downstream = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if adj[u]:
            downstream[u] = max(downstream[v] + times[v] for v in adj[u])
        else:
            downstream[u] = 0.0
    
    # Slack = cycle_time - (own_time + downstream_path)
    # Clamp to avoid negative slack (which would indicate infeasibility)
    slack = cycle_time - (times + downstream)
    slack = np.maximum(slack, 1e-6)  # Ensure positive denominator
    
    # Density = time / (1 + slack)
    # Higher time and lower slack -> higher priority
    density = times / (1.0 + slack)
    
    return density


def priority(instance):
    """
    Score each assembly task. Stations are filled with the highest-priority 
    eligible task that fits the cycle time.
    
    Uses a direct approach: generate several structure-derived priority vectors,
    evaluate each with the greedy assignment, and return the best.
    """
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Generate candidate priority vectors (direct, structure-based)
    candidates = [
        ("critical_path", _priority_critical_path(instance)),
        ("rank", _priority_rank(instance)),
        ("time", _priority_time(instance)),
        ("succ_weighted", _priority_successor_weighted(instance)),
        ("composite", _priority_composite(instance)),
        ("time_rank", _priority_time_rank_product(instance)),
        ("min_pos", _priority_min_pos(instance)),
        ("fwd_rank_weighted", _priority_fwd_rank_weighted(instance)),
        ("soft_critical", _priority_soft_critical(instance)),
        ("rank_plus_time", _priority_rank_plus_time(instance)),
        ("critical_time_ratio", _priority_critical_time_ratio(instance)),
        ("critical_path_density", _priority_critical_path_density(instance)),
    ]
    
    best_priority = np.asarray(times, dtype=float)
    best_num_stations = float('inf')
    
    for name, p in candidates:
        if time.time() > _DEADLINE:
            break
        try:
            ns = _assign_stations(instance, p, cycle_time)
            if ns < best_num_stations:
                best_num_stations = ns
                best_priority = p
        except Exception:
            continue
    
    return best_priority
# EVOLVE-BLOCK-END
