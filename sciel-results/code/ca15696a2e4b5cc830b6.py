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
    
    # Track which station each task is assigned to
    station_of_task = np.full(n, -1, dtype=int)
    # Track tasks in each station (for local search)
    station_tasks = [[]]  # station 0
    
    while assigned.sum() < n:
        # Find eligible tasks
        eligible = [i for i in range(n) if not assigned[i] and rem_preds[i] == 0]
        
        if not eligible:
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
            station_tasks.append([])
            fit = eligible
            best = max(fit, key=lambda i: priority_scores[i])
        
        assigned[best] = True
        station_time += times[best]
        station_of_task[best] = num_stations - 1
        station_tasks[num_stations - 1].append(best)
        
        # Update successors
        for s in succ_map[best]:
            rem_preds[s] -= 1
    
    # Local search: try to move tasks from later stations to earlier ones
    improved = True
    max_iter = 50
    iter_count = 0
    while improved and iter_count < max_iter and time.time() < _DEADLINE - 5:
        improved = False
        iter_count += 1
        # Try moving each task to an earlier station if possible
        for s in range(num_stations - 1, 0, -1):
            if time.time() > _DEADLINE - 5:
                break
            tasks_in_s = list(station_tasks[s])
            for t in tasks_in_s:
                # Check if t can be moved to any earlier station
                for s2 in range(s - 1, -1, -1):
                    # Check precedence: all predecessors of t must be in stations < s2 or == s2
                    can_move = True
                    for p in preds.get(t, []):
                        if station_of_task[p] >= s2:
                            can_move = False
                            break
                    if not can_move:
                        continue
                    # Check capacity
                    cap_s2 = cycle_time - sum(times[x] for x in station_tasks[s2])
                    if times[t] <= cap_s2 + 1e-9:
                        # Move t from s to s2
                        station_tasks[s].remove(t)
                        station_tasks[s2].append(t)
                        station_of_task[t] = s2
                        improved = True
                        break
    
    # Recount stations (some may now be empty)
    non_empty = [s for s in range(num_stations) if len(station_tasks[s]) > 0]
    return len(non_empty)


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
    """
    adj, in_deg, n = _build_graph(instance)
    times = np.asarray(instance["times"], dtype=float)
    topo = _topo_order(n, in_deg, adj)
    
    # Temperature for soft-max
    temp = 2.0
    
    bwd = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if adj[u]:
            succ_vals = np.array([bwd[v] - times[v] for v in adj[u]], dtype=float)
            # logsumexp with temperature
            max_val = succ_vals.max()
            if max_val > -1e15:  # avoid log(0)
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
    ]
    
    best_priority = np.asarray(times, dtype=float)
    best_num_stations = float('inf')
    
    for name, p in candidates:
        if time.time() > _DEADLINE - 2:
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
