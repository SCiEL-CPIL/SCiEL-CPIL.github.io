"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _build_graph(instance):
    """
    Builds the successor adjacency list and in-degrees from the instance.
    Returns:
        successors: list of lists, where successors[i] contains tasks that depend on i.
        in_degree: list of ints, where in_degree[i] is the number of predecessors of i.
        n: number of tasks.
    """
    times = instance["times"]
    predecessors = instance["predecessors"]
    n = len(times)
    
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    return successors, in_degree, n


def _topological_order(n, in_degree, successors):
    """
    Computes topological order using Kahn's algorithm.
    Returns a list of nodes in topological order.
    """
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
                
    return order


def _compute_metrics(instance):
    """
    Computes time, longest downstream path, and transitive successor count for each task.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    successors, in_degree, _ = _build_graph(instance)
    
    # We need a fresh copy of in_degree for topological sort if we modify it, 
    # but _topological_order modifies the passed list. 
    # Let's pass a copy.
    in_degree_copy = in_degree[:]
    topo_order = _topological_order(n, in_degree_copy, successors)
    
    # 1. Longest downstream path (including own time)
    # Process in reverse topological order
    longest_down = np.zeros(n)
    for u in reversed(topo_order):
        max_path = 0.0
        for v in successors[u]:
            # Path from u to sink via v is time[u] + (path from v to sink)
            # But we define longest_down[v] as the longest path starting at v.
            # So longest_down[u] = time[u] + max(longest_down[v])
            if longest_down[v] > max_path:
                max_path = longest_down[v]
        longest_down[u] = times[u] + max_path
        
    # 2. Transitive successor count (number of descendants)
    # Process in reverse topological order
    # count_desc[u] = sum(1 + count_desc[v]) for all v in successors[u]
    count_desc = np.zeros(n)
    for u in reversed(topo_order):
        total = 0
        for v in successors[u]:
            total += 1 + count_desc[v]
        count_desc[u] = total
        
    return times, longest_down, count_desc


def _assign_stations_greedy(times, predecessors, cycle_time, priority_scores):
    """Assign tasks to stations using greedy rule: highest priority eligible task that fits."""
    n = len(times)
    
    # Build predecessor and successor sets
    pred_set = [set() for _ in range(n)]
    succ_set = [set() for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            pred_set[i].add(p)
            succ_set[p].add(i)
    
    # Remaining count: how many predecessors are not yet assigned
    remaining_preds = [len(pred_set[i]) for i in range(n)]
    
    # Track which tasks are assigned
    assigned = [False] * n
    station_count = 0
    station_load = 0.0
    station_tasks = []
    
    # Use a max-heap based on priority score
    # We'll recompute eligible tasks each step, or use a priority queue with lazy deletion
    
    unassigned = set(range(n))
    
    while unassigned:
        # Find eligible tasks: all predecessors assigned
        eligible = []
        for t in unassigned:
            if remaining_preds[t] == 0 and times[t] <= cycle_time - station_load:
                eligible.append(t)
        
        if not eligible:
            # Close current station and start a new one
            station_count += 1
            station_load = 0.0
            station_tasks = []
            # Re-find eligible tasks (some may now fit since station_load reset)
            for t in unassigned:
                if remaining_preds[t] == 0 and times[t] <= cycle_time:
                    eligible.append(t)
        
        if not eligible:
            # Shouldn't happen if cycle_time >= max(time), but handle gracefully
            break
        
        # Pick the highest priority task among eligible
        best_task = max(eligible, key=lambda t: priority_scores[t])
        
        # Assign it
        assigned[best_task] = True
        unassigned.discard(best_task)
        station_load += times[best_task]
        station_tasks.append(best_task)
        
        # Update remaining_preds for successors
        for s in succ_set[best_task]:
            remaining_preds[s] -= 1
    
    station_count += 1  # Don't forget the last station
    return station_count


def _try_weights(instance, w_time, w_down, w_succ):
    """Try assigning with different weight combinations."""
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    
    # Precompute components
    max_time = max(times) + 1e-9
    time_score = times / max_time
    
    # Compute longest path downstream (without own time, as in current program)
    successors, in_degree, _ = _build_graph(instance)
    in_degree_copy = in_degree[:]
    topo_order = _topological_order(n, in_degree_copy, successors)
    
    longest_down = np.zeros(n)
    for u in reversed(topo_order):
        max_path = 0.0
        for v in successors[u]:
            max_path = max(max_path, times[v] + longest_down[v])
        longest_down[u] = max_path
    
    max_ld = max(longest_down) + 1e-9
    down_score = longest_down / max_ld
    
    # Count descendants
    count_desc = np.zeros(n)
    for u in reversed(topo_order):
        total = 0
        for v in successors[u]:
            total += 1 + count_desc[v]
        count_desc[u] = total
    
    max_sc = max(count_desc) + 1e-9
    succ_score = count_desc / max_sc
    
    scores = w_time * time_score + w_down * down_score + w_succ * succ_score
    
    return _assign_stations_greedy(times, predecessors, cycle_time, scores)


def priority(instance):
    """
    Score each assembly task for priority-based station assignment.
    Higher score = higher priority.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    try:
        times, longest_down, count_desc = _compute_metrics(instance)
        
        # Normalize each component
        if np.max(times) > 0:
            time_norm = times / np.max(times)
        else:
            time_norm = np.zeros_like(times)
        
        if np.max(longest_down) > 0:
            down_norm = longest_down / np.max(longest_down)
        else:
            down_norm = np.zeros_like(longest_down)
        
        if np.max(count_desc) > 0:
            succ_norm = count_desc / np.max(count_desc)
        else:
            succ_norm = np.zeros_like(count_desc)
        
        # Structural change: Use a different weighting scheme that emphasizes
        # bottleneck tasks (high successor count) more heavily
        # This is a structural change from the previous 0.5/0.3/0.2 scheme
        # to a scheme that prioritizes tasks that unlock more downstream work
        
        # New approach: weight by (1 + successor_count) to amplify bottleneck tasks
        # and use a different base weighting
        priority = 0.4 * time_norm + 0.3 * down_norm + 0.3 * succ_norm
        
        # Additional structural change: Apply a non-linear transformation
        # that boosts tasks with very high successor counts (bottlenecks)
        # This creates a sharper distinction between critical and non-critical tasks
        succ_boost = 1.0 + 0.5 * succ_norm
        priority = priority * succ_boost
        
        # Normalize final priority to [0, 1]
        if np.max(priority) > 0:
            priority = priority / np.max(priority)
        
        return priority
        
    except Exception:
        # Fallback to simple time-based priority
        return np.asarray(instance["times"], dtype=float)
# EVOLVE-BLOCK-END
