"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_successor_adjacency(n, predecessors):
    """Build successor adjacency list from predecessors."""
    succ = [[] for _ in range(n)]
    for j, preds in enumerate(predecessors):
        for i in preds:
            succ[i].append(j)
    return succ


def _topological_order(n, predecessors):
    """Return tasks in topological order (predecessors before successors)."""
    in_degree = [len(preds) for preds in predecessors]
    queue = [i for i in range(n) if in_degree[i] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for j in range(n):
            if node in predecessors[j]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)
    return order


def _compute_downstream_metrics(n, predecessors, times):
    """Compute longest downstream path and successor counts."""
    succ = _build_successor_adjacency(n, predecessors)
    
    # Topological order
    in_degree = [len(preds) for preds in predecessors]
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    while queue:
        node = queue.pop(0)
        top_order.append(node)
        for j in range(n):
            if node in predecessors[j]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)
    
    # Longest path downstream (including own time)
    longest_down = [0.0] * n
    for node in reversed(top_order):
        max_down = 0.0
        for s in succ[node]:
            max_down = max(max_down, longest_down[s])
        longest_down[node] = times[node] + max_down
    
    # Successor count
    succ_count = [len(succ[i]) for i in range(n)]
    
    # Transitive successor count (number of all descendants)
    trans_succ = [0] * n
    for node in reversed(top_order):
        for s in succ[node]:
            trans_succ[node] += 1 + trans_succ[s]
    
    return np.array(longest_down, dtype=float), np.array(succ_count, dtype=float), np.array(trans_succ, dtype=float)


def _simulate_greedy(n, predecessors, cycle_time, priority_vec, times):
    """Simulate greedy station assignment and return number of stations."""
    assigned = [False] * n
    remaining = set(range(n))
    n_stations = 0
    
    while remaining:
        n_stations += 1
        station_load = 0.0
        # Find eligible tasks (all predecessors assigned)
        eligible = []
        for task in remaining:
            if all(not p in remaining for p in predecessors[task]):
                if station_load + times[task] <= cycle_time:
                    eligible.append(task)
        
        if not eligible:
            # No task fits; this shouldn't happen if cycle_time >= max time
            break
        
        # Pick highest priority eligible task
        best_task = max(eligible, key=lambda t: priority_vec[t])
        remaining.discard(best_task)
        assigned[best_task] = True
        station_load += times[best_task]
    
    return n_stations


def priority(instance):
    """Score each assembly task for priority-based station assignment."""
    start_time = time.time()
    
    try:
        times = np.asarray(instance["times"], dtype=float)
        predecessors = instance["predecessors"]
        cycle_time = instance["cycle_time"]
        n = len(times)
        
        # Compute features
        longest_down, succ_count, trans_succ = _compute_downstream_metrics(n, predecessors, times)
        
        # Normalize features
        def norm(x):
            mx = np.max(x)
            if mx > 0:
                return x / mx
            return np.zeros_like(x)
        
        ld_norm = norm(longest_down)
        sc_norm = norm(succ_count)
        ts_norm = norm(trans_succ)
        t_norm = norm(times)
        
        # Try several weight combinations and pick the best via simulation
        weight_configs = [
            (0.5, 0.3, 0.2, 0.0),
            (0.4, 0.3, 0.2, 0.1),
            (0.3, 0.4, 0.2, 0.1),
            (0.4, 0.3, 0.1, 0.2),
            (0.3, 0.3, 0.2, 0.2),
            (0.5, 0.2, 0.1, 0.2),
            (0.4, 0.3, 0.1, 0.1),
            (0.3, 0.4, 0.1, 0.1),
            (0.2, 0.4, 0.2, 0.1),
            (0.3, 0.3, 0.2, 0.0),
            (0.6, 0.2, 0.1, 0.1),
            (0.4, 0.4, 0.1, 0.1),
            (0.5, 0.3, 0.1, 0.1),
            (0.3, 0.3, 0.3, 0.1),
        ]
        
        best_priority = None
        best_stations = float('inf')
        
        for (wt, wld, wsc, wts) in weight_configs:
            priority_vec = (wt * t_norm + wld * ld_norm + wsc * sc_norm + wts * ts_norm) * np.max(times)
            
            # Simulate to get station count
            n_stations = _simulate_greedy(n, predecessors, cycle_time, priority_vec, times)
            
            if n_stations < best_stations:
                best_stations = n_stations
                best_priority = priority_vec
        
        if best_priority is None:
            # Fallback
            best_priority = 0.5 * times + 0.3 * ld_norm * np.max(times) + 0.2 * sc_norm * np.max(times)
        
        return best_priority
    
    except Exception:
        # Fallback to simple time-based priority
        return np.asarray(instance["times"], dtype=float)
# EVOLVE-BLOCK-END
