"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _topological_levels(times, predecessors):
    """Return list of levels where each level contains tasks that can be done in parallel."""
    n = len(times)
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            in_degree[i] += 1
            successors[p].append(i)
    
    # BFS-based level assignment
    levels = []
    current = [i for i in range(n) if in_degree[i] == 0]
    remaining_in_degree = in_degree.copy()
    
    while current:
        levels.append(current)
        next_level = []
        for node in current:
            for succ in successors[node]:
                remaining_in_degree[succ] -= 1
                if remaining_in_degree[succ] == 0:
                    next_level.append(succ)
        current = next_level
    
    # Assign level index to each task
    task_level = np.zeros(n, dtype=int)
    for lvl, nodes in enumerate(levels):
        for node in nodes:
            task_level[node] = lvl
    
    return task_level, levels


def _compute_rank(times, predecessors, successors_map):
    """Compute rank (longest path to end) for each task."""
    n = len(times)
    rank = np.zeros(n, dtype=float)
    
    # Process in reverse topological order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
    
    # Get reverse topological order using Kahn's algorithm in reverse
    # Actually, let's compute rank by processing from sinks to sources
    out_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            out_degree[p] += 1
    
    # Start from nodes with no successors (sinks)
    from collections import deque
    q = deque([i for i in range(n) if out_degree[i] == 0])
    rank = np.zeros(n, dtype=float)
    processed = np.zeros(n, dtype=bool)
    
    while q:
        node = q.popleft()
        processed[node] = True
        for pred in predecessors[node]:
            if not processed[pred]:
                rank[pred] = max(rank[pred], times[pred] + rank[node])
                out_degree[pred] -= 1
                if out_degree[pred] == 0:
                    q.append(pred)
    
    # For any unprocessed nodes (shouldn't happen in valid DAG), compute remaining
    for i in range(n):
        if not processed[i]:
            for succ in successors_map[i]:
                rank[i] = max(rank[i], times[i] + rank[succ])
    
    return rank


def _compute_successors_map(predecessors):
    n = len(predecessors)
    succ_map = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            succ_map[p].append(i)
    return succ_map


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance.get("cycle_time", 40)
    n = len(times)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Build successors map
    succ_map = _compute_successors_map(predecessors)
    
    # Compute ranks (longest path from task to end)
    rank = _compute_rank(times, predecessors, succ_map)
    
    # Compute in-degree for each task
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
    
    # Strategy: Use a composite priority that considers:
    # 1. Rank (longest path remaining) - higher rank = more critical
    # 2. Task duration - longer tasks should be scheduled earlier (LPT-like)
    # 3. Number of immediate successors - more successors = more constraining
    
    # Count immediate successors
    num_succ = np.array([len(succ_map[i]) for i in range(n)], dtype=float)
    
    # Normalize components to [0, 1]
    max_rank = rank.max() if n > 0 else 1
    max_time = times.max() if n > 0 else 1
    max_succ = num_succ.max() if n > 0 else 1
    
    if max_rank == 0: max_rank = 1
    if max_time == 0: max_time = 1
    if max_succ == 0: max_succ = 1
    
    norm_rank = rank / max_rank
    norm_time = times / max_time
    norm_succ = num_succ / max_succ
    
    # Composite priority: weight rank heavily, then time, then successors
    # Higher priority = should be scheduled earlier
    priority_score = 0.6 * norm_rank + 0.3 * norm_time + 0.1 * norm_succ
    
    # Try a few different weight combinations and pick the best one
    # by simulating the greedy assignment
    
    best_score = np.inf
    best_priority = priority_score.copy()
    
    weight_sets = [
        (0.6, 0.3, 0.1),
        (0.7, 0.2, 0.1),
        (0.5, 0.3, 0.2),
        (0.8, 0.1, 0.1),
        (0.6, 0.4, 0.0),
        (0.5, 0.5, 0.0),
        (0.4, 0.4, 0.2),
        (0.7, 0.2, 0.1),
        (0.65, 0.25, 0.1),
        (0.55, 0.35, 0.1),
    ]
    
    for w_rank, w_time, w_succ in weight_sets:
        if time.time() > deadline - 5:
            break
        
        p_score = w_rank * norm_rank + w_time * norm_time + w_succ * norm_succ
        
        # Simulate greedy assignment to count stations
        stations = _simulate_assignment(times, predecessors, p_score, cycle_time)
        
        if stations < best_score:
            best_score = stations
            best_priority = p_score.copy()
    
    # If we have time, try more granular weight search
    if time.time() < deadline - 10:
        # Try some additional weight combinations
        for w_rank in np.arange(0.5, 0.85, 0.05):
            if time.time() > deadline - 5:
                break
            for w_time in np.arange(0.05, 0.45, 0.05):
                if time.time() > deadline - 5:
                    break
                w_succ = max(0, 1 - w_rank - w_time)
                p_score = w_rank * norm_rank + w_time * norm_time + w_succ * norm_succ
                
                stations = _simulate_assignment(times, predecessors, p_score, cycle_time)
                
                if stations < best_score:
                    best_score = stations
                    best_priority = p_score.copy()
    
    return best_priority


def _simulate_assignment(times, predecessors, priority_scores, cycle_time):
    """Simulate greedy assignment and return number of stations used."""
    n = len(times)
    
    # In-degree tracking
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
    
    # Available tasks (all predecessors assigned)
    available = set(i for i in range(n) if in_degree[i] == 0)
    assigned = np.zeros(n, dtype=bool)
    
    stations = 0
    current_station_load = 0
    
    while not np.all(assigned):
        if not available:
            break
        
        # Find the highest priority available task that fits in current station
        best_task = -1
        best_prio = -np.inf
        
        for task in available:
            if times[task] <= cycle_time - current_station_load:
                if priority_scores[task] > best_prio:
                    best_prio = priority_scores[task]
                    best_task = task
        
        if best_task == -1:
            # No task fits, start new station
            stations += 1
            current_station_load = 0
            # Retry finding a task for the new station
            for task in available:
                if times[task] <= cycle_time:
                    if priority_scores[task] > best_prio:
                        best_prio = priority_scores[task]
                        best_task = task
        
        if best_task == -1:
            # Shouldn't happen if all times <= cycle_time
            break
        
        # Assign the task
        assigned[best_task] = True
        current_station_load += times[best_task]
        available.discard(best_task)
        
        # Update in-degrees of successors
        for succ in range(n):
            if best_task in predecessors[succ]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    available.add(succ)
    
    # Count the last station if it has any load
    if current_station_load > 0:
        stations += 1
    
    return stations


# EVOLVE-BLOCK-END
