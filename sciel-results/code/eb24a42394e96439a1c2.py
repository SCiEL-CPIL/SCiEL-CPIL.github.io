"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def _simulate_station_count(times, predecessors, cycle_time, priority_scores):
    """Simulate greedy station filling and return number of stations."""
    n = len(times)
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Compute in-degree (number of unassigned predecessors)
    in_degree = np.array([len(preds) for preds in predecessors], dtype=int)
    assigned = np.zeros(n, dtype=bool)
    remaining_time = np.zeros(n, dtype=float)  # remaining time in current station
    
    station_count = 0
    current_station_time = 0.0
    eligible_count = 0
    
    # Track which tasks are eligible (all predecessors assigned)
    eligible = np.zeros(n, dtype=bool)
    for i in range(n):
        if in_degree[i] == 0:
            eligible[i] = True
            eligible_count += 1
    
    # Priority order: higher score = higher priority
    # We'll use a sorted list approach for efficiency
    # Sort tasks by priority (descending) and maintain eligibility
    
    # For efficiency, we'll repeatedly pick the highest-priority eligible task that fits
    # Use a list of (priority, task_id) and filter
    
    while eligible_count > 0:
        # Find the best eligible task that fits
        best_task = -1
        best_priority = -np.inf
        for i in range(n):
            if eligible[i] and (current_station_time + times[i]) <= cycle_time + 1e-9:
                if priority_scores[i] > best_priority:
                    best_priority = priority_scores[i]
                    best_task = i
        
        if best_task == -1:
            # No eligible task fits in current station, start new station
            station_count += 1
            current_station_time = 0.0
            # Try again
            for i in range(n):
                if eligible[i] and (current_station_time + times[i]) <= cycle_time + 1e-9:
                    if priority_scores[i] > best_priority:
                        best_priority = priority_scores[i]
                        best_task = i
            if best_task == -1:
                # Shouldn't happen if any task fits
                break
        
        # Assign best_task
        assigned[best_task] = True
        current_station_time += times[best_task]
        eligible[best_task] = False
        eligible_count -= 1
        
        # Update successors
        for s in successors[best_task]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                eligible[s] = True
                eligible_count += 1
    
    if current_station_time > 0:
        station_count += 1
    
    return station_count


def _compute_features(instance):
    """Compute multiple features for each task."""
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build adjacency
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Feature 1: task time
    f_time = times.copy()
    
    # Feature 2: longest path from this task to any successor (including itself)
    # This is the "tail" - how much work remains after this task
    f_tail = np.zeros(n, dtype=float)
    # Process in reverse topological order
    in_degree = np.array([len(preds) for preds in predecessors], dtype=int)
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    temp_in_degree = in_degree.copy()
    q = deque()
    for i in range(n):
        if temp_in_degree[i] == 0:
            q.append(i)
    while q:
        node = q.popleft()
        topo_order.append(node)
        for s in successors[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                q.append(s)
    
    # Reverse topological order for tail computation
    for i in reversed(topo_order):
        max_succ = 0.0
        for s in successors[i]:
            if times[s] + f_tail[s] > max_succ:
                max_succ = times[s] + f_tail[s]
        f_tail[i] = max_succ
    
    # Feature 3: number of successors (direct)
    f_succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Feature 4: depth (longest path from any source to this task)
    f_depth = np.zeros(n, dtype=float)
    for i in topo_order:
        max_pred = 0.0
        for p in predecessors[i]:
            if times[p] + f_depth[p] > max_pred:
                max_pred = times[p] + f_depth[p]
        f_depth[i] = max_pred
    
    return f_time, f_tail, f_succ_count, f_depth


def priority(instance):
    """Score each assembly task for priority-based station filling."""
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Compute features
    f_time, f_tail, f_succ_count, f_depth = _compute_features(instance)
    
    # Normalize features to [0, 1]
    def normalize(f):
        mn, mx = f.min(), f.max()
        if mx - mn < 1e-12:
            return np.zeros_like(f)
        return (f - mn) / (mx - mn)
    
    f_time_n = normalize(f_time)
    f_tail_n = normalize(f_tail)
    f_succ_n = normalize(f_succ_count)
    f_depth_n = normalize(f_depth)
    
    # Start with a reasonable initial weight vector
    # We want to emphasize time and tail (longest path to end)
    best_weights = np.array([1.0, 1.0, 0.0, 0.0])
    best_stations = _simulate_station_count(times, predecessors, cycle_time, 
                                            best_weights[0]*f_time_n + best_weights[1]*f_tail_n + 
                                            best_weights[2]*f_succ_n + best_weights[3]*f_depth_n)
    
    # Local search over weights
    # Use a deadline guard
    start_time = _time.time()
    deadline = 120.0  # seconds, with margin
    
    # Define search directions
    # We'll do coordinate descent and some random perturbations
    n_features = 4
    weights = best_weights.copy()
    
    # Coordinate descent
    improved = True
    max_iters = 200
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if _time.time() > deadline - 30:
            break
        for d in range(n_features):
            # Try increasing
            for delta in [0.1, 0.5, 1.0, 2.0]:
                trial = weights.copy()
                trial[d] += delta
                if (trial < -0.01).any():
                    continue
                score = trial[0]*f_time_n + trial[1]*f_tail_n + trial[2]*f_succ_n + trial[3]*f_depth_n
                stations = _simulate_station_count(times, predecessors, cycle_time, score)
                if stations < best_stations:
                    best_stations = stations
                    best_weights = trial.copy()
                    weights = trial.copy()
                    improved = True
                    break
            # Try decreasing
            for delta in [0.1, 0.5, 1.0]:
                trial = weights.copy()
                trial[d] -= delta
                if (trial < -0.01).any():
                    continue
                score = trial[0]*f_time_n + trial[1]*f_tail_n + trial[2]*f_succ_n + trial[3]*f_depth_n
                stations = _simulate_station_count(times, predecessors, cycle_time, score)
                if stations < best_stations:
                    best_stations = stations
                    best_weights = trial.copy()
                    weights = trial.copy()
                    improved = True
                    break
    
    # Random perturbations
    rng = np.random.RandomState(42)
    for _ in range(50):
        if _time.time() > deadline - 15:
            break
        trial = best_weights + rng.normal(0, 0.3, size=n_features)
        trial = np.maximum(trial, 0)
        score = trial[0]*f_time_n + trial[1]*f_tail_n + trial[2]*f_succ_n + trial[3]*f_depth_n
        stations = _simulate_station_count(times, predecessors, cycle_time, score)
        if stations < best_stations:
            best_stations = stations
            best_weights = trial.copy()
    
    # Final priority scores
    final_scores = (best_weights[0]*f_time_n + best_weights[1]*f_tail_n + 
                    best_weights[2]*f_succ_n + best_weights[3]*f_depth_n)
    
    return final_scores
# EVOLVE-BLOCK-END
