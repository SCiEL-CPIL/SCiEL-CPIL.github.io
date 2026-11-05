"""
Strategy: Direct Local Search on Priority Vector.
Instead of choosing from a fixed set of heuristics, we treat the priority scores as a 
vector to be optimized. We start with a strong "Critical Path" heuristic and then 
perform hill climbing by randomly perturbing priorities and accepting changes that 
reduce the number of stations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _compute_topo_and_metrics(instance):
    """
    Compute topological order, depth, height, and successor counts.
    Returns: topo_order, depth, height, succ_counts
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n_tasks = len(times)
    
    succs = [[] for _ in range(n_tasks)]
    in_deg = np.zeros(n_tasks, dtype=int)
    for i in range(n_tasks):
        for p in preds[i]:
            if p >= 0:
                succs[p].append(i)
                in_deg[i] += 1
    
    q = deque([i for i in range(n_tasks) if in_deg[i] == 0])
    topo = []
    indeg = in_deg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
        
    if len(topo) != n_tasks:
        # Cycle detected, return fallback
        return list(range(n_tasks)), times, times, np.ones(n_tasks)

    # Depth: Longest path from start to u (including u)
    depth = times.copy()
    for u in topo:
        for p in preds[u]:
            if p >= 0:
                if depth[p] + times[p] > depth[u]:
                    depth[u] = depth[p] + times[p]
                    
    # Height: Longest path from u to end (including u)
    height = times.copy()
    for u in reversed(topo):
        max_h = 0
        for v in succs[u]:
            if height[v] > max_h:
                max_h = height[v]
        height[u] = times[u] + max_h
        
    # Successor count
    succ_counts = np.array([len(s) for s in succs], dtype=float)
    
    return topo, depth, height, succ_counts

def _greedy_station_count(instance, scores):
    """
    Simulate the greedy line balancing process using the given scores.
    Returns the number of stations.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    assigned = np.zeros(n_tasks, dtype=bool)
    station_count = 0
    current_load = 0
    current_tasks = []
    
    # Pre-sort indices by score descending for faster lookup? 
    # Actually, we need to check eligibility dynamically.
    # For n=30, O(n^2) per station is fine.
    
    # To speed up finding the best eligible task, we can keep a list of unassigned tasks
    # and filter them.
    
    while np.any(~assigned):
        # Find best eligible task for current station
        best_task = -1
        best_score = -np.inf
        
        # We iterate through all unassigned tasks to find the best eligible one
        # Optimization: We can skip if current_load is 0 and we are starting a new station,
        # but the logic is the same: find max score among eligible.
        
        for t in range(n_tasks):
            if assigned[t]:
                continue
            
            # Check eligibility: all predecessors must be assigned
            # Note: In standard ALB greedy, a task is eligible if all its predecessors 
            # are in previous stations OR in the current station (already assigned).
            # Since we assign one by one, "assigned" covers both.
            eligible = True
            for p in preds[t]:
                if p >= 0 and not assigned[p]:
                    eligible = False
                    break
                    
            if not eligible:
                continue
            
            if current_load + times[t] > cycle_time:
                continue
                
            if scores[t] > best_score:
                best_score = scores[t]
                best_task = t
                
        if best_task == -1:
            # No task fits in current station. Close it.
            if current_load > 0:
                station_count += 1
                current_load = 0
                current_tasks = []
                # Try again with new station (loop continues)
                # To prevent infinite loop if a single task is too big, we handle it below
                # But if best_task is -1 and current_load is 0, it means no task is eligible 
                # or no task fits. 
                # If no task is eligible, we have a cycle or logic error.
                # If no task fits, we have a task > cycle_time.
                # Let's force assign the highest priority eligible task regardless of fit?
                # No, the problem implies tasks fit. 
                # However, if current_load == 0 and best_task == -1, it means NO task is eligible.
                # This can happen if there's a cycle. We break out.
                # Or it means all eligible tasks are too big.
                # Let's just break to avoid infinite loop.
                break
            else:
                # This shouldn't happen if we just closed a station and there are unassigned tasks
                # unless the first task to be picked is ineligible (cycle) or too big.
                # If too big, we can't assign it.
                # We'll break.
                break
        else:
            assigned[best_task] = True
            current_load += times[best_task]
            current_tasks.append(best_task)
            
    if current_load > 0:
        station_count += 1
        
    return station_count

def _initial_priority(instance):
    """
    Generate a strong initial priority score based on Critical Path (Height) 
    and Depth, plus successor count as a tie-breaker.
    """
    topo, depth, height, succ_counts = _compute_topo_and_metrics(instance)
    
    # Normalize components
    d_max = max(depth) if max(depth) > 0 else 1
    h_max = max(height) if max(height) > 0 else 1
    s_max = max(succ_counts) if max(succ_counts) > 0 else 1
    
    # Weighted sum: Height is usually the strongest predictor for ALB (Longest Processing Time / Critical Path)
    # We combine them to create a robust baseline.
    scores = 0.6 * (height / h_max) + 0.3 * (depth / d_max) + 0.1 * (succ_counts / s_max)
    
    return scores

def priority(instance):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    times = instance["times"]
    n_tasks = len(times)
    
    # 1. Generate initial strong heuristic
    best_scores = _initial_priority(instance)
    best_stations = _greedy_station_count(instance, best_scores)
    
    # 2. Local Search (Hill Climbing with Random Restarts/Perturbations)
    # We will try to improve the score vector by randomly perturbing some tasks' priorities.
    # Accept if station count decreases.
    # If station count is same, we might accept with a small probability to explore (Simulated Annealing lite).
    # Given the small n=30, we can afford many iterations.
    
    rng = np.random.RandomState(42)
    current_scores = best_scores.copy()
    current_stations = best_stations
    
    # Parameters
    alpha = 1.0  # Perturbation magnitude
    min_alpha = 0.01
    
    # Time budget for search
    search_deadline = start_time + 140
    
    while time.time() < search_deadline:
        # Perturb: Pick a few random tasks and adjust their scores
        n_perturb = rng.randint(1, 5)
        indices = rng.choice(n_tasks, n_perturb, replace=False)
        
        new_scores = current_scores.copy()
        for idx in indices:
            # Add Gaussian noise
            new_scores[idx] += rng.normal(0, alpha)
            # Keep scores positive? Not strictly necessary, but helps with relative ordering stability
            # Actually, just relative order matters.
            
        # Evaluate
        new_stations = _greedy_station_count(instance, new_scores)
        
        # Acceptance Criteria
        if new_stations < current_stations:
            # Improvement
            current_scores = new_scores
            current_stations = new_stations
            if current_stations < best_stations:
                best_scores = current_scores.copy()
                best_stations = current_stations
                # Restart alpha if we found a new best?
                # alpha = 1.0 
            # Decay alpha slightly on improvement? Or keep it?
            # Let's decay slowly to refine.
            alpha = max(min_alpha, alpha * 0.95)
            
        elif new_stations == current_stations:
            # Neutral move. Accept with probability p to escape local optima
            # p = 0.1 * (alpha / 1.0)
            if rng.random() < 0.05:
                current_scores = new_scores
                # current_stations unchanged
                
        else:
            # Worse. Reject.
            # Increase alpha slightly to jump out?
            alpha = min(5.0, alpha * 1.1)
            
        # Optional: If we are stuck for a long time (no improvement in best_stations), 
        # we could do a full restart from a different random heuristic, 
        # but hill climbing on the vector is usually very effective for n=30.
        
    return best_scores

# EVOLVE-BLOCK-END
