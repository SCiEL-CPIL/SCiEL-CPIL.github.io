"""Combined program: Structure-initialized Simulated Annealing for ALB."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _simulate_greedy(order, times, succ, pred_counts_init, cycle_time):
    """
    Simulate the greedy packing process for a given priority order.
    Optimized to avoid repeated predecessor checks by maintaining counts.
    """
    n = len(times)
    assigned = [False] * n
    # Copy pred counts as they will be modified
    pred_counts = pred_counts_init.copy()
    
    # Initial available tasks
    available = [i for i in range(n) if pred_counts[i] == 0]
    
    # Rank array for quick lookup of priority
    # Lower rank index = higher priority
    rank = [0] * n
    for idx, task in enumerate(order):
        rank[task] = idx
        
    stations = 0
    current_load = 0.0
    assigned_count = 0
    
    # We need to efficiently pick the highest priority task from 'available' that fits.
    # Since N is small (30), we can just scan 'available'.
    # To make it slightly faster, we could keep 'available' sorted, but insertion/removal is O(N).
    # Scanning is O(K) where K is size of available.
    
    while assigned_count < n:
        best_task = -1
        best_rank = n + 1
        remaining_space = cycle_time - current_load
        
        # Find best task in available
        # We iterate through available list
        for t in available:
            if times[t] <= remaining_space + 1e-9:
                if rank[t] < best_rank:
                    best_rank = rank[t]
                    best_task = t
                    
        if best_task != -1:
            # Assign the task
            assigned[best_task] = True
            assigned_count += 1
            current_load += times[best_task]
            
            # Remove from available
            # O(N) removal
            available.remove(best_task)
            
            # Update successors
            for s in succ[best_task]:
                pred_counts[s] -= 1
                if pred_counts[s] == 0:
                    available.append(s)
        else:
            # No task fits in the current station
            if current_load > 1e-9:
                stations += 1
                current_load = 0.0
            else:
                # Stuck, should not happen if max time <= cycle_time
                # But if it does, we must break to avoid infinite loop
                # In valid ALB instances, this shouldn't occur unless cycle_time < max(time)
                break

    if current_load > 1e-9:
        stations += 1
        
    return stations

def priority(instance):
    """
    Uses Simulated Annealing initialized with structural heuristics 
    to find the best static priority order.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin for 150s limit
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph
    pred = instance["predecessors"]
    succ = [[] for _ in range(n)]
    pred_counts_init = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
            pred_counts_init[i] += 1
            
    # --- Initialization: Structural Heuristic ---
    # Compute longest path to terminal (dist)
    # Compute total transitive successors (total_succ)
    
    in_degree = pred_counts_init.copy()
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    while queue:
        # Sort to ensure deterministic topological sort if multiple choices
        # Though for dist calculation, any topological order works as long as we process in reverse
        queue.sort()
        u = queue.pop(0)
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) != n:
        # Cycle detected, fallback
        return times.copy()
        
    dist = np.zeros(n, dtype=float)
    total_succ = np.zeros(n, dtype=int)
    
    for u in reversed(top_order):
        max_succ_dist = 0.0
        count_succ = 0
        for v in succ[u]:
            if dist[v] > max_succ_dist:
                max_succ_dist = dist[v]
            count_succ += 1 + total_succ[v]
        dist[u] = times[u] + max_succ_dist
        total_succ[u] = count_succ
        
    # Initial order: 
    # Primary: dist descending (critical path)
    # Secondary: total_succ descending (bottleneck)
    # Tertiary: times descending (packing)
    # We create a tuple key for sorting
    initial_order = sorted(range(n), key=lambda x: (-dist[x], -total_succ[x], -times[x]))
    
    # Evaluate initial
    best_order = initial_order.copy()
    best_score = _simulate_greedy(best_order, times, succ, pred_counts_init, cycle_time)
    
    # Simulated Annealing
    current_order = best_order.copy()
    current_score = best_score
    T = 1.0
    cooling_rate = 0.99
    
    # Pre-allocate for speed? Not necessary for N=30.
    
    # Move types: 0 for swap, 1 for insertion
    # Insertion moves are often better for sequencing
    
    while time.time() < deadline:
        # Choose move type
        if random.random() < 0.5:
            # Swap
            new_order = current_order.copy()
            i, j = random.sample(range(n), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Insertion
            new_order = current_order.copy()
            i, j = random.sample(range(n), 2)
            if i == j: continue
            item = new_order.pop(i)
            new_order.insert(j, item)
            
        new_score = _simulate_greedy(new_order, times, succ, pred_counts_init, cycle_time)
        
        if new_score < current_score:
            current_order = new_order
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_order = current_order.copy()
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = new_score - current_score
            if delta > 0:
                # Avoid overflow
                if T < 1e-6:
                    prob = 0.0
                else:
                    prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_order = new_order
                    current_score = new_score
                
        T *= cooling_rate
        if T < 0.01:
            T = 0.01 # Floor temperature to allow some exploration
            
    # Convert best_order to scores
    # Higher score = higher priority
    # Rank 0 in order = highest priority = highest score
    scores = np.zeros(n)
    for idx, task in enumerate(best_order):
        scores[task] = n - idx
        
    return scores
# EVOLVE-BLOCK-END
