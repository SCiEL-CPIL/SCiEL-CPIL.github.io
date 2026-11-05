"""Combined program: Structure-initialized Simulated Annealing for ALB."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _simulate_greedy(order, times, succ, pred_counts_init, cycle_time):
    """
    Simulate the greedy packing process for a given priority order.
    Uses a sorted available list for faster lookups.
    """
    n = len(times)
    pred_counts = pred_counts_init.copy()
    
    # Initial available tasks
    available = [i for i in range(n) if pred_counts[i] == 0]
    
    # Rank array for quick lookup of priority
    rank = [0] * n
    for idx, task in enumerate(order):
        rank[task] = idx
        
    stations = 0
    current_load = 0.0
    assigned_count = 0
    
    while assigned_count < n:
        best_task = -1
        best_rank = n + 1
        remaining_space = cycle_time - current_load
        
        # Find best task in available
        for t in available:
            if times[t] <= remaining_space + 1e-9:
                if rank[t] < best_rank:
                    best_rank = rank[t]
                    best_task = t
                    
        if best_task != -1:
            assigned_count += 1
            current_load += times[best_task]
            
            # Remove from available
            available.remove(best_task)
            
            # Update successors
            for s in succ[best_task]:
                pred_counts[s] -= 1
                if pred_counts[s] == 0:
                    available.append(s)
        else:
            if current_load > 1e-9:
                stations += 1
                current_load = 0.0
            else:
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
        
    # Critical Path Density: ratio of task duration to longest downstream path
    # High density means the task is a bottleneck on its own critical path
    # Combine with immediate successor count for branching points
    cpd = np.zeros(n, dtype=float)
    for i in range(n):
        if dist[i] > 1e-9:
            cpd[i] = times[i] / dist[i]
        else:
            cpd[i] = 1.0  # Terminal tasks get max density
    
    # Immediate successor count
    imm_succ_count = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Initial order: 
    # Primary: dist descending (critical path)
    # Secondary: cpd descending (critical path density - bottleneck)
    # Tertiary: total_succ descending (bottleneck)
    # Quaternary: imm_succ_count descending (branching)
    # Quinary: times descending (packing)
    initial_order = sorted(range(n), key=lambda x: (-dist[x], -cpd[x], -total_succ[x], -imm_succ_count[x], -times[x]))
    
    # Evaluate initial
    best_order = initial_order.copy()
    best_score = _simulate_greedy(best_order, times, succ, pred_counts_init, cycle_time)
    
    # Simulated Annealing
    current_order = best_order.copy()
    current_score = best_score
    T = 2.0
    cooling_rate = 0.995
    min_T = 0.05
    
    while time.time() < deadline:
        # Choose move type
        if random.random() < 0.4:
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
            delta = new_score - current_score
            if delta > 0:
                if T < 1e-6:
                    prob = 0.0
                else:
                    prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_order = new_order
                    current_score = new_score
                
        T *= cooling_rate
        if T < min_T:
            T = min_T
    
    # Local polish: try all single-position insertions from best_order
    for _ in range(3):  # Multiple passes
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                new_order = best_order.copy()
                item = new_order.pop(i)
                new_order.insert(j, item)
                new_score = _simulate_greedy(new_order, times, succ, pred_counts_init, cycle_time)
                if new_score < best_score:
                    best_score = new_score
                    best_order = new_order
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Second polish: adjacent swaps for finer-grained improvements
    if time.time() < deadline:
        for _ in range(3):  # Multiple passes
            if time.time() > deadline:
                break
            improved = False
            for i in range(n - 1):
                new_order = best_order.copy()
                new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
                new_score = _simulate_greedy(new_order, times, succ, pred_counts_init, cycle_time)
                if new_score < best_score:
                    best_score = new_score
                    best_order = new_order
                    improved = True
                    break
            if not improved:
                break
    
    # Convert best_order to scores
    scores = np.zeros(n)
    for idx, task in enumerate(best_order):
        scores[task] = n - idx
        
    return scores
# EVOLVE-BLOCK-END
