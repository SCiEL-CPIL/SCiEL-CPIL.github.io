"""Combined program: Structure-initialized Simulated Annealing for ALB with Critical Path Density."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _simulate_greedy(order, times, succ, pred_counts_init, cycle_time):
    """
    Simulate the greedy packing process for a given priority order.
    Optimized for speed with early termination checks.
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
                r = rank[t]
                if r < best_rank:
                    best_rank = r
                    best_task = t
                    if r == 0: # Can't get better than rank 0
                        break
                    
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
    (including Critical Path Density) to find the best static priority order.
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
        
    # Compute "Critical Path Density" metric
    # Ratio of task's duration to total length of its longest downstream path
    # This prioritizes tasks that are bottlenecks in the critical path
    # Also combine with immediate successors count (fanout)
    
    # dist[u] is the longest path from u to terminal (including u)
    # The "downstream" part is dist[u] - times[u]
    # But we want to prioritize tasks where times[u] is a large fraction of the path
    # Let's define density = times[u] / dist[u] (if dist[u] > 0)
    
    density = np.zeros(n, dtype=float)
    for i in range(n):
        if dist[i] > 1e-9:
            density[i] = times[i] / dist[i]
        else:
            density[i] = 0.0
            
    # Immediate successors count (fanout)
    fanout = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Normalize components for initial ordering
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
        
    norm_dist = normalize(dist)
    norm_density = normalize(density)
    norm_fanout = normalize(fanout)
    norm_total_succ = normalize(total_succ.astype(float))
    norm_times = normalize(times)
    
    # Initial order:
    # Primary: dist descending (critical path)
    # Secondary: density descending (bottleneck ratio)
    # Tertiary: fanout descending (branching)
    # Quaternary: total_succ descending
    # Quint: times descending
    
    # We create a tuple key for sorting
    initial_order = sorted(range(n), key=lambda x: (-dist[x], -density[x], -fanout[x], -total_succ[x], -times[x]))
    
    # Evaluate initial
    best_order = initial_order.copy()
    best_score = _simulate_greedy(best_order, times, succ, pred_counts_init, cycle_time)
    
    # Simulated Annealing
    current_order = best_order.copy()
    current_score = best_score
    T = 3.0
    cooling_rate = 0.999
    min_T = 0.1
    
    # Track stagnation to trigger restarts or intensification
    no_improve_count = 0
    max_no_improve = 2000 # Lower threshold for restart
    
    while time.time() < deadline:
        # Choose move type
        # Weight towards local moves (insertion) more often for fine-tuning
        if random.random() < 0.3:
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
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_order = current_order.copy()
        else:
            no_improve_count += 1
            delta = new_score - current_score
            if delta > 0:
                if T < 1e-6:
                    prob = 0.0
                else:
                    prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_order = new_order
                    current_score = new_score
            else:
                # Equal score, accept to move
                current_order = new_order
                current_score = new_score
                
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        # Restart mechanism: if stuck for a long time, reset to best and reheat
        if no_improve_count > max_no_improve:
            current_order = best_order.copy()
            current_score = best_score
            T = 1.0 # Reheat
            no_improve_count = 0
    
    # Local polish: try all single-position insertions from best_order
    # Use more passes if time permits
    # Increased passes and tighter time management
    while time.time() < deadline - 2:
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
    
    # Convert best_order to scores
    scores = np.zeros(n)
    for idx, task in enumerate(best_order):
        scores[task] = n - idx
        
    return scores
# EVOLVE-BLOCK-END
