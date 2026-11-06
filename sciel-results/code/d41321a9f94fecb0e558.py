"""Assembly Line Balancing - Model and Optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque

def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0
    
    successors = defaultdict(list)
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    eligible = set(i for i in range(n) if unassigned_preds[i] == 0)
    
    if n > 0 and not eligible:
        return 10**9
    
    station_count = 1
    current_load = 0
    unplaced_count = n
    
    while eligible:
        candidates = sorted(eligible, key=lambda i: -priority_scores[i])
        placed = False
        
        for task in candidates:
            if times[task] <= cycle_time - current_load:
                current_load += times[task]
                eligible.discard(task)
                unplaced_count -= 1
                placed = True
                
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.add(succ)
                break
        
        if not placed:
            station_count += 1
            current_load = 0
            if any(times[t] > cycle_time for t in eligible):
                return 10**9
    
    return station_count

def _compute_critical_path(times, predecessors, successors, n):
    """Compute longest path from each task to any sink."""
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    if len(topo_order) != n:
        return None
    
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            cp[u] = max(cp[u], times[u] + cp[v])
    
    return cp

def _simulate_annealing(priority, times, predecessors, cycle_time, n, deadline):
    """
    Simulated Annealing with diverse moves: swaps, adjacent swaps, and local perturbations.
    """
    current_priority = priority.copy()
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    best_priority = current_priority.copy()
    best_score = current_score
    
    # Adaptive temperature schedule
    T = 2.0
    cooling_rate = 0.92
    min_T = 0.005
    
    start_time = time.time()
    
    while T > min_T:
        # More swaps at higher temperatures, fewer at lower
        swaps_per_step = max(3, int(8 * T / 2.0))
        
        for _ in range(swaps_per_step):
            if time.time() > deadline:
                return best_priority, best_score
            
            # Choose move type based on temperature
            r = random.random()
            
            if r < 0.4:
                # Random swap
                i, j = random.sample(range(n), 2)
            elif r < 0.7:
                # Adjacent swap (in priority ranking)
                order = np.argsort(-current_priority)
                pos = random.randint(0, n - 2)
                i = order[pos]
                j = order[pos + 1]
            else:
                # Local perturbation: adjust one task's priority slightly
                i = random.randint(0, n - 1)
                # Store old value
                old_i = current_priority[i]
                # Add small random offset
                current_priority[i] = old_i + random.gauss(0, 0.5 * (np.max(current_priority) - np.min(current_priority) + 1e-6))
                # Evaluate
                new_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_priority = current_priority.copy()
                    if best_score == 1:
                        return best_priority, best_score
                else:
                    current_priority[i] = old_i
                continue
            
            # For swap moves
            old_i, old_j = current_priority[i], current_priority[j]
            current_priority[i], current_priority[j] = old_j, old_i
            
            new_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_priority = current_priority.copy()
                if best_score == 1:
                    return best_priority, best_score
            else:
                current_priority[i], current_priority[j] = old_i, old_j
        
        T *= cooling_rate
    
    # Final polish: more iterations at very low temperature
    polish_deadline = deadline
    polish_iters = 50
    for _ in range(polish_iters):
        if time.time() > polish_deadline:
            break
        i, j = random.sample(range(n), 2)
        old_i, old_j = current_priority[i], current_priority[j]
        current_priority[i], current_priority[j] = old_j, old_i
        new_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
        delta = new_score - current_score
        if delta < 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_priority = current_priority.copy()
        else:
            current_priority[i], current_priority[j] = old_i, old_j
    
    return best_priority, best_score

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute features
    cp = _compute_critical_path(times, predecessors, successors, n)
    if cp is None:
        cp = np.zeros(n, dtype=float)
    
    num_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    num_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Generate diverse initial heuristics
    heuristics = []
    
    # 1. Critical Path
    heuristics.append(cp.copy())
    
    # 2. Time + CP
    heuristics.append(times + cp)
    
    # 3. CP weighted by successors
    heuristics.append(cp + 0.5 * num_succ)
    
    # 4. Time weighted by CP
    heuristics.append(times * (cp / (np.max(cp) + 1e-6)))
    
    # 5. Pure Time
    heuristics.append(times.copy())
    
    # 6. CP - Preds
    heuristics.append(cp - 0.1 * num_pred)
    
    # 7. Combined: 0.5*CP + 0.3*Time + 0.2*Successors
    heuristics.append(0.5 * cp + 0.3 * times + 0.2 * num_succ)
    
    # 8. Rank-based: Sort by CP, assign rank
    rank_cp = np.argsort(np.argsort(cp))
    heuristics.append(rank_cp.astype(float))
    
    # 9. Bottleneck slack: time / (1 + successors) - favor capacity-filling tasks
    heuristics.append(times / (1.0 + num_succ))
    
    # 10. Downstream weight: sum of times in downstream subtree
    # Compute downstream time sum
    downstream_time = np.zeros(n, dtype=float)
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    for u in reversed(topo_order):
        downstream_time[u] = times[u]
        for v in successors[u]:
            downstream_time[u] += downstream_time[v]
    heuristics.append(downstream_time.copy())
    
    # Evaluate initial heuristics quickly
    best_priority = None
    best_score = float('inf')
    heuristic_scores = []
    
    for h in heuristics:
        if time.time() > deadline - 30:
            break
        score = _greedy_balance(times, predecessors, cycle_time, h)
        heuristic_scores.append(score)
        if score < best_score:
            best_score = score
            best_priority = h.copy()
        if best_score == 1:
            break
    
    if best_priority is None:
        best_priority = heuristics[0]
        best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)
    
    # Run Simulated Annealing from the best initial
    if time.time() < deadline - 15:
        best_priority, best_score = _simulate_annealing(
            best_priority, times, predecessors, cycle_time, n, deadline - 5
        )
        
        # If time remains, try SA from a few other good heuristics
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 15 and best_score > 1:
            # Sort heuristics by score
            sorted_indices = np.argsort(heuristic_scores)
            
            # Try the 2nd and 3rd best
            for idx in sorted_indices[1:3]:
                if time.time() > deadline - 5:
                    break
                if heuristic_scores[idx] >= best_score:
                    break
                pri, score = _simulate_annealing(
                    heuristics[idx], times, predecessors, cycle_time, n, deadline - 5
                )
                if score < best_score:
                    best_score = score
                    best_priority = pri
    
    return best_priority
# EVOLVE-BLOCK-END
