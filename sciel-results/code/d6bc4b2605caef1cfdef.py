"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
import math
from collections import defaultdict

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill

def _build_graph(instance):
    """
    Precompute graph properties:
    - n: number of tasks
    - times: array of processing times
    - preds: dict of task -> list of predecessors
    - succ: dict of task -> list of successors
    - in_deg: array of in-degrees
    - topo: topological order
    - rank_to_end: longest path from task to end (including task time)
    - rank_from_start: longest path from start to task (including task time)
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Ensure preds is a dict with all keys, even if empty
    if not isinstance(preds, dict):
        # Handle case where preds might be a list of lists or similar
        preds_dict = defaultdict(list)
        if preds:
             # If it's a list of lists, index is task
             for i in range(n):
                 if i < len(preds):
                     preds_dict[i] = list(preds[i])
        preds = preds_dict
    else:
        preds = dict(preds) # Copy to be safe

    # Build successor map
    succ = defaultdict(list)
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                succ[i].append(j)
                in_deg[j] += 1
    
    # Topological Sort (Kahn's Algorithm)
    queue = [i for i in range(n) if in_deg[i] == 0]
    topo = []
    in_deg_copy = in_deg.copy()
    while queue:
        # Pick any node, order doesn't matter for correctness of topo sort
        u = queue.pop()
        topo.append(u)
        for v in succ[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    
    # If topo length < n, there's a cycle, but task assumes valid DAG
    if len(topo) < n:
        # Fallback: append remaining
        for i in range(n):
            if i not in topo:
                topo.append(i)

    # Compute Rank to End (Longest path from u to any sink)
    # Process in reverse topological order
    rank_to_end = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ_rank = 0.0
        for v in succ[u]:
            if rank_to_end[v] > max_succ_rank:
                max_succ_rank = rank_to_end[v]
        rank_to_end[u] = times[u] + max_succ_rank

    # Compute Rank from Start (Longest path from any source to u)
    # Process in topological order
    rank_from_start = np.zeros(n, dtype=float)
    for u in topo:
        max_pred_rank = 0.0
        if u in preds:
            for v in preds[u]:
                if rank_from_start[v] > max_pred_rank:
                    max_pred_rank = rank_from_start[v]
        rank_from_start[u] = times[u] + max_pred_rank

    # Count direct and indirect successors
    succ_count = np.zeros(n, dtype=int)
    for i in range(n):
        visited = set()
        stack = list(succ[i])
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(succ[v])
        succ_count[i] = len(visited)

    return {
        'n': n,
        'times': times,
        'preds': preds,
        'succ': succ,
        'in_deg': in_deg,
        'topo': topo,
        'rank_to_end': rank_to_end,
        'rank_from_start': rank_from_start,
        'succ_count': succ_count
    }

def _simulate(instance, graph, priority):
    """
    Simulate the greedy assignment using the given priority vector.
    Returns the number of stations used.
    """
    n = graph['n']
    times = graph['times']
    preds = graph['preds']
    succ = graph['succ']
    in_deg = graph['in_deg']
    cycle_time = instance['cycle_time']
    
    # State
    assigned = np.zeros(n, dtype=bool)
    rem_preds = in_deg.copy()
    
    num_stations = 1
    current_station_time = 0.0
    
    # We need to efficiently find the best eligible task.
    # Since N=30, we can just scan all unassigned tasks.
    # To speed up, we can maintain a list of eligible tasks, but updating it
    # after each assignment requires checking successors.
    # Given N=30, O(N^2) or O(N^3) is fine.
    # Let's stick to a simple scan for clarity and correctness.
    
    while assigned.sum() < n:
        # Find eligible tasks: not assigned and rem_preds[i] == 0
        eligible = [i for i in range(n) if not assigned[i] and rem_preds[i] == 0]
        
        if not eligible:
            # Should not happen in valid DAG
            break
            
        # Find those that fit in current station
        fit = [i for i in eligible if times[i] <= cycle_time - current_station_time + 1e-9]
        
        if fit:
            # Pick highest priority among fit
            best = max(fit, key=lambda i: priority[i])
        else:
            # Close current station, open new one
            num_stations += 1
            current_station_time = 0.0
            # Pick highest priority among eligible
            best = max(eligible, key=lambda i: priority[i])
            
        # Assign best
        assigned[best] = True
        current_station_time += times[best]
        
        # Update successors
        for s in succ[best]:
            rem_preds[s] -= 1
            
    return num_stations

def _generate_initial_priority(graph):
    """
    Generate a strong initial priority vector.
    Combines Rank to End, LPT, and Successor Count.
    """
    n = graph['n']
    times = graph['times']
    rank_to_end = graph['rank_to_end']
    succ_count = graph['succ_count']
    
    # Normalize components
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    p_time = norm(times)
    p_rank = norm(rank_to_end)
    p_succ = norm(succ_count.astype(float))
    
    # Weighted sum. Rank is usually the strongest signal for ALB.
    # We want higher priority for tasks that should go earlier.
    # In the simulation, higher priority means picked first.
    # Tasks with high rank_to_end should be picked early to allow successors to fit.
    # So higher score = higher priority.
    
    priority = 0.5 * p_rank + 0.3 * p_time + 0.2 * p_succ
    return priority

def priority(instance):
    """
    Score each assembly task. 
    Uses Simulated Annealing to optimize the priority vector for the greedy evaluator.
    """
    start_time = time.time()
    graph = _build_graph(instance)
    n = graph['n']
    
    # Initial Priority
    current_priority = _generate_initial_priority(graph)
    current_score = _simulate(instance, graph, current_priority)
    
    best_priority = current_priority.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    # Since the objective is integer (number of stations), we use a step-based cooling
    # or time-based. Let's use time-based.
    initial_temp = 1.0
    final_temp = 0.01
    # We will run until deadline
    
    rng = np.random.RandomState(42)
    random.seed(42)
    
    # To avoid getting stuck, we also keep a "random restart" capability
    # If no improvement for a long time, restart with a new initial heuristic
    
    last_improvement_time = time.time()
    
    while time.time() < _DEADLINE - 5:
        # Generate a neighbor
        # Strategy: Perturb current priority
        # 1. Swap two random elements
        # 2. Add small noise
        # 3. Occasionally, re-weight the components of the initial heuristic
        
        r = rng.rand()
        new_priority = current_priority.copy()
        
        if r < 0.7:
            # Swap
            i, j = rng.choice(n, 2, replace=False)
            new_priority[i], new_priority[j] = new_priority[j], new_priority[i]
        elif r < 0.9:
            # Noise
            noise_scale = 0.1 * (1.0 if n > 1 else 0.0)
            new_priority += rng.normal(0, noise_scale, n)
        else:
            # Re-initialize with slight randomization of weights
            times = graph['times']
            rank_to_end = graph['rank_to_end']
            succ_count = graph['succ_count']
            
            def norm(arr):
                mn, mx = arr.min(), arr.max()
                if mx - mn < 1e-9:
                    return np.zeros_like(arr)
                return (arr - mn) / (mx - mn)
            
            w1 = rng.uniform(0.3, 0.7)
            w2 = rng.uniform(0.1, 1-w1)
            w3 = 1.0 - w1 - w2
            
            new_priority = w1 * norm(rank_to_end) + w2 * norm(times) + w3 * norm(succ_count.astype(float))
            # Add some noise to break ties
            new_priority += rng.normal(0, 0.01, n)
            
        # Evaluate
        new_score = _simulate(instance, graph, new_priority)
        
        # Acceptance Criteria (Simulated Annealing)
        delta = new_score - current_score
        
        if delta < 0:
            # Improvement
            current_priority = new_priority
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_priority = new_priority.copy()
            last_improvement_time = time.time()
        else:
            # Worse or same
            # Calculate temperature based on elapsed time
            elapsed = time.time() - start_time
            total_time = 140.0 # approx budget
            # Linear cooling
            temp = initial_temp - (initial_temp - final_temp) * (elapsed / total_time)
            if temp < final_temp:
                temp = final_temp
                
            if delta == 0:
                # Accept same score to move around the landscape
                if rng.rand() < 0.5:
                    current_priority = new_priority
            else:
                # Probabilistic acceptance
                # exp(-delta / temp)
                # delta is positive (worse), so exponent is negative
                # If temp is small, probability is low
                prob = math.exp(-delta / temp)
                if rng.rand() < prob:
                    current_priority = new_priority
                    current_score = new_score
        
        # Optional: Random Restart if stuck
        # If no improvement for 5 seconds, try a fresh start
        if time.time() - last_improvement_time > 5.0:
            # Reset to a new random initial
            current_priority = _generate_initial_priority(graph)
            current_priority += rng.normal(0, 0.1, n)
            current_score = _simulate(instance, graph, current_priority)
            last_improvement_time = time.time()

    return best_priority

# EVOLVE-BLOCK-END
