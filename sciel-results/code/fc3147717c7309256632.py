# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_station_count(times, predecessors, cycle_time, priorities):
    """Simulate greedy assignment and return number of stations."""
    n = len(times)
    # Build successor list
    successors = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i]
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    # Track remaining predecessors for each task
    remaining_preds = pred_count[:]
    available = set(i for i in range(n) if remaining_preds[i] == 0)
    
    # Sort by priority (higher first)
    # We'll repeatedly pick the best available that fits
    
    station_count = 0
    remaining_time = 0
    assigned = set()
    
    while len(assigned) < n:
        if remaining_time == 0:
            station_count += 1
            remaining_time = cycle_time
        
        # Find best available task that fits
        best_task = -1
        best_priority = -np.inf
        for t in available:
            if t in assigned:
                continue
            if times[t] <= remaining_time:
                if priorities[t] > best_priority:
                    best_priority = priorities[t]
                    best_task = t
        
        if best_task == -1:
            # No task fits in current station, open new one
            station_count += 1
            remaining_time = cycle_time
            # Try again
            best_task = -1
            best_priority = -np.inf
            for t in available:
                if t in assigned:
                    continue
                if times[t] <= remaining_time:
                    if priorities[t] > best_priority:
                        best_priority = priorities[t]
                        best_task = t
            if best_task == -1:
                # Shouldn't happen if all tasks fit in cycle_time
                # Force assign the smallest task
                min_t = min(available - assigned, key=lambda x: times[x])
                best_task = min_t
        
        # Assign best_task
        assigned.add(best_task)
        remaining_time -= times[best_task]
        available.discard(best_task)
        
        # Update successors
        for s in successors[best_task]:
            remaining_preds[s] -= 1
            if remaining_preds[s] == 0:
                available.add(s)
    
    return station_count


def _compute_ranks(times, predecessors):
    """Compute rank for each task: longest path from task to end."""
    n = len(times)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Topological order (reverse: from end to start)
    # Compute out-degree
    out_degree = [len(successors[i]) for i in range(n)]
    # Use Kahn's algorithm on reversed graph
    in_degree = [0] * n
    for i in range(n):
        for s in successors[i]:
            in_degree[s] += 1  # This is wrong for reversed
    
    # Actually, let me compute longest path from each node to any sink
    # Using topological sort
    in_deg = [len(predecessors[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for s in successors[u]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    
    # Longest path from node to end: process in reverse topo order
    longest = [0.0] * n
    for u in reversed(topo_order):
        if successors[u]:
            longest[u] = times[u] + max(longest[s] for s in successors[u])
        else:
            longest[u] = times[u]
    
    return longest


def priority(instance):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Compute ranks
    ranks = _compute_ranks(times, predecessors)
    
    # Initial priority: combination of rank and time
    # Try a few different combinations
    base_priorities = ranks + times  # rank already includes own time
    
    # Evaluate initial
    best_score = _simulate_station_count(times, predecessors, cycle_time, base_priorities)
    best_priorities = base_priorities.copy()
    
    # Local search: try perturbing priorities
    # Strategy: for each pair of tasks, try swapping their relative order
    # Or: try adjusting priority of individual tasks
    
    # A better approach: try different weighting schemes
    # And then do local search on the priority vector
    
    # First, try several analytical priority schemes
    candidates = []
    
    # Scheme 1: rank (longest path to end)
    candidates.append(ranks.copy())
    
    # Scheme 2: time only (LPT)
    candidates.append(times.copy())
    
    # Scheme 3: rank + alpha * time for various alpha
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        candidates.append(ranks + alpha * times)
    
    # Scheme 4: number of successors (out-degree)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)
    candidates.append(ranks + out_degree)
    
    # Scheme 5: rank * (1 + out_degree)
    candidates.append(ranks * (1.0 + out_degree))
    
    # Evaluate all candidates
    for cand in candidates:
        score = _simulate_station_count(times, predecessors, cycle_time, cand)
        if score < best_score:
            best_score = score
            best_priorities = cand.copy()
    
    # Local search: perturb best_priorities
    # Try small random perturbations and keep improvements
    rng = np.random.default_rng(42)
    
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            # Perturb: pick a random task and adjust its priority
            i = rng.integers(0, n)
            perturbation = rng.normal(0, 1.0)
            new_priorities = best_priorities.copy()
            new_priorities[i] += perturbation
            
            score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
            if score < best_score:
                best_score = score
                best_priorities = new_priorities
                improved = True
    
    # Try pair swaps as well
    if time.time() < deadline - 5:
        improved = True
        while improved and (time.time() < deadline - 5):
            improved = False
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                i, j = rng.integers(0, n, size=2)
                if i == j:
                    continue
                new_priorities = best_priorities.copy()
                # Swap priorities
                new_priorities[i], new_priorities[j] = new_priorities[j], new_priorities[i]
                score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
                if score < best_score:
                    best_score = score
                    best_priorities = new_priorities
                    improved = True
    
    return best_priorities
# EVOLVE-BLOCK-END
