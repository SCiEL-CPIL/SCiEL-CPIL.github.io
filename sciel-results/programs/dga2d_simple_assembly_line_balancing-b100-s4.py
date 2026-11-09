"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from collections import deque


def priority(instance):
    """
    Score each assembly task for the priority rule.
    
    Strategy:
    1. Compute several candidate priority score vectors using different heuristics.
    2. For each candidate, simulate the greedy station-filling algorithm.
    3. Return the priority scores that yield the fewest stations for this instance.
    
    Includes a deadline guard to ensure we finish well within the 150s limit.
    """
    start_time = _time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor lists and in-degree for topological sort
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    # Topological order (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    indeg_copy = in_degree[:]
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg_copy[v] -= 1
            if indeg_copy[v] == 0:
                queue.append(v)
    
    # If not all nodes in topo, there's a cycle; fall back to LPT
    if len(topo) != n:
        return times.copy()
    
    # Compute longest downstream path (rank position) in reverse topological order
    # rank[i] = times[i] + max(rank[j] for j in successors[i], default 0)
    rank = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ = 0.0
        for v in successors[u]:
            if rank[v] > max_succ:
                max_succ = rank[v]
        rank[u] = times[u] + max_succ
    
    # Longest path to start (longest path ending at node)
    lp_start = np.zeros(n, dtype=float)
    for node in topo:
        preds = predecessors[node] if node < len(predecessors) else []
        if preds:
            lp_start[node] = max(times[p] + lp_start[p] for p in preds)
        lp_start[node] += times[node]
    
    # Number of successors
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Generate candidate priority score vectors
    candidates = {}
    candidates["LTF"] = times.copy()
    candidates["LPE"] = rank.copy()
    candidates["LPS"] = lp_start.copy()
    candidates["NS"] = num_successors.copy()
    candidates["RANK"] = (num_successors + 1) * rank
    candidates["COMBO1"] = rank * (1 + 0.1 * num_successors)
    candidates["COMBO2"] = times * (1 + 0.5 * num_successors)
    candidates["COMBO3"] = rank + 0.1 * times
    candidates["COMBO4"] = lp_start + 0.1 * num_successors
    
    def simulate_stations(priority_scores):
        """Clean simulation of greedy station filling."""
        completed = [False] * n
        stations = 0
        current_load = 0.0
        
        while any(not c for c in completed):
            # Find all eligible tasks: all preds completed, fits in remaining cycle time
            eligible = []
            for task in range(n):
                if completed[task]:
                    continue
                preds = predecessors[task] if task < len(predecessors) else []
                all_done = all(completed[p] for p in preds)
                if all_done and times[task] <= cycle_time - current_load:
                    eligible.append(task)
            
            if not eligible:
                # Move to next station
                stations += 1
                current_load = 0.0
                # Re-evaluate eligible with fresh station
                for task in range(n):
                    if completed[task]:
                        continue
                    preds = predecessors[task] if task < len(predecessors) else []
                    all_done = all(completed[p] for p in preds)
                    if all_done and times[task] <= cycle_time:
                        eligible.append(task)
            
            if not eligible:
                # Infeasible or bug; break to avoid infinite loop
                stations += 1
                break
            
            # Pick highest priority
            best_task = max(eligible, key=lambda t: priority_scores[t])
            completed[best_task] = True
            current_load += times[best_task]
        
        return stations
    
    # Evaluate each candidate and pick the best
    best_priority = None
    best_stations = float('inf')
    
    for name, p in candidates.items():
        # Check deadline
        if _time.time() - start_time > DEADLINE:
            break
        
        stations = simulate_stations(p)
        if stations < best_stations:
            best_stations = stations
            best_priority = p
    
    if best_priority is None:
        best_priority = times.copy()
    
    return np.asarray(best_priority, dtype=float)


# EVOLVE-BLOCK-END
