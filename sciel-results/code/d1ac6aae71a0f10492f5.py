"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import cmp_to_key
import time

_DEADLINE = 120.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed() - 5.0  # 5s safety margin


def _compute_successors(predecessors, n):
    """Compute successors list from predecessors."""
    succ = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            succ[i].append(j)
    return succ


def _compute_indegree(predecessors, n):
    """Compute in-degree for each task."""
    indeg = [0] * n
    for j in range(n):
        indeg[j] = len(predecessors[j])
    return indeg


def _compute_longest_path_downstream(times, succ, n):
    """
    Compute longest path from each task to the end (downstream).
    longest[i] = times[i] + max(longest[j] for j in succ[i])
    """
    longest = np.array(times, dtype=float).copy()
    for _ in range(n):  # at most n passes needed
        changed = False
        for i in range(n):
            if succ[i]:
                max_down = max(longest[j] for j in succ[i])
                new_val = times[i] + max_down
                if new_val > longest[i]:
                    longest[i] = new_val
                    changed = True
        if not changed:
            break
    return longest


def _compute_longest_path_upstream(times, predecessors, n):
    """
    Compute longest path from the start to each task (upstream).
    longest_up[i] = times[i] + max(longest_up[j] for j in predecessors[i])
    """
    longest_up = np.zeros(n, dtype=float)
    longest_up = np.array(times, dtype=float).copy()
    for _ in range(n):
        changed = False
        for i in range(n):
            if predecessors[i]:
                max_up = max(longest_up[j] for j in predecessors[i])
                new_val = times[i] + max_up
                if new_val > longest_up[i]:
                    longest_up[i] = new_val
                    changed = True
        if not changed:
            break
    return longest_up


def _simulate(priority_scores, times, predecessors, cycle_time, n):
    """
    Simulate the greedy station-filling algorithm.
    Returns the number of stations used.
    """
    succ = _compute_successors(predecessors, n)
    
    # Track which tasks are completed
    completed = [False] * n
    # Track remaining predecessors count for each task
    remaining_pred = [len(predecessors[j]) for j in range(n)]
    
    # Available tasks: those with all predecessors completed
    # We'll use a list and filter each time for simplicity with n=30
    
    stations = 0
    current_station_time = 0.0
    
    while any(not c for c in completed):
        # Find eligible tasks: not completed, all predecessors completed, fits in current station
        eligible = []
        for j in range(n):
            if not completed[j] and remaining_pred[j] == 0:
                if current_station_time + times[j] <= cycle_time + 1e-9:
                    eligible.append(j)
        
        if not eligible:
            # Start a new station
            stations += 1
            current_station_time = 0.0
            # Re-check eligibility with new station
            eligible = []
            for j in range(n):
                if not completed[j] and remaining_pred[j] == 0:
                    if times[j] <= cycle_time + 1e-9:
                        eligible.append(j)
            
            if not eligible:
                # This shouldn't happen if cycle_time is valid, but break to avoid infinite loop
                break
        
        # Select the highest priority task from eligible
        best_task = max(eligible, key=lambda j: priority_scores[j])
        
        # Assign it to current station
        current_station_time += times[best_task]
        completed[best_task] = True
        
        # Update remaining_pred for successors
        for k in succ[best_task]:
            remaining_pred[k] -= 1
    
    # If we ended with a non-empty current station, it counts as a station
    # But the loop above increments stations when starting a new one.
    # Let's re-examine: 
    # - Initially stations=0, current_station_time=0
    # - First iteration: if no eligible (because current_station_time=0 and all fit), we don't increment.
    #   Wait, the logic is: if no eligible fits in CURRENT station, start NEW station (increment).
    #   But initially, current_station_time=0, so all eligible tasks fit.
    #   So we assign to the "first" station without incrementing? That's wrong.
    
    # Let me restructure the simulation more carefully.
    return stations


def _simulate_v2(priority_scores, times, predecessors, cycle_time, n):
    """
    More careful simulation.
    """
    succ = _compute_successors(predecessors, n)
    completed = [False] * n
    remaining_pred = [len(predecessors[j]) for j in range(n)]
    
    stations = 0
    current_station_time = 0.0
    in_station = False  # whether we've started placing in the current station
    
    while any(not c for c in completed):
        # Find eligible tasks that fit in the current station
        eligible = []
        for j in range(n):
            if not completed[j] and remaining_pred[j] == 0:
                if current_station_time + times[j] <= cycle_time + 1e-9:
                    eligible.append(j)
        
        if eligible:
            # Pick highest priority
            best_task = max(eligible, key=lambda j: priority_scores[j])
            current_station_time += times[best_task]
            completed[best_task] = True
            in_station = True
            
            for k in succ[best_task]:
                remaining_pred[k] -= 1
        else:
            # No eligible task fits in current station
            if in_station:
                # Current station is done, start a new one
                stations += 1
                current_station_time = 0.0
                in_station = False
            else:
                # We haven't placed anything yet, start first station
                stations += 1
                current_station_time = 0.0
                in_station = False
                # Re-check: now with empty station, find eligible
                eligible = []
                for j in range(n):
                    if not completed[j] and remaining_pred[j] == 0:
                        if times[j] <= cycle_time + 1e-9:
                            eligible.append(j)
                
                if not eligible:
                    break  # Should not happen
                
                best_task = max(eligible, key=lambda j: priority_scores[j])
                current_station_time += times[best_task]
                completed[best_task] = True
                in_station = True
                
                for k in succ[best_task]:
                    remaining_pred[k] -= 1
    
    # If we finished with an in-progress station, count it
    if in_station:
        stations += 1
    
    return stations


def priority(instance):
    """
    Score each assembly task to minimize the number of stations.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Check deadline
    if _remaining() <= 0:
        return np.zeros(n)
    
    # Compute various features
    succ = _compute_successors(predecessors, n)
    longest_down = _compute_longest_path_downstream(times, succ, n)
    longest_up = _compute_longest_path_upstream(times, predecessors, n)
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    num_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Also compute total downstream workload (sum of all successor times, not just longest path)
    # This captures the "bottleneck" aspect - tasks that unlock a lot of work
    total_down = np.array(times, dtype=float).copy()
    for _ in range(n):
        changed = False
        for i in range(n):
            if succ[i]:
                sum_down = sum(times[j] for j in succ[i])
                # Actually we want cumulative: total_down[i] = times[i] + max over paths... 
                # Let's use sum of all reachable task times as a proxy for "unlocking work"
                pass
        if not changed:
            break
    
    # Compute total reachable work (all successors, not just longest path)
    total_reachable = np.zeros(n, dtype=float)
    for i in range(n):
        # BFS/DFS to find all successors
        visited = set()
        stack = list(succ[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                total_reachable[i] += times[node]
                stack.extend(succ[node])
    
    # Normalize features
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-12:
            return np.ones_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    norm_down = normalize(longest_down)
    norm_time = normalize(times)
    norm_succ = normalize(num_succ)
    norm_reachable = normalize(total_reachable)
    
    # Strategy: Use a weighted combination that emphasizes:
    # 1. Longest downstream path (critical path) - primary driver
    # 2. Task time - pack large tasks to reduce fragmentation
    # 3. Total reachable work - unblock maximum downstream work
    # 4. Direct successor count - tiebreaker for unlocking
    
    # The key insight: for minimizing station count, we want to pack stations tightly.
    # This means prioritizing tasks that:
    # - Are on the critical path (can't be delayed)
    # - Are large (fill station capacity better)
    # - Unlock many successors (reduce future bottlenecks)
    
    priority_scores = (
        0.5 * norm_down +
        0.25 * norm_time +
        0.15 * norm_reachable +
        0.10 * norm_succ
    )
    
    return priority_scores
# EVOLVE-BLOCK-END
