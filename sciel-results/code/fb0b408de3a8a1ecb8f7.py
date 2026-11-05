"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _compute_heights(times, predecessors, n):
    """Compute longest path to sink (height) for each task."""
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    out_degree = [len(succs) for succs in successors]
    heights = np.zeros(n, dtype=float)
    
    queue = deque()
    for i in range(n):
        if out_degree[i] == 0:
            queue.append(i)
            heights[i] = times[i]
            
    processed = 0
    while queue:
        u = queue.popleft()
        processed += 1
        for p in predecessors[u]:
            candidate_height = times[p] + heights[u]
            if candidate_height > heights[p]:
                heights[p] = candidate_height
            out_degree[p] -= 1
            if out_degree[p] == 0:
                queue.append(p)
                
    if processed < n:
        for i in range(n):
            if heights[i] == 0 and times[i] > 0:
                heights[i] = times[i]
                
    return heights

def _simulate_balance(times, predecessors, cycle_time, priority_scores):
    """
    Simulates the greedy line balancing process.
    Returns the number of stations used.
    """
    n = len(times)
    # Sort tasks by priority (descending)
    # We need to handle ties stably, perhaps by index or just stable sort
    sorted_indices = np.argsort(-priority_scores, kind='stable')
    
    # State
    assigned = np.zeros(n, dtype=bool)
    # For each task, we need to know if all its predecessors are assigned.
    # We can maintain a count of unassigned predecessors for each task.
    unassigned_preds = np.zeros(n, dtype=int)
    for i in range(n):
        unassigned_preds[i] = len(predecessors[i])
        
    # Also, for efficiency, we might want to know which tasks are "eligible"
    # Eligible: unassigned AND unassigned_preds[i] == 0
    # And fits in current station.
    
    stations = 0
    current_load = 0.0
    
    # To make this fast, we can just iterate through the sorted list and pick the first valid one.
    # But "highest priority eligible task" implies we check the global priority list.
    # Since n=30, O(N^2) or even O(N^3) per simulation is fine.
    
    # Let's use a simple loop:
    # While there are unassigned tasks:
    #   Start new station if current station is empty or no eligible task fits
    #   Find the highest priority task among eligible ones that fits in current_load + time <= cycle_time
    #   If found, assign it, update state, continue station
    #   If not found, close station, start new one
    
    while np.any(~assigned):
        # Try to find a task for the current station
        found = False
        best_idx = -1
        best_prio = -np.inf
        
        for idx in sorted_indices:
            if assigned[idx]:
                continue
            if unassigned_preds[idx] > 0:
                continue
            if current_load + times[idx] <= cycle_time:
                # This task is eligible and fits
                if priority_scores[idx] > best_prio:
                    best_prio = priority_scores[idx]
                    best_idx = idx
                    # We can't break immediately because there might be a higher priority one later in the list?
                    # No, sorted_indices is sorted by priority. So the FIRST one that is eligible and fits is the best.
                    break
        
        if best_idx != -1:
            # Assign task
            assigned[best_idx] = True
            current_load += times[best_idx]
            
            # Update unassigned_preds for successors
            # We need successors list. Let's build it once outside? 
            # For now, inside simulation we can just check predecessors of all tasks? 
            # No, that's slow. Let's build successors in priority() and pass it? 
            # Or just compute it in _simulate_balance. Since n=30, it's cheap.
            pass # We need successors to update unassigned_preds.
            
            # Actually, let's refactor _simulate_balance to take successors too or compute it.
            # To keep interface simple, I'll compute successors inside or pass it.
            # Let's pass it as a global or compute inside. Computing inside is O(N+E) which is fine.
            
            # Wait, I need to update unassigned_preds for the successors of best_idx.
            # Let's compute successors for the whole graph once in priority() and pass it?
            # The interface is priority(instance). I can't change the signature of the call from evaluator.
            # But I can define helper functions.
            # Let's just compute successors inside _simulate_balance for simplicity, or better,
            # pre-compute it in priority() and use a closure or global? 
            # No, let's just pass it. But _simulate_balance is called internally.
            
            # Let's re-implement _simulate_balance to be self-contained or take successors.
            # I will modify _simulate_balance to accept successors.
            
            # For now, let's assume I have successors.
            # I will refactor the code structure below.
            
            found = True
        else:
            # No task fits in current station
            # If current station is empty (current_load == 0), it means no eligible task fits at all?
            # Or no eligible tasks exist?
            # If no eligible tasks exist, but unassigned tasks remain, there's a deadlock (cycle) or error.
            # Assuming valid DAG, if no eligible task fits, we must have current_load > 0 or no eligible tasks.
            # If no eligible tasks, we can't proceed. But in a valid ALB, there is always at least one eligible task if unassigned remain.
            # So if best_idx == -1, it means no eligible task FITS.
            # So we must start a new station.
            
            # But wait, if current_load == 0, and no task fits, that means all eligible tasks have time > cycle_time.
            # This is an infeasible instance for this cycle time. 
            # The problem statement implies feasible instances.
            # If infeasible, we might infinite loop. 
            # Let's add a check: if current_load == 0 and no task fits, we might need to handle error or just return large number.
            # But assuming feasibility, current_load > 0 when we fail to place a task.
            
            stations += 1
            current_load = 0.0
            # We loop again. The same task that didn't fit might still not fit.
            # Actually, if a task didn't fit in a full station, it might fit in an empty one.
            # So restarting the station is correct.
            
            # Optimization: If we just started a new station (current_load=0), and the same task still doesn't fit, 
            # then that task's time > cycle_time. This is infeasible.
            # We'll assume feasibility.
            
            # To avoid infinite loop in case of infeasibility:
            if current_load == 0 and not found:
                 # Check if any unassigned task has time > cycle_time
                 # If so, return a very large number or raise error.
                 # For now, break and return large number.
                 return 1000

    # Don't forget to count the last station if it has tasks
    if current_load > 0:
        stations += 1
        
    return stations

def priority(instance):
    """
    Adaptive local search for priority scores.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Initial priority: Heights
    init_heights = _compute_heights(times, predecessors, n)
    current_scores = init_heights.copy()
    
    # Refactored simulate function that takes successors
    def simulate(scores):
        assigned = np.zeros(n, dtype=bool)
        unassigned_preds = np.array([len(preds) for preds in predecessors], dtype=int)
        stations = 0
        current_load = 0.0
        
        # Pre-sort indices by score? 
        # Scores change, so we must re-sort or find max each time.
        # Finding max each time is O(N). Total O(N^2).
        
        while np.any(~assigned):
            best_idx = -1
            best_score = -np.inf
            
            for i in range(n):
                if assigned[i]:
                    continue
                if unassigned_preds[i] > 0:
                    continue
                if current_load + times[i] <= cycle_time:
                    if scores[i] > best_score:
                        best_score = scores[i]
                        best_idx = i
                        
            if best_idx != -1:
                assigned[best_idx] = True
                current_load += times[best_idx]
                
                # Update unassigned_preds for successors
                for s in successors[best_idx]:
                    unassigned_preds[s] -= 1
            else:
                # No task fits
                if current_load > 0:
                    stations += 1
                    current_load = 0.0
                else:
                    # Infeasible or deadlock
                    return 1000
                    
        if current_load > 0:
            stations += 1
            
        return stations

    best_stations = simulate(current_scores)
    best_scores = current_scores.copy()
    
    start_time = time.time()
    time_limit = 140.0 # Safety margin for 150s limit
    
    # Local Search: Swap-based mutation
    # We will try to improve the ordering by swapping priorities of pairs of tasks.
    # Since the greedy algorithm picks the "highest priority eligible", the relative order of priorities matters.
    # We can view the scores as defining a permutation (with ties broken by index).
    # Swapping scores of two tasks i and j effectively swaps their priority levels.
    
    # To make it more effective, we can do a steepest descent:
    # Try all pairs (i, j), swap scores, simulate, if better, keep swap.
    # With n=30, there are 435 pairs.
    # Each simulation is ~O(N^2) = 900 ops.
    # 435 * 900 = ~400k ops per full pass.
    # This is very fast. We can do many passes.
    
    # However, simple swaps might get stuck in local minima.
    # We can also try "insertion" moves or random restarts.
    
    # Let's start with simple steepest descent swaps.
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # To avoid bias, we can randomize the order of pairs in each pass
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for i, j in pairs:
            if (time.time() - start_time) > time_limit:
                break
                
            # Swap scores
            temp = current_scores[i]
            current_scores[i] = current_scores[j]
            current_scores[j] = temp
            
            new_stations = simulate(current_scores)
            
            if new_stations < best_stations:
                best_stations = new_stations
                best_scores = current_scores.copy()
                improved = True
            else:
                # Revert swap if not improved
                current_scores[i] = current_scores[j]
                current_scores[j] = temp
            
        # If no improvement in a full pass, we might want to do a random restart or perturbation
        if not improved:
            # Try a random perturbation: pick 2-5 random pairs and swap them
            # This helps escape local minima
            num_swaps = random.randint(2, 5)
            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                temp = current_scores[i]
                current_scores[i] = current_scores[j]
                current_scores[j] = temp
                
            # Try to improve from this perturbed state
            improved = True # Force another pass
            # We don't update best_scores here, we just continue searching from this new state
            # But we need to ensure we don't lose the global best.
            # The loop above updates best_scores if a swap improves it.
            # The perturbation itself might be worse, but it opens up new search space.
            
            # Check time again
            if (time.time() - start_time) > time_limit:
                break

    return best_scores
# EVOLVE-BLOCK-END
