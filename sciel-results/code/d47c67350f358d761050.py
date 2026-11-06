"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def priority(instance):
    """
    Scores tasks using a simulation-guided penalty heuristic.
    
    Strategy:
    1. Parse instance data.
    2. Compute static metrics: Rank from Start (RS), Rank to End (RE/CPL).
    3. Perform a greedy simulation of the station filling process using a 
       baseline priority (e.g., RE/CPL).
    4. During simulation, track for each task:
       - How many times it was the cause of a new station opening (breaks).
       - The "waste" left in the previous station when this task triggered a break.
    5. Final Score = Base_Priority (RE) - Penalty * (Breaks + Waste).
       This encourages the evaluator to pick tasks that fit better into existing 
       stations, reducing the total number of stations.
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    if n == 0:
        return np.array([])
        
    predecessors = instance["predecessors"]
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Build successor list and in-degree count
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        if isinstance(predecessors, list):
            preds = predecessors[i]
        else:
            preds = list(predecessors[i])
            
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological Sort to compute ranks
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topological_order = []
    temp_in_degree = in_degree[:]
    
    while queue:
        u = queue.popleft()
        topological_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
                
    if len(topological_order) < n:
        # Cycle fallback
        topological_order = list(range(n))

    # Calculate Rank from Start (RS)
    rs = [0.0] * n
    for u in topological_order:
        max_pred_rs = 0.0
        for p in predecessors[u]:
            max_pred_rs = max(max_pred_rs, rs[p])
        rs[u] = times[u] + max_pred_rs

    # Calculate Rank to End (RE)
    re = [0.0] * n
    for u in reversed(topological_order):
        max_succ_re = 0.0
        for s in successors[u]:
            max_succ_re = max(max_succ_re, re[s])
        re[u] = times[u] + max_succ_re

    # Baseline Priority: RE (Critical Path Length)
    # We use this to run the simulation.
    # Higher RE means more critical, should be placed earlier.
    
    # Simulation of Greedy Packing
    # State:
    # - completed: set of task indices
    # - eligible: list of tasks whose predecessors are all completed
    # - current_station_time: time used in current station
    
    completed = set()
    eligible = [i for i in range(n) if in_degree[i] == 0]
    
    # Metrics to track
    break_count = np.zeros(n)
    waste_sum = np.zeros(n)
    placement_count = np.zeros(n)
    
    current_station_time = 0.0
    station_id = 0
    
    # We need a way to pick the "best" eligible task in the simulation.
    # The evaluator will use our final scores, but here we simulate a "reasonable" 
    # greedy choice to gather data. We'll use RE as the tie-breaker/priority for sim.
    
    while len(completed) < n:
        # If no eligible tasks but not all completed, something is wrong (cycle?)
        if not eligible:
            break
            
        # Find the eligible task with highest RE
        # For efficiency in simulation, we can just sort or max
        # Since n=30, sorting is cheap.
        eligible_sorted = sorted(eligible, key=lambda x: -re[x])
        
        # Try to place the first task that fits
        placed_task = -1
        for task in eligible_sorted:
            if current_station_time + times[task] <= cycle_time:
                placed_task = task
                break
        
        if placed_task != -1:
            # Place the task
            current_station_time += times[placed_task]
            completed.add(placed_task)
            placement_count[placed_task] += 1
            
            # Update eligible list
            eligible.remove(placed_task)
            for s in successors[placed_task]:
                # Check if all predecessors of s are completed
                all_preds_done = True
                for p in predecessors[s]:
                    if p not in completed:
                        all_preds_done = False
                        break
                if all_preds_done:
                    eligible.append(s)
        else:
            # No task fits in current station.
            # This means the current station is "closed" or we start a new one?
            # In ALB, if no eligible task fits, we start a new station with the 
            # highest priority eligible task (even if it exceeds? No, must fit).
            # If NO task fits, it implies the highest priority task is too big?
            # Or we just start a new station with the first eligible task.
            # Standard ALB: If no eligible task fits, start new station and place 
            # the highest priority eligible task (which must fit, otherwise infeasible).
            
            # The "break" is caused by the fact that the previous station couldn't 
            # accommodate any of the eligible tasks. 
            # Who is to blame? The task that was placed last in the previous station 
            # left too little space, OR the current eligible tasks are just too big.
            
            # Let's attribute the "waste" to the last task placed in the previous station.
            # We need to track last_task_placed.
            
            # Actually, a simpler metric: 
            # When we open a new station, the "waste" is (cycle_time - current_station_time).
            # We penalize the task that was placed LAST in the previous station 
            # for leaving this waste, IF that waste is significant.
            
            # However, we don't track last_task explicitly in the loop above.
            # Let's modify the loop to track last_placed.
            
            # Re-doing the loop logic to track last_placed for better attribution.
            # This is a simplification: We will attribute the waste of the closed station
            # to the task that was placed last in it.
            
            # To avoid complex state management in this quick sim, let's just 
            # count "breaks" and attribute waste to the task that *failed* to fit 
            # in the previous station? No, that's not right.
            
            # Let's use a different attribution:
            # When a station closes, the waste is W.
            # We penalize all tasks in that station proportionally to their time?
            # Or just the last one?
            
            # For now, let's just count how many times a station was closed 
            # "prematurely" (i.e., not because we ran out of tasks, but because 
            # no eligible task fit).
            
            # We'll increment a global "station_breaks" and later distribute penalty.
            # But we need per-task scores.
            
            # Let's restart the simulation logic with better tracking.
            pass # Break out to rewrite simulation properly

    # REWRITE SIMULATION WITH PROPER TRACKING
    
    completed = set()
    eligible = [i for i in range(n) if in_degree[i] == 0]
    current_station_time = 0.0
    last_placed_in_station = None
    station_tasks = []
    
    break_count = np.zeros(n)
    waste_sum = np.zeros(n)
    
    while len(completed) < n:
        if not eligible:
            break
            
        eligible_sorted = sorted(eligible, key=lambda x: -re[x])
        
        placed_task = -1
        for task in eligible_sorted:
            if current_station_time + times[task] <= cycle_time:
                placed_task = task
                break
        
        if placed_task != -1:
            current_station_time += times[placed_task]
            completed.add(placed_task)
            eligible.remove(placed_task)
            last_placed_in_station = placed_task
            station_tasks.append(placed_task)
            
            for s in successors[placed_task]:
                all_preds_done = True
                for p in predecessors[s]:
                    if p not in completed:
                        all_preds_done = False
                        break
                if all_preds_done:
                    eligible.append(s)
        else:
            # No task fits. Close current station.
            # Waste is cycle_time - current_station_time
            waste = cycle_time - current_station_time
            
            # Attribute waste to the last task placed? Or all tasks?
            # Let's attribute to the last task placed, as it had the final say on space.
            if last_placed_in_station is not None:
                waste_sum[last_placed_in_station] += waste
                break_count[last_placed_in_station] += 1
            
            # Start new station
            current_station_time = 0.0
            last_placed_in_station = None
            station_tasks = []
            
            # Place the highest priority eligible task in the new station
            # It MUST fit, otherwise the problem is infeasible.
            # But what if it doesn't fit? Then we have a problem.
            # Assume feasible.
            if eligible:
                task = eligible_sorted[0]
                current_station_time += times[task]
                completed.add(task)
                eligible.remove(task)
                last_placed_in_station = task
                station_tasks.append(task)
                
                for s in successors[task]:
                    all_preds_done = True
                    for p in predecessors[s]:
                        if p not in completed:
                            all_preds_done = False
                            break
                    if all_preds_done:
                        eligible.append(s)
            
            if time.time() - start_time > 110:
                # Timeout guard
                break

    # Calculate Final Scores
    # Base score: RE (Critical Path)
    # Penalty: waste_sum + alpha * break_count
    # We want to REDUCE the score of tasks that caused breaks/waste.
    # So: Score = RE - (waste_sum + 0.1 * break_count)
    
    scores = np.array(re)
    
    # Normalize waste to be comparable to RE?
    # RE is roughly sum of times. Waste is sum of (CT - used).
    # Let's scale penalty.
    penalty = waste_sum + 0.5 * break_count
    
    scores = scores - penalty
    
    # Ensure finite
    if not np.all(np.isfinite(scores)):
        scores = np.array(re)
        
    return scores

# EVOLVE-BLOCK-END
