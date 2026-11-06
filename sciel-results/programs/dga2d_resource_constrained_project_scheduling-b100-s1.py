import numpy as np
import time
import random
import sys

def _simulate_sgs(instance, priorities):
    """
    Simulates the Serial Schedule Generation Scheme (SGS).
    Returns the makespan.
    """
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    capacities = np.array(instance["capacities"], dtype=float)
    R = len(capacities)
    
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors for potential optimization if needed, but SGS only needs preds
    # Actually, SGS needs to know when an activity becomes eligible.
    # Eligible if all predecessors are finished.
    
    # Pre-compute pred counts
    pred_counts = [len(p) for p in preds]
    
    # State
    # remaining_demand[r] is current available capacity for resource r
    remaining_cap = capacities.copy()
    
    # finished[i] is bool
    finished = [False] * N
    # start_time[i]
    start_time = [0.0] * N
    # end_time[i]
    end_time = [0.0] * N
    
    # List of activities that are currently running
    # We need to track when they finish to update resources
    # Since N is small (30), we can just keep a list of (end_time, activity_idx)
    
    # Initial eligible set
    eligible = [i for i in range(N) if pred_counts[i] == 0]
    
    # Sort eligible by priority (descending)
    # We need a stable sort or handle ties.
    
    # To speed up SGS, we can use a heap or just sort the eligible list each step.
    # N=30 is very small. Sorting 30 items is trivial.
    
    # Current time
    current_time = 0.0
    # Makespan
    makespan = 0.0
    
    # Active activities: list of [end_time, activity_index]
    active = []
    
    # We will process in "events".
    # An event is either:
    # 1. An activity finishes (time advances to its end time).
    # 2. We need to start an eligible activity (time stays same or advances if no active).
    
    # Actually, standard SGS logic:
    # 1. Find earliest time t where at least one eligible activity can start.
    # 2. Among eligible activities that can start at t, pick highest priority.
    # 3. Start it.
    # 4. Repeat until all started.
    
    # However, a simpler loop for small N:
    # 1. Determine next event time (min end time of active activities).
    # 2. If no active activities, next event time is current_time.
    # 3. Advance time to next event time.
    # 4. Finish any activities that end at this time (release resources, mark finished, update successors).
    # 5. Check if any new activities became eligible.
    # 6. If there are eligible activities, check if they can fit in resources.
    #    - If yes, start the highest priority one.
    #    - If no, we might need to wait for more resources.
    #      But in SGS, we only start when resources are available.
    #      So if eligible but not enough resources, we do NOT start. We wait for next event.
    
    # Wait, standard SGS:
    # "The serial SGS generates a schedule by sequentially scheduling activities.
    # At each step, the eligible activity with the highest priority is scheduled
    # at its earliest possible start time."
    
    # So the algorithm is:
    # 1. Initialize eligible set.
    # 2. While eligible set is not empty:
    #    a. Pick highest priority activity A from eligible.
    #    b. Find earliest start time t_A such that:
    #       - t_A >= max(end_time of predecessors of A)
    #       - For all resources r, sum(demands[A][r] + sum(demands[B][r] for B active at t)) <= capacity[r]
    #         Where "active at t" means activities started before t and ending after t.
    #    c. Schedule A at t_A.
    #    d. Update state (start/end times, resources).
    #    e. Remove A from eligible.
    #    f. Add successors of A to eligible if all their preds are done.
    
    # This is O(N^2 * R) roughly, which is fine for N=30.
    
    # Let's implement this strictly.
    
    # State
    # start[i], end[i]
    # active_intervals: list of (start, end, idx)
    # We can just keep a list of scheduled activities and check overlaps.
    
    scheduled = [False] * N
    start = [0.0] * N
    end = [0.0] * N
    
    # For efficiency, keep a list of (start, end, idx) for scheduled activities
    scheduled_list = []
    
    # Eligible set as a list, we will sort it each time or use a max-heap.
    # Since N is small, simple list is fine.
    eligible = [i for i in range(N) if len(preds[i]) == 0]
    
    # Count of unsatisfied preds for each node
    unsat_preds = [len(p) for p in preds]
    
    # Successors map
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
            
    while eligible:
        # Sort eligible by priority descending
        eligible.sort(key=lambda x: priorities[x], reverse=True)
        
        # Try to schedule the first one (highest priority)
        # If it can't be scheduled due to resources, we might need to try the next?
        # No, SGS picks the highest priority eligible activity and schedules it at its EARLIEST possible time.
        # The "earliest possible time" accounts for resource contention.
        # So we don't skip activities in the eligible list. We just find the start time for the top one.
        
        act = eligible.pop(0)
        
        # Find earliest start time for act
        # Constraint 1: Predecessors
        earliest_start = 0.0
        for p in preds[act]:
            if end[p] > earliest_start:
                earliest_start = end[p]
        
        # Constraint 2: Resources
        # We need to find t >= earliest_start such that resources are available.
        # Since resources are piecewise constant, the "available" periods change only at start/end times of other activities.
        # Candidate start times:
        # 1. earliest_start
        # 2. end times of any scheduled activity that overlaps with [earliest_start, ...]
        
        # Let's collect candidate times
        candidates = [earliest_start]
        for s, e, idx in scheduled_list:
            if e > earliest_start:
                # This activity is active at some point after earliest_start
                # If it ends after earliest_start, its end time is a candidate for resource release
                candidates.append(e)
        
        # Also, if there are activities starting after earliest_start but ending before we check?
        # No, we only care about activities that are active *during* the execution of 'act'.
        # If an activity starts at t_start and ends at t_end, and t_start > earliest_start,
        # it might conflict if t_end > t_start and t_start < t_start + duration[act].
        
        # Actually, the set of times where resource availability changes is the set of start and end times of scheduled activities.
        # We should consider all end times of scheduled activities that are > earliest_start.
        # And potentially start times? No, start times don't release resources.
        
        # So candidates are {earliest_start} U {end[j] | end[j] > earliest_start}
        
        # Sort candidates
        candidates = sorted(list(set(candidates)))
        
        scheduled_act = False
        for t in candidates:
            # Check if resources are available for act starting at t
            # Duration d = durations[act]
            d = durations[act]
            end_t = t + d
            
            # Check resource constraints
            # For each resource r, sum of demands of activities overlapping [t, end_t] + demand[act][r] <= capacity[r]
            
            feasible = True
            # Optimization: We can just iterate through scheduled activities
            # N is small, so O(N*R) per candidate is fine.
            for s, e, idx in scheduled_list:
                # Check overlap
                if s < end_t and e > t:
                    # Overlap exists
                    for r in range(R):
                        if demands[idx][r] + demands[act][r] > capacities[r]:
                            # Wait, this is wrong. Multiple activities can overlap.
                            # We need to sum demands of ALL overlapping activities.
                            pass
                            
            # Correct check:
            # Sum demands of all activities overlapping [t, end_t]
            # Plus demand of 'act'
            # Must be <= capacity
            
            # Let's re-do the check properly
            resource_usage = np.zeros(R)
            for s, e, idx in scheduled_list:
                if s < end_t and e > t:
                    resource_usage += demands[idx]
            
            if np.all(resource_usage + demands[act] <= capacities):
                # Feasible
                start[act] = t
                end[act] = end_t
                scheduled[act] = True
                scheduled_list.append((t, end_t, act))
                scheduled_act = True
                
                # Update successors
                for v in succs[act]:
                    unsat_preds[v] -= 1
                    if unsat_preds[v] == 0:
                        eligible.append(v)
                        
                break
        
        if not scheduled_act:
            # This should not happen in a valid instance where capacities are sufficient for single activities
            # But if it does, we might have an issue.
            # For robustness, if no candidate works, try a very late time?
            # Or raise error.
            # Given "valid instances", this shouldn't happen.
            # But let's add a fallback: try time = max(end) + 1
            max_end = max([e for s, e, _ in scheduled_list] + [0])
            t = max_end
            start[act] = t
            end[act] = t + durations[act]
            scheduled[act] = True
            scheduled_list.append((t, t + durations[act], act))
            
            for v in succs[act]:
                unsat_preds[v] -= 1
                if unsat_preds[v] == 0:
                    eligible.append(v)

    makespan = max(end) if end else 0
    return makespan

def priority(instance):
    """
    Uses Simulated Annealing to optimize the priority vector.
    """
    N = len(instance["durations"])
    
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0 # Keep well under 120s
    
    # Initial priority: Random or based on CPL?
    # Let's start with a reasonable heuristic to give SA a good start.
    # Compute CPL
    durations = np.array(instance["durations"], dtype=float)
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]
         
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
            
    # Topological sort for CPL
    in_degree = [len(p) for p in preds]
    q = [i for i in range(N) if in_degree[i] == 0]
    topo = []
    indeg_temp = in_degree[:]
    while q:
        u = q.pop(0)
        topo.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)
                
    cpl = np.zeros(N)
    for i in range(len(topo) - 1, -1, -1):
        u = topo[i]
        max_succ = 0
        for v in succs[u]:
            if cpl[v] > max_succ:
                max_succ = cpl[v]
        cpl[u] = durations[u] + max_succ
        
    # Initial priority = CPL
    current_prio = cpl.copy()
    current_cost = _simulate_sgs(instance, current_prio)
    
    best_prio = current_prio.copy()
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.995
    min_T = 0.01
    
    # We need to define a neighborhood.
    # Perturbation: Swap two priority values? Or add noise?
    # Swapping indices in the priority list changes the order.
    # Since SGS picks highest priority, the relative order matters.
    # Let's use "Swap" mutation: pick two random activities, swap their priority values.
    # Or "Insert": pick an activity, move it to a random position in the priority ranking.
    
    # For speed, we will perform a fixed number of iterations or until time runs out.
    # N=30.
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Generate neighbor
        new_prio = current_prio.copy()
        
        # Mutation strategy: Randomly swap two elements
        i, j = random.sample(range(N), 2)
        new_prio[i], new_prio[j] = new_prio[j], new_prio[i]
        
        # Evaluate
        new_cost = _simulate_sgs(instance, new_prio)
        
        # Acceptance probability
        if new_cost < current_cost:
            current_prio = new_prio
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_prio = current_prio.copy()
        else:
            delta = new_cost - current_cost
            # Probability of accepting worse solution
            # exp(-delta / T)
            # delta is positive (worse)
            # If T is small, exp is small.
            p = np.exp(-delta / T)
            if random.random() < p:
                current_prio = new_prio
                current_cost = new_cost
        
        T *= cooling
        
    # Final check on time
    if time.time() - start_time > time_limit:
        return best_prio
        
    # Optional: Local search (Hill Climbing) on best_prio
    # Try swapping each pair once?
    # N=30, 435 pairs.
    # Each eval is ~1ms? 435ms is fine.
    
    improved = True
    while improved and (time.time() - start_time < time_limit - 5):
        improved = False
        for i in range(N):
            for j in range(i+1, N):
                if time.time() - start_time > time_limit - 5:
                    return best_prio
                new_prio = best_prio.copy()
                new_prio[i], new_prio[j] = new_prio[j], new_prio[i]
                new_cost = _simulate_sgs(instance, new_prio)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_prio = new_prio
                    improved = True
                    
    return best_prio
