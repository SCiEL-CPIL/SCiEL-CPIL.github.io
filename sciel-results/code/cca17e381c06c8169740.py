import numpy as np
import time
import random

def priority(instance):
    start_time = time.time()
    deadline = 118.0  # Safety margin before 120s kill
    
    # 1. Extract and preprocess instance data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    
    if "capacities" in instance:
        capacities = np.asarray(instance["capacities"], dtype=float)
    else:
        # Fallback if capacities are missing or under a different key
        caps_key = "capacities" if "capacities" in instance else "capancies"
        if caps_key in instance:
            capacities = np.asarray(instance[caps_key], dtype=float)
        else:
            # Default to 1.0 if not found, assuming 1 resource if demands is 1D
            num_res = demands.shape[1] if demands.ndim > 1 else 1
            capacities = np.ones(num_res)

    num_resources = capacities.shape[0]
    
    # Precompute adjacency structures for fast SGS simulation
    # successors[i] is a list of indices
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Precompute in-degrees for topological sorting (used for initial guess)
    in_degree = [0] * n
    for i in range(n):
        in_degree[i] = len(predecessors[i])
        
    # 2. Initial Priority Vector: Critical Path Length (CPL)
    # Topological sort
    topo_order = []
    queue = [i for i in range(n) if in_degree[i] == 0]
    in_deg_temp = in_degree[:]
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for s in successors[node]:
            in_deg_temp[s] -= 1
            if in_deg_temp[s] == 0:
                queue.append(s)
                
    if len(topo_order) != n:
        # Cycle detected, return random or simple heuristic
        return np.random.rand(n)

    # Forward pass for ES/EF
    ES = np.zeros(n)
    EF = np.zeros(n)
    for i in topo_order:
        max_pred_ef = 0.0
        for p in predecessors[i]:
            if EF[p] > max_pred_ef:
                max_pred_ef = EF[p]
        ES[i] = max_pred_ef
        EF[i] = ES[i] + durations[i]
        
    project_duration_unres = np.max(EF)
    
    # Backward pass for LS/LF
    LF = np.full(n, project_duration_unres)
    LS = np.full(n, project_duration_unres)
    for i in reversed(topo_order):
        if successors[i]:
            min_succ_ls = float('inf')
            for s in successors[i]:
                if LS[s] < min_succ_ls:
                    min_succ_ls = LS[s]
            LF[i] = min_succ_ls
        else:
            LF[i] = project_duration_unres
        LS[i] = LF[i] - durations[i]
        
    # Initial priority: Higher urgency (smaller LF) -> Higher priority
    # We want the SGS to pick the one with highest score.
    # So score = -LF (or project_duration - LF)
    initial_priority = project_duration_unres - LF
    
    # 3. Fast SGS Simulation Function
    def simulate_sgs(priorities):
        """
        Simulates Serial SGS.
        Returns makespan.
        """
        # eligible: set of activities whose predecessors are all scheduled
        # scheduled: set of scheduled activities
        # resource_load[t][r]: load on resource r at time t
        
        # To make it fast, we don't use a full time grid if durations are large/continuous.
        # However, standard SGS with renewable resources usually implies discrete or continuous time.
        # Given the "30 activities" size, we can use an event-based or simple list-based check.
        # Let's use a simple loop that finds the earliest feasible start time for each candidate.
        
        scheduled = [False] * n
        start_times = [0.0] * n
        end_times = [0.0] * n
        current_time = 0.0
        
        # We need to track resource usage over time.
        # Since n is small (30), we can maintain a list of (end_time, resource_idx, amount) events or just check intervals.
        # A more efficient way for small n: maintain a list of scheduled intervals per resource.
        # resource_intervals[r] = list of (start, end)
        
        resource_intervals = [[] for _ in range(num_resources)]
        
        remaining = n
        while remaining > 0:
            # Find eligible activities
            eligible = []
            for i in range(n):
                if not scheduled[i]:
                    # Check predecessors
                    is_eligible = True
                    for p in predecessors[i]:
                        if not scheduled[p]:
                            is_eligible = False
                            break
                    if is_eligible:
                        eligible.append(i)
                        
            if not eligible:
                # Should not happen if DAG is valid and we schedule all
                break
                
            # Pick highest priority among eligible
            # Priorities are scores; higher is better
            best_idx = -1
            best_score = -float('inf')
            for i in eligible:
                if priorities[i] > best_score:
                    best_score = priorities[i]
                    best_idx = i
            
            if best_idx == -1:
                break
                
            # Determine earliest start time for best_idx
            # Constraint 1: Predecessors
            es_pred = 0.0
            for p in predecessors[best_idx]:
                if end_times[p] > es_pred:
                    es_pred = end_times[p]
                    
            # Constraint 2: Resources
            es_res = es_pred
            for r in range(num_resources):
                d = demands[best_idx][r]
                if d == 0:
                    continue
                # Check all existing intervals on resource r
                # We need to find the earliest time t >= es_pred such that
                # sum of overlapping demands <= capacity[r]
                
                # Simple approach: scan through sorted intervals
                # Since n is small, we can just check candidate start times.
                # Candidate start times are: es_pred, and end times of intervals that start after es_pred?
                # Actually, the earliest feasible time is either es_pred or the end time of some interval
                # that is blocking it.
                
                # Let's collect all relevant end times
                candidate_times = [es_pred]
                for (st, en) in resource_intervals[r]:
                    if en > es_pred:
                        candidate_times.append(en)
                    
                # Sort unique candidates
                candidate_times = sorted(list(set(candidate_times)))
                
                feasible_time = None
                for ct in candidate_times:
                    # Check if activity [ct, ct + duration] fits
                    # It fits if for all existing intervals (st, en) on r:
                    # overlap_demand + d <= capacity[r]
                    # Overlap exists if ct < en and st < ct + duration
                    
                    valid = True
                    dur = durations[best_idx]
                    for (st, en) in resource_intervals[r]:
                        if st < ct + dur and ct < en:
                            # There is overlap. We need to check total load.
                            # Wait, this simple check only checks against one interval at a time.
                            # We need to sum demands of ALL overlapping intervals.
                            pass
                    
                    # The above is inefficient and incorrect for multiple overlaps.
                    # Better: For a specific candidate start time ct, calculate total load on resource r
                    # during [ct, ct + dur].
                    
                    total_load = d
                    for (st, en) in resource_intervals[r]:
                        if st < ct + dur and ct < en:
                            # Overlap
                            total_load += demands[best_idx][r] # NO, this adds d again.
                            # We need the demand of the OTHER activity.
                            # But we don't know which activity created the interval!
                            # We must store (start, end, activity_id) or just demand.
                            # Let's store (start, end, demand)
                            pass
                            
            # REDESIGN: Store (start, end, activity_id) in resource_intervals
            # Then we can look up demand[activity_id][r]
            
            # Let's restart the simulation logic with proper data structures
            # To avoid rewriting the whole function, I will implement a robust version below.
            
            # Actually, let's just implement the simulation correctly from scratch inside the function.
            # The previous block was pseudocode.
            
            break # Exit this broken simulation logic
            
        # This function needs a complete rewrite to be correct.
        return 0.0 # Placeholder

    # Let's write a clean, correct simulation function.
    def simulate(priorities):
        scheduled = [False] * n
        start_times = [0.0] * n
        end_times = [0.0] * n
        
        # resource_intervals[r] is a list of tuples (start, end, activity_id)
        resource_intervals = [[] for _ in range(num_resources)]
        
        remaining = n
        while remaining > 0:
            # 1. Find eligible
            eligible = []
            for i in range(n):
                if not scheduled[i]:
                    ok = True
                    for p in predecessors[i]:
                        if not scheduled[p]:
                            ok = False
                            break
                    if ok:
                        eligible.append(i)
            
            if not eligible:
                break # Error: cycle or deadlock?
                
            # 2. Pick best priority
            best_i = -1
            best_score = -1e18
            for i in eligible:
                if priorities[i] > best_score:
                    best_score = priorities[i]
                    best_i = i
            
            if best_i == -1:
                break
                
            # 3. Find earliest start time
            es = 0.0
            for p in predecessors[best_i]:
                if end_times[p] > es:
                    es = end_times[p]
            
            dur = durations[best_i]
            
            # Check resources
            feasible_start = es
            # We need to find the earliest t >= es such that resource constraints are met.
            # Candidate times: es, and end times of any interval on any resource that starts after es.
            
            candidate_times = [es]
            for r in range(num_resources):
                if demands[best_i][r] > 0:
                    for (st, en, aid) in resource_intervals[r]:
                        if en > es:
                            candidate_times.append(en)
            
            candidate_times = sorted(list(set(candidate_times)))
            
            found = False
            for t in candidate_times:
                if t < es:
                    continue
                # Check if [t, t+dur] is feasible for all resources
                feasible = True
                for r in range(num_resources):
                    d = demands[best_i][r]
                    if d == 0:
                        continue
                    # Calculate load on resource r during [t, t+dur]
                    # Load = d + sum(demand[j][r] for j in intervals overlapping [t, t+dur])
                    current_load = d
                    for (st, en, aid) in resource_intervals[r]:
                        if st < t + dur and t < en:
                            current_load += demands[aid][r]
                    if current_load > capacities[r] + 1e-9:
                        feasible = False
                        break
                if feasible:
                    feasible_start = t
                    found = True
                    break
            
            if not found:
                # If no candidate time works, we might need to check further?
                # In SGS, if the earliest candidate fails, the next candidate is the next end time.
                # If all candidates fail, it means we can't schedule it? 
                # Actually, if we check all end times, one of them MUST work eventually 
                # unless the activity itself exceeds capacity (which is invalid instance)
                # or we missed a time.
                # For safety, if not found, use the last candidate (should be late enough)
                # But strictly, if the last candidate (max end time) fails, it's an error.
                # Let's assume valid instances.
                feasible_start = candidate_times[-1] if candidate_times else es
            
            # 4. Schedule it
            start_times[best_i] = feasible_start
            end_times[best_i] = feasible_start + dur
            scheduled[best_i] = True
            remaining -= 1
            
            for r in range(num_resources):
                if demands[best_i][r] > 0:
                    resource_intervals[r].append((feasible_start, feasible_start + dur, best_i))
            
        # Makespan is max end time
        if remaining > 0:
            return float('inf')
        return max(end_times) if n > 0 else 0.0

    # 4. Simulated Annealing
    # Initial state
    current_prio = initial_priority.copy()
    current_score = simulate(current_prio)
    
    best_prio = current_prio.copy()
    best_score = current_score
    
    # SA Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # We need to move fast.
    # n=30. Simulation is O(n^2 * R * N_intervals).
    # N_intervals grows with n. So O(n^3 * R).
    # 30^3 * 4 = 108,000 operations per sim.
    # 10,000 sims = 1e9 ops. Might be too slow in Python.
    # We need to limit iterations or optimize sim.
    
    # Optimization: Use numpy where possible? Hard with the list of intervals.
    # Let's estimate time.
    # 1000 sims should take < 1s if optimized.
    # Let's try 5000-10000 iterations.
    
    max_iterations = 5000
    
    for k in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Perturb: Swap two random priorities
        i, j = random.sample(range(n), 2)
        old_pi, old_pj = current_prio[i], current_prio[j]
        current_prio[i] = old_pj
        current_prio[j] = old_pi
        
        new_score = simulate(current_prio)
        
        # Acceptance
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_prio = current_prio.copy()
        else:
            # Revert
            current_prio[i] = old_pi
            current_prio[j] = old_pj
            
        T *= cooling
        if T < min_T:
            T = min_T
            
    return best_prio
