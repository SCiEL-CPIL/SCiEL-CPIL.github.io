"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

_DEADLINE_SECONDS = 145.0
_START_TIME = time.time()

def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)

def _run_serial_sgs(instance, priority_vec):
    """
    Simulates the serial schedule generation scheme.
    Returns the makespan.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    if n == 0:
        return 0.0
        
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # Build successors and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i]
        if preds:
            in_degree[i] = len(preds)
            for p in preds:
                successors[p].append(i)
                
    # State
    scheduled = [False] * n
    start_times = np.zeros(n, dtype=float)
    current_time = 0.0
    
    # Resource usage tracking: list of (end_time, amount) for each resource
    # Or simpler: keep a list of active jobs and check availability when starting
    # Since n is small (30), we can just check all active jobs each time.
    
    # Active jobs: list of (job_id, end_time, demands_vector)
    active_jobs = []
    
    # Eligible set
    # We need to pick the highest priority eligible job at each step.
    # Eligible: all predecessors scheduled AND resources available at current_time?
    # Serial SGS definition: 
    # 1. Identify set of eligible activities (predecessors scheduled).
    # 2. Select the one with highest priority.
    # 3. Determine start time: max(current_time, earliest time resources are available for that activity).
    #    Wait, standard Serial SGS:
    #    - At each step, consider all unscheduled activities whose predecessors are all scheduled.
    #    - Select the one with highest priority.
    #    - Schedule it at the earliest possible time given resource constraints and current time.
    #    - Update current time to the finish time of the scheduled activity? 
    #    No, current time is usually the "decision point". 
    #    Standard Serial SGS:
    #    t = 0
    #    while unscheduled:
    #       eligible = {i | preds(i) scheduled}
    #       select i* from eligible with max priority
    #       start[i*] = max(t, min_time_resources_available(i*))
    #       t = start[i*]  (or does t stay? usually t is the current decision epoch)
    #       Actually, in Serial SGS, you schedule one job, then the next job's start time is max(previous_decision_time, resource_availability).
    #       Let's implement the standard:
    #       The "current time" for the next decision is the start time of the job just scheduled? 
    #       No, typically:
    #       t = 0
    #       while ...:
    #          eligible = ...
    #          pick i
    #          start_i = max(t, resource_availability_i)
    #          t = start_i  # The clock advances to the start of the new job? 
    #          # Or does t advance to finish? 
    #          # If t advances to start, we might schedule overlapping jobs.
    #          # If t advances to finish, it's parallel SGS?
    #          # Serial SGS: Jobs are scheduled one by one. The start time of the next job is at least the start time of the previous job? 
    #          # No, that's not right. 
    #          # Let's look up "Serial Schedule Generation Scheme".
    #          # "The serial SGS schedules one activity at a time. At each step, the activity with the highest priority among the eligible activities is scheduled at the earliest possible time."
    #          # "Earliest possible time" is determined by resource availability and predecessor finish times.
    #          # The "current time" variable in the algorithm usually tracks the time at which we are making the decision. 
    #          # However, since we pick the *highest priority* eligible, we don't necessarily wait for time to pass unless resources are blocked.
    #          # Actually, the standard implementation:
    #          # 1. Initialize t=0.
    #          # 2. Find eligible set E (preds done).
    #          # 3. Pick j in E with max priority.
    #          # 4. Determine start time s_j = max(t, min_time_resources_available(j)).
    #          # 5. Set t = s_j? Or t = finish_j? 
    #          # If t = s_j, the next job can start at s_j if resources allow.
    #          # If t = finish_j, it forces sequential start times, which is not standard.
    #          # Most implementations set the "current time" to the start time of the newly scheduled job for the next iteration's lower bound, 
    #          # BUT resource availability is the hard constraint.
    #          # Let's assume t is the lower bound for start times.
            
    scheduled_count = 0
    while scheduled_count < n:
        # Find eligible
        eligible = []
        for i in range(n):
            if not scheduled[i]:
                # Check preds
                if all(scheduled[p] for p in predecessors[i]):
                    eligible.append(i)
                    
        if not eligible:
            # Should not happen in a DAG
            break
            
        # Select highest priority
        # priority_vec is the score. Higher is better.
        # If there's a tie, pick lower index? Or just max.
        best_job = max(eligible, key=lambda x: priority_vec[x])
        
        # Determine earliest start time for best_job
        # 1. Predecessor constraint: max(finish[p] for p in preds)
        es_pred = 0.0
        for p in predecessors[best_job]:
            if start_times[p] + durations[p] > es_pred:
                es_pred = start_times[p] + durations[p]
                
        # 2. Resource constraint
        # We need to find the earliest time >= es_pred and >= current_time (if we use that) 
        # where resources are available.
        
        # Let's track resource usage as a list of (end_time, amount) for each resource.
        # Or simpler: for each resource r, we have a list of active intervals.
        # Since n=30, we can just check all scheduled jobs that overlap.
        
        # Let's maintain resource_usage: list of (start, end, amount) for each resource?
        # Or just check all scheduled jobs to see if they conflict.
        
        # Efficient way:
        # For resource r, find the latest end time of any scheduled job that uses resource r 
        # and starts before the candidate start time? 
        # Actually, we need the earliest time T such that for all r, sum(demands[j][r] for j active at T) + demand[best_job][r] <= capacity[r].
        
        # Candidate start times:
        # - es_pred
        # - current_time (if we track it)
        # - End times of all scheduled jobs that use resources demanded by best_job.
        
        candidate_times = set([es_pred])
        if demands is not None:
            for r in range(len(demands[best_job])):
                if demands[best_job][r] > 0:
                    # Add end times of all jobs using resource r
                    for j in range(n):
                        if scheduled[j] and demands[j][r] > 0:
                            candidate_times.add(start_times[j] + durations[j])
        
        # Also consider 0
        candidate_times.add(0.0)
        
        earliest_start = float('inf')
        
        # Sort candidate times
        for t in sorted(candidate_times):
            if t < es_pred:
                continue
            # Check if resources are available at t
            # A resource is available at t if sum of demands of jobs active at t <= capacity
            # Job j is active at t if start_times[j] <= t < start_times[j] + durations[j]
            
            feasible = True
            if demands is not None and capacities is not None:
                for r in range(len(demands[best_job])):
                    if demands[best_job][r] == 0:
                        continue
                    current_load = 0.0
                    for j in range(n):
                        if scheduled[j]:
                            s_j = start_times[j]
                            e_j = s_j + durations[j]
                            if s_j <= t < e_j:
                                current_load += demands[j][r]
                    if current_load + demands[best_job][r] > capacities[r]:
                        feasible = False
                        break
            if feasible:
                earliest_start = t
                break
                
        if earliest_start == float('inf'):
            # Fallback
            earliest_start = es_pred
            
        # Schedule
        start_times[best_job] = earliest_start
        scheduled[best_job] = True
        scheduled_count += 1
        
    makespan = np.max(start_times + durations) if n > 0 else 0.0
    return makespan

def priority(instance):
    """
    Simulates several static priority heuristics and returns the priority vector
    that yields the best (lowest) makespan in the simulation.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    if n == 0:
        return np.array([])
        
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # Pre-compute some metrics
    # 1. Longest Duration (LPT)
    p_lpt = durations.copy()
    
    # 2. Most Successors
    # Count successors for each node
    succ_count = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            succ_count[p] += 1
    p_ms = succ_count.copy()
    
    # 3. Slack (Lowest Slack first -> High Priority)
    # Compute Forward/Backward pass
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    ls = np.zeros(n, dtype=float)
    lf = np.zeros(n, dtype=float)
    
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i]
        if preds:
            in_degree[i] = len(preds)
            for p in preds:
                successors[p].append(i)
                
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    for u in topo_order:
        if predecessors[u]:
            es[u] = max(ef[p] for p in predecessors[u])
        else:
            es[u] = 0.0
        ef[u] = es[u] + durations[u]
        
    makespan_no_res = np.max(ef) if n > 0 else 0.0
        
    for u in reversed(topo_order):
        if successors[u]:
            lf[u] = min(ls[s] for s in successors[u])
        else:
            lf[u] = makespan_no_res
        ls[u] = lf[u] - durations[u]
        
    slack = ls - es
    # Priority = -Slack (lower slack = higher priority)
    p_slack = -slack
    
    # 4. Resource Weighted Slack
    # Pressure = sum(demand/capacity)
    pressure = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        try:
            d_mat = np.asarray(demands, dtype=float)
            c_vec = np.asarray(capacities, dtype=float)
            normalized = d_mat / c_vec
            pressure = np.sum(normalized, axis=1)
        except Exception:
            pressure = np.zeros(n, dtype=float)
            
    # Normalize slack and pressure
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0
    max_pressure = np.max(pressure) if np.max(pressure) > 0 else 1.0
    norm_slack = slack / max_slack
    norm_pressure = pressure / max_pressure
    
    # Priority = (1 - norm_slack) + weight * norm_pressure
    # Let's try a few weights
    p_rw1 = (1.0 - norm_slack) + 0.5 * norm_pressure
    p_rw2 = (1.0 - norm_slack) + 1.0 * norm_pressure
    p_rw3 = (1.0 - norm_slack) + 2.0 * norm_pressure
    
    # 5. Random (to check if structure matters)
    # Not useful for deterministic return, skip.
    
    # List of candidate priority vectors
    candidates = [
        ("LPT", p_lpt),
        ("MS", p_ms),
        ("Slack", p_slack),
        ("RW1", p_rw1),
        ("RW2", p_rw2),
        ("RW3", p_rw3),
    ]
    
    best_score = float('inf')
    best_vector = p_lpt # Default
    
    # Check time budget
    if _remaining_budget() < 10.0:
        # Not enough time to simulate all, just return one
        return best_vector
        
    for name, vec in candidates:
        # Ensure vector is float
        vec = np.asarray(vec, dtype=float)
        # Simulate
        try:
            makespan = _run_serial_sgs(instance, vec)
            if makespan < best_score:
                best_score = makespan
                best_vector = vec
        except Exception:
            pass
            
        if _remaining_budget() < 5.0:
            break
            
    return best_vector

# EVOLVE-BLOCK-END
