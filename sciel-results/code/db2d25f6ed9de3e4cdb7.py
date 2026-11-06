"""
RCPSP Heuristic: Iterative Feedback Priority
Strategy:
1. Initial Priority: Use a strong static heuristic (e.g., Min Total Float + Duration) to generate an initial priority order.
2. Simulation: Run a full Serial Schedule Generation Scheme (SGS) using this initial order.
3. Feedback: For each activity, calculate the 'delay' it experienced due to resource contention (Actual Start - Earliest Start).
4. Refinement: Update priority scores to penalize activities that caused significant delays to others or suffered high delays themselves (indicating they are bottlenecks or critical).
5. Iteration: Repeat simulation with updated priorities a few times to converge on a better static priority vector.

This approach moves beyond static graph properties to incorporate dynamic resource interaction effects.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Returns a priority score for each activity.
    Higher score = higher priority.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin for 150s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Handle potential missing keys or different formats
    preds_raw = instance.get("predecessors", instance.get("preds", [[] for _ in range(n)]))
    demands_raw = instance.get("demands", instance.get("resource_demands", None))
    capacities_raw = instance.get("capacities", instance.get("resource_capacities", None))
    
    # Normalize predecessors to list of lists of ints
    preds = [list(p) if p else [] for p in preds_raw]
    if len(preds) < n:
        preds = [[] for _ in range(n)]
        for i in range(n):
            if i < len(preds_raw) and preds_raw[i]:
                preds[i] = list(preds_raw[i])
                
    # Build successors
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            if 0 <= i < n:
                succs[i].append(j)
                
    # Handle resources
    if demands_raw is not None:
        try:
            demands = np.asarray(demands_raw, dtype=float)
            if demands.ndim == 1:
                demands = demands.reshape(-1, 1)
        except:
            demands = np.zeros((n, 1))
    else:
        demands = np.zeros((n, 1))
        
    if capacities_raw is not None:
        try:
            capacities = np.asarray(capacities_raw, dtype=float)
            if capacities.ndim == 0:
                capacities = np.array([capacities])
        except:
            capacities = np.ones(demands.shape[1])
    else:
        capacities = np.ones(demands.shape[1])
        
    m_resources = demands.shape[1]
    if len(capacities) < m_resources:
        capacities = np.pad(capacities, (0, m_resources - len(capacities)), mode='constant')
        
    # Topological Sort
    in_degree = np.array([len(preds[i]) for i in range(n)], dtype=int)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) < n:
        return durations.copy()
        
    # CPM Forward Pass for Earliest Starts (ES)
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in topo_order:
        es[u] = 0
        for p in preds[u]:
            if ef[p] > es[u]:
                es[u] = ef[p]
        ef[u] = es[u] + durations[u]
        
    # CPM Backward Pass for Latest Starts (LS)
    makespan_cpm = np.max(ef)
    lf = np.zeros(n)
    ls = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) > 0:
            lf[u] = np.min([ls[v] for v in succs[u]])
        else:
            lf[u] = makespan_cpm
        ls[u] = lf[u] - durations[u]
        
    tf = ls - es
    max_tf = np.max(tf) if np.max(tf) > 0 else 1.0
    crit_score = 1.0 - (tf / max_tf)
    
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    dur_score = durations / max_dur

    # Initial Priority: Strong Criticality + Duration
    # We want to start with a good baseline to simulate
    current_priority = 2.0 * crit_score + 1.0 * dur_score
    
    # Iterative Feedback Loop
    # We simulate the SGS, measure how much each task is "delayed" by resources,
    # and how much "pressure" it exerts.
    # If a task is delayed, it's likely competing for resources. 
    # If a task causes others to be delayed, it's a bottleneck.
    
    # To keep it fast and robust, we do a few iterations of:
    # 1. Simulate SGS with current_priority
    # 2. Calculate 'resource_delay' for each task (Start_Time - ES)
    # 3. Update priority: High priority for tasks with high delay (they are critical/contended)
    #    BUT we must be careful: if we prioritize everything that gets delayed, we might just reinforce the delay.
    #    Instead, we look at the "impact". 
    #    Let's use a simpler metric: The "slack" in the actual schedule.
    #    Tasks that finish late relative to their CPM Latest Finish are critical in the actual schedule.
    
    iterations = 10
    # We will accumulate a "feedback score"
    feedback_score = np.zeros(n)
    
    for _ in range(iterations):
        if time.time() - start_time > DEADLINE:
            break
            
        # Run SGS
        # State: 
        # - resource_usage over time is complex to track exactly without event simulation.
        # - For SGS, we pick the highest priority eligible task, start it at the earliest possible time 
        #   considering precedence AND resource availability.
        
        # To simulate efficiently:
        # We maintain a list of scheduled tasks and their start/end times.
        # For a new task, we check all future time points (integers) to find the first slot where resources fit.
        # Since n=30, this is very fast.
        
        scheduled = {} # task_id -> (start, end)
        remaining = set(range(n))
        # Track resource usage at each time step? 
        # Max duration is likely small. Let's find max possible time.
        max_time = int(np.sum(durations)) + 1
        
        # resource_usage[t][r] = amount of resource r used at time t
        # Using a dict of lists or numpy array. 
        # For n=30, max_time might be ~100-200. 200*4 floats is small.
        resource_usage = np.zeros((max_time, m_resources))
        
        # Eligible tasks: predecessors scheduled.
        # We need to pick the highest priority eligible task.
        # "Eligible" means all preds are in 'scheduled'.
        
        # Optimization: Precompute predecessors
        # We'll iterate n times.
        
        for _ in range(n):
            # Find eligible tasks
            eligible = []
            for i in remaining:
                if all(p in scheduled for p in preds[i]):
                    eligible.append(i)
            
            if not eligible:
                break # Should not happen if topo valid
                
            # Pick highest priority
            # current_priority is the score. Higher is better.
            # We need to handle ties or just pick max.
            best_task = max(eligible, key=lambda x: current_priority[x])
            
            # Find earliest start time for best_task
            start_t = 0
            # Earliest start based on precedence
            pred_end = 0
            for p in preds[best_task]:
                if p in scheduled:
                    end_p = scheduled[p][1]
                    if end_p > pred_end:
                        pred_end = end_p
            start_t = pred_end
            
            # Check resource constraints
            # We need to find the first t >= start_t such that for all resources r:
            # resource_usage[t..t+dur-1, r] + demand[best_task, r] <= capacity[r]
            
            dur = int(durations[best_task])
            valid_start = -1
            
            # Scan forward
            for t in range(start_t, max_time - dur + 1):
                # Check if resources are available from t to t+dur-1
                # We can check the max usage in this window + demand <= capacity
                # For speed, since n is small, we can just sum the usage in the window? 
                # No, usage is per time step.
                # Simple check: for each r, max(resource_usage[t:t+dur, r]) + demand[best_task, r] <= capacity[r]
                is_valid = True
                for r in range(m_resources):
                    # If demand is 0, skip?
                    if demands[best_task, r] == 0:
                        continue
                    # Max usage in window
                    # If dur is 0, check t. If dur > 0, check t to t+dur-1
                    if dur > 0:
                        max_usage = np.max(resource_usage[t:t+dur, r])
                    else:
                        max_usage = resource_usage[t, r]
                    
                    if max_usage + demands[best_task, r] > capacities[r]:
                        is_valid = False
                        break
                
                if is_valid:
                    valid_start = t
                    break
            
            if valid_start == -1:
                # Fallback: schedule at max_time (should not happen if capacities are reasonable)
                valid_start = max_time - dur if max_time >= dur else 0
            
            # Schedule it
            scheduled[best_task] = (valid_start, valid_start + dur)
            remaining.remove(best_task)
            
            # Update resource usage
            for t in range(valid_start, valid_start + dur):
                resource_usage[t, :] += demands[best_task, :]
                
        # After simulation, calculate feedback
        # For each task, how much was it delayed?
        # Delay = Actual Start - CPM Earliest Start
        # If a task is delayed, it means resources were tight.
        # We want to prioritize tasks that are "critical" in the resource-constrained sense.
        # A common heuristic is to prioritize tasks with high "Resource Criticality".
        # Let's define feedback based on the delay.
        # If a task has high delay, it's suffering. 
        # However, simply prioritizing delayed tasks might just move the delay.
        # A better signal: 
        # 1. Tasks on the actual critical path (finish time = makespan) should be high priority.
        # 2. Tasks that "block" others.
        
        # Let's compute the actual makespan
        actual_makespan = max(end for _, end in scheduled.values())
        
        # Identify tasks that are "late" relative to CPM
        # We can use the delay itself as a signal of contention.
        # But we also want to favor tasks that *cause* contention.
        
        # Alternative Feedback: 
        # Calculate the "slack" in the actual schedule.
        # If a task has low slack in the actual schedule, it's critical.
        # Let's approximate actual slack by seeing how much we could push it later?
        # That's expensive.
        
        # Let's use a simpler proxy:
        # Score = Actual_Delay + (1 if task is on actual critical path else 0)
        # Actually, let's look at the "pressure" again.
        
        # New Idea: Use the delay to update the "urgency".
        # If a task was delayed, it implies that the resources it needs were busy.
        # Those resources were likely busy with other tasks that were scheduled earlier.
        # This is complex to trace.
        
        # Let's try a different feedback: 
        # "Regret" based. 
        # If we had scheduled this task earlier, would the makespan have been better?
        # Hard to know without re-simulating.
        
        # Let's stick to a robust static improvement:
        # The initial priority was Crit + Dur.
        # In the simulation, we observed which tasks were delayed.
        # Tasks that are delayed are likely "bottlenecked".
        # We should increase the priority of tasks that have high "resource intensity" AND were delayed.
        # Or, we can just use the delay as a tie-breaker or additive term.
        
        # Let's add the delay to the priority score, normalized.
        delays = np.zeros(n)
        for i, (s, e) in scheduled.items():
            delays[i] = s - es[i]
            
        max_delay = np.max(delays) if np.max(delays) > 0 else 1.0
        delay_score = delays / max_delay
        
        # Update priority: 
        # We want to boost tasks that are critical (high crit_score) AND have experienced delay (high delay_score).
        # This identifies tasks that are both important and resource-constrained.
        # Also, tasks with long duration that are delayed are very bad.
        
        # Blend:
        # current_priority = 0.8 * current_priority + 0.2 * (crit_score + delay_score)
        # This allows the priority to adapt to the specific resource contention pattern observed.
        
        # To avoid oscillation, we mix with the original strong heuristic.
        # The original heuristic is a good prior. The delay is the "observation".
        
        # Let's try: 
        # New Priority = Crit + Dur + 0.5 * Delay_Score
        # This effectively says: "Be critical, be long, and if you're being delayed, you're important to fix."
        
        # Wait, if I prioritize delayed tasks, they get scheduled earlier next time, reducing their delay.
        # This is a feedback loop that should reduce overall delay/makespan.
        
        # Let's update:
        current_priority = 1.5 * crit_score + 1.0 * dur_score + 0.5 * delay_score
        
        # We don't strictly need to iterate if the first simulation gives good delay info.
        # But iterating allows the delay pattern to stabilize.
        # Given n=30, 10 iterations is very fast.
        
    # Final Priority
    # Ensure it's a numpy array of floats
    return current_priority

# EVOLVE-BLOCK-END
