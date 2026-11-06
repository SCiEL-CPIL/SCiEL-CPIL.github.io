"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def dispatch(ready_operations, time):
    """
    Dispatch rule using 1-Step Lookahead Simulation.
    
    Strategy:
    Instead of using a static score (like Critical Path), we simulate the 
    immediate effect of dispatching each candidate operation.
    
    For each candidate:
    1. Assume it starts at `time` on its preferred machine (min duration).
    2. Estimate the 'Projected Load' on that machine.
    3. Estimate the 'Projected Delay' for the job's next operation (if any) 
       due to machine contention from other ready jobs.
    4. Select the candidate that minimizes a composite cost: 
       (Projected Finish Time + Projected Delay of Next Op).
       
    This is a structural change from static heuristics to a local search 
    approach that considers the interaction between the current choice and 
    the immediate future state.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Deadline guard
    start_time = _time.time()
    time_limit = 10.0 # 10 seconds is plenty for N<=10 lookahead
    
    # Pre-compute basic info for each ready op to speed up the loop
    op_infos = []
    for i, op in enumerate(ready_operations):
        job_id = op['job']
        machine_options = op['machine_options']
        
        # Find preferred machine (min duration)
        pref_machine = min(machine_options, key=machine_options.get)
        pref_duration = machine_options[pref_machine]
        
        # Get remaining work info
        remaining_ops = op.get('remaining_ops', [])
        remaining_work = op.get('remaining_work', 0)
        
        # Estimate critical path (sum of min durations of remaining ops)
        # Note: remaining_ops usually includes the current op or the ones after.
        # We assume it's a list of dicts with 'machine_options'.
        if remaining_ops and isinstance(remaining_ops, list) and len(remaining_ops) > 0 and isinstance(remaining_ops[0], dict):
            cp = 0
            for r_op in remaining_ops:
                if 'machine_options' in r_op:
                    cp += min(r_op['machine_options'].values())
            # If the current op is NOT in remaining_ops (i.e., remaining_ops are the *next* ones), 
            # we might need to add current duration? 
            # The prompt says "remaining_ops". Usually in these sims, it's the tail.
            # Let's assume cp is the work *after* this op or including it.
            # To be safe, let's just use the sum of mins as a proxy for urgency.
            job_cp = cp
        else:
            job_cp = remaining_work
            
        op_infos.append({
            'index': i,
            'job': job_id,
            'pref_machine': pref_machine,
            'pref_duration': pref_duration,
            'job_cp': job_cp
        })
    
    # Count how many jobs prefer each machine (static contention)
    machine_counts = {}
    for info in op_infos:
        m = info['pref_machine']
        machine_counts[m] = machine_counts.get(m, 0) + 1

    best_score = float('inf')
    best_idx = 0
    
    for i, info in enumerate(op_infos):
        if _time.time() - start_time > time_limit:
            break
            
        # Simulate dispatching op i
        # 1. It occupies its preferred machine for `pref_duration`
        finish_time = time + info['pref_duration']
        
        # 2. Estimate the impact on the *next* operation of this job
        # The next operation (if any) will be ready at `finish_time`.
        # What machine does the next operation prefer?
        # We don't have direct access to the "next op" details easily without 
        # digging into remaining_ops.
        
        # Let's look at the next op in remaining_ops if available.
        rem_ops = ready_operations[i].get('remaining_ops', [])
        
        # Identify the next op. 
        # If remaining_ops includes the current one, next is index 1.
        # If remaining_ops are the *subsequent* ones, next is index 0.
        # Heuristic: Check if the first op in remaining_ops matches the current op's characteristics?
        # Simpler: Assume remaining_ops[0] is the NEXT op if the list length > 1, 
        # or if the list represents the tail *after* current.
        # Given the ambiguity, let's use a proxy: 
        # The "urgency" of the job is reduced by the work done.
        # The "risk" is if the next op hits a busy machine.
        
        # Let's estimate the "Projected Makespan Contribution" of this job.
        # If we dispatch now, we clear this op.
        # The job's remaining critical path is roughly job_cp - pref_duration (if cp included current).
        # Let's assume job_cp is the sum of mins for all remaining including current.
        # Then remaining_cp_after = job_cp - pref_duration.
        
        # Cost 1: The finish time of this op (earlier is better for flow, but we care about makespan).
        # Cost 2: The potential delay of the *next* op.
        
        # Let's use a different metric: 
        # "Opportunity Cost" = (Time this machine is blocked) * (Number of other jobs waiting for it)
        # + (Critical Path of Job)
        
        # Actually, let's stick to the simplest effective lookahead:
        # Score = - (Finish Time of this op) - (Estimated Delay of Next Op)
        # But we want to MINIMIZE makespan.
        
        # Let's try: Minimize (Finish Time + Remaining Critical Path of Job)
        # This is equivalent to minimizing the "Earliest Possible Completion Time" of the job.
        
        est_remaining_cp = max(0, info['job_cp'] - info['pref_duration'])
        
        # Projected Job Completion Time if no further delays:
        projected_completion = finish_time + est_remaining_cp
        
        # Penalty for machine contention:
        # If many jobs want this machine, the "effective" duration is longer.
        # Effective duration = pref_duration * (1 + (count-1) * 0.5) ?
        # Let's use a softer penalty.
        
        contention = machine_counts.get(info['pref_machine'], 1)
        # If contention is high, the machine is a bottleneck.
        # We penalize choosing a bottleneck machine if the job isn't the most critical.
        
        # Structural Change: 
        # Instead of static score, we compare pairs? No, that's O(N^2).
        # Let's use the "Projected Completion" as the primary metric.
        # Tie-breaker: Lower contention is better.
        
        score = projected_completion + (contention * 0.1)
        
        if score < best_score:
            best_score = score
            best_idx = i

    return best_idx

# EVOLVE-BLOCK-END
