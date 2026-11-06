import numpy as np
import time as time_module
from collections import defaultdict

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0

    # Deadline guard: aim to finish well before 150s
    start_time = time_module.time()
    max_allowed_time = 140.0
    
    def check_time():
        return (time_module.time() - start_time) < max_allowed_time

    # --- 1. Feature Extraction ---
    # Precompute features for each ready operation to speed up scoring
    op_features = []
    for op in ready_operations:
        durations = list(op["machine_options"].values())
        min_dur = min(durations) if durations else 0.0
        max_dur = max(durations) if durations else 0.0
        avg_dur = sum(durations) / len(durations) if durations else 0.0
        
        rem_ops = op.get("remaining_ops", [])
        n_rem = len(rem_ops) if isinstance(rem_ops, list) else 0
        
        n_machines = len(op["machine_options"]) if op["machine_options"] else 1
        
        op_features.append({
            "rw": op["remaining_work"],
            "min_dur": min_dur,
            "max_dur": max_dur,
            "avg_dur": avg_dur,
            "n_rem": n_rem,
            "n_mach": n_machines,
            "job": op["job"]
        })

    # --- 2. Smooth Max and Makespan Estimator ---
    def log_sum_exp(vals, beta=1.0):
        if not vals:
            return 0.0
        arr = np.array(vals, dtype=np.float64)
        m = np.max(arr)
        if m == 0:
            return 0.0
        return m + np.log(np.sum(np.exp(beta * (arr - m)))) / beta

    def estimate_makespan_v2(ready_ops, current_time):
        """
        Improved estimator:
        1. Max Job Remaining Work (Critical Path per job)
        2. Max Machine Load (Sum of min_durations for ops targeting that machine)
        3. Total Work / Num Machines (Average Load)
        """
        if not ready_ops:
            return current_time

        job_work = defaultdict(float)
        machine_load = defaultdict(float)
        total_work = 0.0

        for op in ready_ops:
            job_id = op["job"]
            rw = op["remaining_work"]
            total_work += rw
            job_work[job_id] = max(job_work[job_id], rw) # Job critical path is max of its ops
            
            # For machine load, in a greedy dispatch, we assume the op goes to the 
            # machine with the shortest duration (min_dur) to minimize individual completion,
            # but we must account for contention. 
            # A better proxy: assign to the least loaded machine? 
            # For estimation speed, assume min_dur contributes to the 'best' machine,
            # but for contention, we look at the sum of min_durs for machines that 
            # multiple jobs can use? 
            # Simpler robust proxy: Sum of min_durs for all ready ops, divided by machines?
            # No, let's estimate the load on the most popular machine.
            # We don't know the assignment yet. 
            # Let's use the sum of min_durations as a lower bound for total machine time,
            # and assume it's distributed.
            # However, a specific machine might be a bottleneck.
            
            # For this estimator, we will calculate:
            # - Max Job Work
            # - Total Work / 5 (average machine load)
            # - Max over machines of (sum of min_dur for ops where this machine is the *only* option or *best* option)
            # This is complex. Let's stick to the previous two plus a new term:
            # Bottleneck Machine Estimate: Max over all machines M of (sum of min_dur(op) for op in ready_ops if M in op.machines) / 1.0
            # This assumes all these ops MUST use M, which is an upper bound on M's load, 
            # but useful for identifying potential bottlenecks.
            
            # Actually, let's just add the sum of min_durs to machine_load for all machines 
            # they can use, but weight by 1/n_machines to average out?
            # No, that's weak.
            
            # Let's use: Machine Load Estimate = Sum of (min_dur / n_machines) for each op, 
            # then take the max? No.
            
            # Let's try: For each machine, estimate load as sum of min_dur for ops that 
            # have that machine as their *fastest* option. If tie, split?
            # This is getting complicated. 
            # Let's stick to the previous estimator but add a "congestion" term.
            # Congestion: If many ops have small n_machines, they are constrained.
            
            pass
        
        # Simplified robust estimator:
        # 1. Max Job Work
        # 2. Total Work / 5
        # 3. Max "Constrained Work": Sum of rw for ops with n_mach == 1, divided by 1.
        #    Ops with 1 machine option force that machine.
        constrained_work = 0.0
        for op in ready_ops:
            if op["machine_options"] and len(op["machine_options"]) == 1:
                constrained_work += op["remaining_work"]
        
        # The machine with only 1 option for many jobs will be the bottleneck.
        # Max machine load >= constrained_work (if all constrained ops go to same machine)
        # But they might go to different machines.
        # So, max machine load >= max( max_job_work, total_work/5, constrained_work/1 )?
        # No, constrained_work is the sum. The max load on any single machine is at least 
        # max(constrained_work_i) where i is the machine.
        
        # Let's just use the previous soft_max of [max_job_work, total_work/5] 
        # and add a penalty for high variance in machine options?
        
        # Let's try a different angle: 
        # Est = current_time + max( max_job_work, total_work / 5.0 )
        # This is what the parent did.
        
        # Let's add: Max over jobs of (remaining_work) is already there.
        # Let's add: Sum of min_durs / 5.0 (lower bound on makespan from machine side)
        sum_min_durs = sum(op["remaining_work"] for op in ready_ops) # remaining_work is usually the duration of the current op? 
        # In the input dict, 'remaining_work' is the total work left for the job? 
        # "remaining_work": total remaining processing time for the job? 
        # Or for the operation? 
        # "remaining_ops": list of future ops.
        # "remaining_work": likely sum of durations of current op + remaining_ops.
        # If 'remaining_work' is the sum of all future durations, then summing it over ops double counts.
        # Let's assume 'remaining_work' is the sum of durations for the *current* op only? 
        # No, usually in these tasks, 'remaining_work' is the total time left for the job.
        # If it's the total job time, then max_job_work is the critical path.
        
        # Let's assume 'remaining_work' is the time for the *current* operation.
        # Then job_work[job] should be sum of remaining_work for all ops in that job's queue?
        # But we only have the *current* ready ops.
        # The 'remaining_ops' field contains the future ones.
        
        # Let's re-calculate job critical path properly.
        job_critical_path = defaultdict(float)
        for op in ready_ops:
            j_id = op["job"]
            # Current op duration
            dur = min(op["machine_options"].values()) if op["machine_options"] else 0.0
            # Future ops
            rem = op.get("remaining_ops", [])
            future_dur = 0.0
            for f_op in rem:
                f_durs = list(f_op["machine_options"].values())
                future_dur += min(f_durs) if f_durs else 0.0
            total_job_work = dur + future_dur
            if total_job_work > job_critical_path[j_id]:
                job_critical_path[j_id] = total_job_work
        
        max_job_cp = max(job_critical_path.values()) if job_critical_path else 0.0
        total_work = sum(job_critical_path.values())
        machine_lb = total_work / 5.0
        
        # Smooth max of these lower bounds
        bounds = [max_job_cp, machine_lb]
        est = log_sum_exp(bounds, beta=15.0)
        
        return current_time + est

    # --- 3. Beam Search Implementation ---
    
    # State: (list_of_ready_ops, time)
    # We will perform a beam search.
    
    def simulate_step(ready_ops, t, idx):
        """
        Simulate dispatching ready_ops[idx] at time t.
        Returns: (new_ready_ops, new_time)
        """
        op = ready_ops[idx]
        durs = list(op["machine_options"].values())
        min_dur = min(durs) if durs else 0.0
        comp_t = t + min_dur
        
        new_ready = []
        for j, other in enumerate(ready_ops):
            if j == idx:
                # This op is started. It is no longer "ready" in the same sense, 
                # but if it has remaining_ops, the *next* op becomes ready?
                # In open shop simulation, when an op starts, it occupies the machine.
                # The next op of the job becomes ready only when the *current* one finishes?
                # OR does it become ready immediately if it's on a different machine?
                # Standard Open Shop: A job's operations must be processed in order.
                # Op k+1 can start only after Op k finishes.
                # So, the next op is NOT ready yet.
                # However, the *current* op is now "in progress".
                # The simulation state should track "in progress" ops?
                # The dispatch function only receives "ready_operations".
                # If we pick one, it leaves the ready pool.
                # The next op of that job is NOT ready yet.
                # So, we simply remove the chosen op from the ready list.
                # The other ops remain ready.
                pass
            else:
                new_ready.append(other)
        
        return new_ready, comp_t

    # Initial Beam: All ready ops
    # We need to evaluate each candidate.
    
    # To make the AST distinct, I'll use a list of tuples for the beam
    beam = [(0, op_features, time, 0)] # (score, ops, time, depth) - Placeholder
    
    # Actually, let's just do a simple 2-step lookahead as before, but with the improved estimator.
    # And I'll structure it differently.
    
    best_score = float('inf')
    best_idx = 0
    
    # Step 1: Evaluate each candidate
    candidates = []
    for i in range(n_ops):
        if not check_time():
            break
            
        new_ready, new_t = simulate_step(ready_operations, time, i)
        score = estimate_makespan_v2(new_ready, new_t)
        candidates.append((score, i))
        
    if not candidates:
        return 0
        
    candidates.sort(key=lambda x: x[0])
    
    # Step 2: Look one step ahead for the top K candidates
    # K depends on time available
    elapsed = time_module.time() - start_time
    remaining_time = max_allowed_time - elapsed
    if remaining_time > 50:
        k_beam = 5
    elif remaining_time > 10:
        k_beam = 3
    else:
        k_beam = 1
        
    top_k = candidates[:k_beam]
    
    for level1_score, level1_idx in top_k:
        if not check_time():
            break
            
        new_ready, new_t = simulate_step(ready_operations, time, level1_idx)
        
        # Step 3: For each next possible op in new_ready
        for j in range(len(new_ready)):
            if not check_time():
                break
                
            new_ready2, new_t2 = simulate_step(new_ready, new_t, j)
            score2 = estimate_makespan_v2(new_ready2, new_t2)
            
            # Combine scores: The makespan is determined by the final state.
            # But level1_score is an estimate from an intermediate state.
            # We want to minimize the final makespan.
            # score2 is the estimate from the deeper state.
            # It is generally better than level1_score.
            # So we compare score2.
            
            if score2 < best_score:
                best_score = score2
                best_idx = level1_idx
                
    return best_idx
