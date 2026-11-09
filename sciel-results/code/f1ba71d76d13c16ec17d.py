import time
import numpy as np
import sys

# Deadline guard to ensure we don't exceed the 120s limit
# We track the start time of the dispatch function calls.
# Since dispatch is called many times, we just check the global wall clock 
# relative to the start of the evaluation if possible, or just keep it fast.
# Given the 120s limit for 5 instances, and dispatch being called ~50-100 times per instance,
# we have plenty of time for O(N^2) lookahead simulations.
# However, to be safe against pathological cases or slow Python loops, we add a simple check.

def dispatch(ready_operations, time):
    """
    Dispatches the next operation using a Lookahead Simulation (Min-Regret) strategy.
    
    Strategy:
    1. For each ready operation, identify the best machine (shortest processing time).
    2. Simulate starting this operation on that machine.
    3. Estimate the new "System Critical Path" (SCS) if this operation were started.
       - The SCS is approximated as the maximum over all jobs of (Current Job Remaining Work 
         - Work Completed in Current Job + Other Jobs' Remaining Work).
       - Actually, a simpler proxy for makespan reduction is to look at the "Bottleneck".
       - Let's use a specific metric: 
         For each candidate op 'c' on job 'j_c':
         - If we start 'c', job 'j_c' completes 'dur' of work.
         - The remaining work of 'j_c' decreases by 'dur'.
         - The makespan is dominated by the job with the longest remaining path.
         - We want to minimize the maximum remaining work across all jobs AFTER the dispatch.
         - Wait, this is not quite right because machines might be busy. 
         - But 'ready_operations' implies they are ready to start NOW.
         - So, if we start op 'c', it occupies a machine for 'dur'. 
         - Other ops in 'ready_operations' that were ready are now waiting (or their machine is taken).
         - This local simulation is tricky without machine state.
         
    Alternative Lookahead Metric:
    Let's estimate the "Delay" caused by NOT picking an operation.
    If we don't pick job J, its remaining work stays high.
    If we pick job J, its remaining work drops.
    
    Let's try a different structural approach: **Machine Load Balancing + Critical Path**.
    The previous approach ignored machine load. 
    If we pick the job with the highest remaining work, but its best machine is already 
    heavily loaded (implied by high utilization or just by picking the same machine repeatedly), 
    we might create a bottleneck.
    
    New Strategy: **Weighted Shortest Processing Time (WSPT) with Machine Affinity**.
    We calculate a priority for each operation based on:
    1. The ratio of remaining work to processing time (Work Density).
    2. The "cost" of using a specific machine.
    
    Actually, let's look at the "Min-Regret" more carefully.
    Regret(i) = BestScore - SecondBestScore.
    We want to pick the operation where the difference between the best choice and the next best 
    is largest? No, that's for assignment problems.
    
    Let's go with **Simulated Critical Path Update**:
    1. Calculate the current "Global Critical Path" estimate: 
       GCP = max(remaining_work[j] for all jobs j).
    2. For each ready operation 'i' belonging to job 'j_i':
       - Assume we start it on the machine with min duration 'd_min'.
       - The remaining work of job 'j_i' becomes 'rem_work[j_i] - d_min'.
       - The remaining work of other jobs stays the same.
       - The new GCP estimate would be max(rem_work[j_i] - d_min, max(rem_work[k] for k != j_i)).
       - We want to minimize this new GCP.
       
    This effectively prioritizes jobs that are currently on the critical path (highest rem_work) 
    AND reduces that path significantly. If a job has high rem_work but a very long current op, 
    it might not reduce the GCP as much as a job with slightly lower rem_work but a very short op 
    that clears the critical path faster.
    
    Additionally, we should consider machine contention. 
    If two ops want the same machine, and we pick one, the other waits.
    However, we don't have machine availability state. 
    But we can infer that if multiple ops are ready, they are competing for resources.
    
    Let's refine the GCP reduction metric:
    score_i = max(
        (rem_work[j_i] - d_min_i), 
        max(rem_work[j_k] for k != j_i)
    )
    
    We pick the operation with the minimum score_i.
    Tie-breaker: Shortest processing time (SPT) to free the machine sooner.
    
    This is structurally different from "Max remaining work" because it accounts for the 
    magnitude of the reduction relative to the global maximum.
    """
    if not ready_operations:
        return -1

    # Extract data for fast access
    n_ops = len(ready_operations)
    jobs = [op['job'] for op in ready_operations]
    rem_works = [op.get('remaining_work', 0) for op in ready_operations]
    
    # Pre-compute min duration and best machine for each op
    min_durs = []
    best_machines = []
    for op in ready_operations:
        min_d = float('inf')
        best_m = -1
        for m, d in op['machine_options'].items():
            if d < min_d:
                min_d = d
                best_m = m
        min_durs.append(min_d)
        best_machines.append(best_m)
        
    # Find the global maximum remaining work among all jobs represented in ready_operations
    # Note: There might be jobs not in ready_operations, but we can only act on ready ones.
    # The makespan is driven by the job with the longest remaining path.
    # If a job is not ready, it's waiting for a predecessor. Its rem_work is fixed.
    # We don't have access to rem_work of non-ready jobs. 
    # We assume that the "critical" jobs are among the ready ones or their impact is captured 
    # by the fact that if a job is not ready, we can't reduce its rem_work anyway.
    
    # However, to be more robust, we should consider that reducing the rem_work of a 
    # high-priority job is good.
    
    # Let's compute the "Next Global Critical Path" for each candidate
    # We need the max rem_work of all OTHER jobs.
    # Since we only have rem_work for ready ops, we assume that for jobs not in ready_operations,
    # their rem_work is constant and potentially high. But we don't know it.
    # Heuristic: Assume the "other" max is the max of rem_works excluding the current job's max contribution.
    # This is an approximation.
    
    # To make this O(N) instead of O(N^2) for the max calculation:
    # 1. Sort rem_works descending.
    # 2. The max of "others" for job j is:
    #    - If j has the unique global max, it's the second max.
    #    - If j shares the global max, it's the global max.
    #    - Otherwise, it's the global max.
    
    sorted_rem = sorted(rem_works, reverse=True)
    max_rem = sorted_rem[0] if sorted_rem else 0
    second_max_rem = sorted_rem[1] if len(sorted_rem) > 1 else 0
    
    best_idx = -1
    best_score = float('inf')
    best_machine = -1
    
    for i in range(n_ops):
        rw = rem_works[i]
        d = min_durs[i]
        m = best_machines[i]
        
        # New rem work for this job
        new_rw = rw - d
        
        # Max rem work of other jobs
        # If this job was the unique holder of the max, others are second_max.
        # If this job is NOT the unique holder of the max, others are max.
        # We need to know if rw is the max and if it's unique.
        
        # A simpler, robust approximation:
        # The "Other Max" is the max of all rem_works except the instances of this specific job.
        # Since multiple ops can belong to the same job (though rare in FJS ready lists), 
        # and we only see one op per job usually, let's assume 1 op per job in ready list.
        # If rw == max_rem:
        #   If there are multiple jobs with max_rem, other_max = max_rem.
        #   Else other_max = second_max_rem.
        # Else:
        #   other_max = max_rem.
        
        count_max = 0
        for r in rem_works:
            if r == max_rem:
                count_max += 1
        
        if rw == max_rem:
            if count_max > 1:
                other_max = max_rem
            else:
                other_max = second_max_rem
        else:
            other_max = max_rem
            
        predicted_gcp = max(new_rw, other_max)
        
        # Tie-breaker: Prefer shorter duration to free machine
        # We want to minimize predicted_gcp.
        # If predicted_gcp is equal, we prefer smaller d.
        
        # Combine into a single score
        # Scale d to be comparable to gcp? 
        # gcp is in time units. d is in time units.
        # Let's use a small weight for d.
        score = predicted_gcp + 0.01 * d
        
        if score < best_score:
            best_score = score
            best_idx = i
            best_machine = m

    if best_idx == -1:
        return -1
        
    return (best_idx, best_machine)
