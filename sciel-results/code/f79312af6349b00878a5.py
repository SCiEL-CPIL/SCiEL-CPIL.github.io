"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import heapq

# Stateful tracking for CPM-based dispatch
# We maintain the current estimated free time for each machine.
# This allows us to simulate the impact of each candidate dispatch.
_machine_free_times = {}
_start_time = None
_dispatch_count = 0
_last_time = -1

def dispatch(ready_operations, time):
    global _start_time, _dispatch_count, _last_time, _machine_free_times
    
    # Detect new instance (time goes backwards or significantly resets)
    # This is crucial for stateful heuristics to avoid carrying over stale machine times
    if time < _last_time - 1e-6 or (time == 0 and _last_time > 0):
        _machine_free_times = {}
        _dispatch_count = 0
    _last_time = time
    
    if _start_time is None:
        _start_time = _time.time()
        
    # Deadline guard: If we are running too long, fall back to a very fast heuristic
    # to ensure we finish within the 120s limit.
    if _time.time() - _start_time > 105:
        # Fallback: SPT on the machine that is free earliest
        best_idx = 0
        best_key = (float('inf'), float('inf'))
        for i, op in enumerate(ready_operations):
            options = op["machine_options"]
            if not options:
                continue
            min_completion = float('inf')
            for m, dur in options.items():
                free_t = _machine_free_times.get(m, 0)
                completion = max(time, free_t) + dur
                if completion < min_completion:
                    min_completion = completion
            if (min_completion, 0) < best_key:
                best_key = (min_completion, 0)
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0

    _dispatch_count += 1
    
    # Strategy: Critical Path Method (CPM) with Machine Contention Awareness
    # 
    # 1. For each ready operation, estimate the "Critical Path Length" (CPL) if we schedule it now.
    #    CPL = (Time until this op finishes) + (Estimated time to finish the rest of the job)
    #
    # 2. To estimate "Time to finish the rest of the job":
    #    - We look at the remaining operations.
    #    - For each remaining op, we estimate its start time based on the *current* machine free times 
    #      (plus a buffer for the op we are currently scheduling).
    #    - We sum these estimated durations to get a rough total remaining work.
    #    - To be more accurate, we use a "bottleneck-aware" sum: for each machine, we track the 
    #      cumulative load. The job's remaining time is roughly the max over machines of the 
    #      cumulative load assigned to that job on that machine, but since we don't know the 
    #      assignment of future ops, we use a weighted average or simple sum of min-durations.
    #
    #    A robust proxy for remaining time in Job-Shop is:
    #    Sum of (min_duration_of_op / max(1, number_of_ops_on_that_machine_in_job_remaining))
    #    However, a simpler and often effective heuristic is the sum of the minimum possible 
    #    durations of the remaining operations, adjusted by the current machine load.
    #
    #    Let's use: Estimated_Remaining_Time = Sum(min_duration(op) for op in remaining_ops)
    #    This ignores machine contention for future ops, but combined with the current op's 
    #    completion time (which *does* account for current contention), it provides a good 
    #    signal of which job is furthest from completion.
    #
    #    To improve this, we can add a "contention penalty" to the remaining time if the 
    #    machines are heavily loaded.
    
    candidates = []
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        job_id = op["job"]
        remaining_ops = op.get("remaining_ops", [])
        
        # 1. Calculate the earliest completion time for THIS operation
        #    We choose the machine that minimizes the completion time for this specific op.
        best_m_for_op = None
        min_completion_for_op = float('inf')
        
        for m, dur in options.items():
            free_t = _machine_free_times.get(m, 0)
            start_t = max(time, free_t)
            completion_t = start_t + dur
            
            if completion_t < min_completion_for_op:
                min_completion_for_op = completion_t
                best_m_for_op = m

        if best_m_for_op is None:
            continue
            
        # 2. Estimate the remaining work for the job
        #    We sum the minimum possible duration of each remaining operation.
        #    This represents the theoretical minimum time left if machines were never busy.
        remaining_work_min = 0
        for rem_op in remaining_ops:
            rem_opts = rem_op.get("machine_options", {})
            if rem_opts:
                remaining_work_min += min(rem_opts.values())
        
        # 3. Calculate the Critical Path Estimate
        #    The job will be "done" roughly at:
        #    completion_of_this_op + remaining_work_min
        #    However, if the machine for this op is very busy, it might delay subsequent ops 
        #    that need the same machine. A simple additive model is a good first approximation.
        
        cpm_estimate = min_completion_for_op + remaining_work_min
        
        # Tie-breakers:
        # If CPM estimates are close, we should prefer:
        # 1. The operation that uses the machine with the *highest* current load (to avoid 
        #    starving a busy machine? No, usually we want to balance load. But CPM already 
        #    accounts for load via min_completion_for_op).
        # 2. The job with the most remaining operations (complexity)?
        # 3. Let's use a secondary metric: "Urgency" = remaining_work_min / (number of remaining ops + 1)
        #    This is the average remaining op length. Shorter average ops might be easier to schedule.
        
        # Actually, a better tie-breaker for CPM is to prefer the job that has the *highest* 
        # ratio of remaining work to current completion time? No.
        
        # Let's stick to CPM as primary. For ties, prefer the op with the smallest 
        # "slack" relative to its job's total remaining work.
        
        candidates.append((cpm_estimate, i, best_m_for_op))
        
    if not candidates:
        return 0

    # Sort by CPM estimate (lower is better, as we want to finish the critical path fastest)
    candidates.sort(key=lambda x: x[0])
    
    best_candidate = candidates[0]
    best_idx = best_candidate[1]
    best_machine = best_candidate[2]
    
    # Update global state with the chosen assignment
    sel_op = ready_operations[best_idx]
    dur = sel_op["machine_options"][best_machine]
    start_t = max(time, _machine_free_times.get(best_machine, 0))
    finish_t = start_t + dur
    _machine_free_times[best_machine] = finish_t
    
    # Return (index, machine) to ensure state consistency
    return (int(best_idx), best_machine)

# EVOLVE-BLOCK-END
