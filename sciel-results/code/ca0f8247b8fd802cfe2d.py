"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

def dispatch(ready_operations, time):
    """
    Stateless dispatch rule using Critical Path Method (CPM) proxy.
    
    Strategy:
    1. Job Priority: Prioritize jobs with the longest remaining total processing time.
       We approximate the critical path length by summing the minimum possible duration
       for each remaining operation in the job. This ensures we start the jobs that 
       have the most "critical" work left, reducing the risk of them becoming bottlenecks 
       later in the schedule.
    2. Machine Selection: For the selected job's operation, choose the machine with the 
       shortest processing time (SPT) to minimize the immediate load on that machine 
       and potentially free it up sooner for other jobs.
    
    This approach is stateless, avoiding potential errors from incorrect state tracking 
    across instances or dispatch calls, and relies on a robust global heuristic.
    """
    if not ready_operations:
        return 0

    best_index = 0
    best_machine = None
    best_priority = -1
    best_spt = float('inf')

    # Calculate priority for each ready operation
    for i, op in enumerate(ready_operations):
        options = op.get("machine_options", {})
        if not options:
            continue
        
        # 1. Job Priority: Longest Remaining Processing Time (LRPT) proxy
        # Sum the minimum duration for each remaining operation in this job.
        # We assume 'remaining_ops' gives the count, but we don't have the durations 
        # of future ops directly. 
        # However, 'remaining_work' is provided in the dict! 
        # Let's check the dict keys: job, op, machine_options, remaining_ops, remaining_work.
        # 'remaining_work' is likely the sum of durations for remaining ops.
        # If 'remaining_work' is the sum of *minimum* durations or *average*? 
        # Usually, in these tasks, 'remaining_work' is a scalar representing the total 
        # work left. We will use it as the primary priority.
        
        job_priority = op.get("remaining_work", 0)
        
        # 2. Machine Selection: Shortest Processing Time (SPT) for this specific op
        min_duration = float('inf')
        chosen_machine = None
        for m, dur in options.items():
            if dur < min_duration:
                min_duration = dur
                chosen_machine = m

        if chosen_machine is None:
            continue

        # Tie-breaking: If job priorities are equal, prefer the operation with the 
        # shortest processing time (SPT) to keep the system responsive.
        
        # We want to MAXIMIZE job_priority (longest remaining work)
        # and MINIMIZE min_duration (shortest op time)
        
        # Combine into a single score for comparison:
        # Primary: higher job_priority is better
        # Secondary: lower min_duration is better
        
        # Let's just iterate and keep the best based on tuple comparison
        # (job_priority, -min_duration) -> we want max job_priority, min min_duration
        # So we compare: if (job_priority > best_priority) or (job_priority == best_priority and min_duration < best_spt)
        
        if job_priority > best_priority or (job_priority == best_priority and min_duration < best_spt):
            best_priority = job_priority
            best_spt = min_duration
            best_index = i
            best_machine = chosen_machine

    # Fallback if no valid operation was found (should not happen)
    if best_machine is None:
        return 0

    return (int(best_index), best_machine)

# EVOLVE-BLOCK-END
