"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard to ensure we finish well before the 120s limit
_start_time = time.time()
_DEADLINE = _start_time + 115.0  # 5s safety margin

def dispatch(ready_operations, time):
    """
    Dispatches the next ready operation based on a bottleneck-elimination strategy.
    
    Strategy:
    1. Estimate the critical path length for each ready operation as 
       (remaining_ops * min_duration) + current_duration.
    2. Select the operation with the maximum estimated critical path length.
    3. Assign it to the machine with the lowest estimated load (approximated by min duration).
    
    Returns:
        (index, machine) tuple.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to simple heuristic if we're running out of time
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0

    n_ops = len(ready_operations)
    
    # Precompute min durations and remaining ops for each operation
    min_durations = []
    remaining_ops_list = []
    best_machines = []
    
    for op in ready_operations:
        mach_opts = op["machine_options"]
        # Find machine with minimum duration (best case for this op)
        min_dur = min(mach_opts.values())
        min_mach = min(mach_opts, key=mach_opts.get)
        
        min_durations.append(min_dur)
        remaining_ops_list.append(op["remaining_ops"])
        best_machines.append(min_mach)
        
        # Note: remaining_work is also available, but we use remaining_ops * min_dur
        # as a proxy for critical path contribution if we assume all remaining ops
        # have similar min durations. remaining_work is sum of durations, which is 
        # more accurate for total work, but critical path is about the longest chain.
        # Using remaining_ops * min_dur is a lower bound on remaining critical path.
        # However, remaining_work / max_duration might be better? 
        # Let's stick to remaining_ops * min_dur as per directive hint, 
        # but refine: Critical path is driven by the longest sequence.
        # A better estimate for "urgency" is remaining_work / max_parallelism? 
        # No, let's use the directive's suggestion: remaining_ops * min_duration.
        # Actually, remaining_work is the sum of durations. 
        # If we assume the remaining ops are on different machines, the critical path 
        # is roughly remaining_work / max_parallelism? No, it's the longest chain.
        # Let's use remaining_work as the primary metric but weight it by the 
        # "bottleneck" factor.
        
    # Directive says: "estimated as remaining_ops * min_duration"
    # Let's calculate scores based on that.
    # score_i = (remaining_ops_i * min_duration_i) + min_duration_i (current op)
    # This prioritizes jobs that have long chains of short operations (high count) 
    # or jobs with a single very long operation?
    # Actually, if min_duration is small, the product is small. 
    # If a job has 10 ops of 1 unit, score = 10*1 + 1 = 11.
    # If a job has 1 op of 10 units, score = 1*10 + 10 = 20.
    # This correctly identifies the single long op as more critical for makespan 
    # if it's on the critical path.
    
    critical_path_estimates = []
    for i in range(n_ops):
        cp = remaining_ops_list[i] * min_durations[i] + min_durations[i]
        critical_path_estimates.append(cp)
        
    # Find the operation with the max critical path estimate
    max_cp_idx = int(np.argmax(critical_path_estimates))
    
    # Now, select the machine for this operation.
    # Strategy: Assign to the machine with the lowest current load.
    # We don't have direct load info, but we can approximate.
    # The directive says: "assigns it to the machine with the lowest current load 
    # (approximated by the minimum duration among available options)"
    # Wait, the min duration is the same for all machines? No, min duration is per machine.
    # The machine with the *minimum duration* for this operation is the "best" machine 
    # in terms of processing time. 
    # However, "lowest current load" is not directly available. 
    # In a simple heuristic without state, we often pick the machine that minimizes 
    # the completion time of this operation, which is just min duration.
    # But if we want to balance load, we might pick a machine that is *not* the 
    # most popular? We don't have popularity.
    # Let's assume the machine with the min duration is the best choice for this op.
    # The directive says "approximated by the minimum duration among available options".
    # This implies we pick the machine that gives the min duration for this op.
    
    selected_machine = best_machines[max_cp_idx]
    
    return (max_cp_idx, selected_machine)

# EVOLVE-BLOCK-END
