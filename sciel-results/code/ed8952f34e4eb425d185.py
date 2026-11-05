"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard: 140s is a safe margin before the 150s kill limit.
_START_TIME = _time.time()
_DEADLINE = 140.0

def dispatch(ready_operations, time):
    """
    Dispatch rule: Minimum Projected Completion Time (MPCT).
    
    Strategy:
    1. For each ready operation, determine the machine that would result in the
       earliest completion time if the operation were started now.
    2. Select the operation with the minimum projected completion time.
    
    Rationale:
    - Prioritizing operations that can finish quickly reduces the risk of
      blocking other operations that might be waiting on the same machine.
    - By considering the current machine load (implicitly via the assumption
      that the machine is free at 'time' for the dispatch decision, or more
      accurately, just comparing durations since the scheduler calls dispatch
      when a machine is free), we effectively use a Shortest Processing Time
      (SPT) heuristic but adapted to the specific machine options.
    - In a job-shop where machine availability is the bottleneck, picking the
      job that finishes soonest on the currently available machine often leads
      to better utilization than picking based on total remaining work.
    
    Note: Since the dispatch function is called when a machine becomes free or
    at the start, and we don't have explicit machine state in the interface
    (only ready ops), we assume the "time" parameter is the current simulation
    time. The most robust simple heuristic without machine state is to pick
    the operation with the minimum duration on its best available machine.
    """
    # Check deadline
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    best_idx = 0
    best_score = float('inf')

    for i, op in enumerate(ready_operations):
        # Find the minimum duration among available machines for this op
        # This acts as a proxy for "which machine is least loaded/most capable"
        # assuming all machines are equivalent in speed but differ in current load
        # which we don't know. However, SPT on the available options is a strong
        # baseline.
        
        min_duration = float('inf')
        for machine, duration in op["machine_options"].items():
            if duration < min_duration:
                min_duration = duration
        
        # Tie-breaker: If durations are equal, prefer the job with less remaining work
        # to clear the pipeline faster, or more remaining work?
        # Usually, SPT is the primary driver. Let's stick to min_duration.
        # If min_duration is inf (no options?), skip or handle.
        
        if min_duration < best_score:
            best_score = min_duration
            best_idx = i
        elif min_duration == best_score:
            # Tie-breaker: Pick the job with the most remaining work to ensure
            # critical path jobs aren't starved? Or least?
            # Let's try: if tie, pick the one with *more* remaining work to keep
            # critical jobs moving.
            if op["remaining_work"] > ready_operations[best_idx]["remaining_work"]:
                best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
