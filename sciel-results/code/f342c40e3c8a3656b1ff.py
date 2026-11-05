"""Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard: evaluator kills after 150s. We keep a generous margin.
_DEADLINE_SECONDS = 140.0
_start_time = _time.monotonic()


def _elapsed():
    return _time.monotonic() - _start_time


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Critical-path-based greedy dispatch.
    For each ready operation, estimate the 'critical path length' if we schedule it now:
      - The operation's duration (choose min over machine_options for a lower bound, 
        or the actual chosen machine's duration)
      - Plus the sum of minimum durations of all remaining operations in the same job.
    
    We want to prioritize operations that have the largest downstream critical path,
    because delaying them increases the makespan. So we select the operation that
    MAXIMIZES (critical_path_estimate), which is equivalent to minimizing the slack.
    
    Additionally, we incorporate a machine-load consideration: if an operation can be
    done on a less-loaded machine, that's better. But since we don't have machine load
    info directly, we use the minimum duration as a proxy for 'ease of scheduling'.
    
    Score for each ready op:
      critical_path = min_duration(op) + sum(min_duration for all remaining_ops in job)
    
    We pick the op with the HIGHEST critical_path (most urgent / longest downstream chain).
    
    Tie-breaking: prefer the op with smaller remaining_work (less total work left),
    then smaller op index for determinism.
    """
    if not ready_operations:
        return 0

    # Safety: if we're close to deadline, just return first
    if _elapsed() > _DEADLINE_SECONDS - 5.0:
        return 0

    n = len(ready_operations)
    
    # Precompute for each ready operation:
    # 1. min_duration of the operation itself
    # 2. sum of min durations of all remaining ops in the job (including this one)
    # 3. remaining_work for tie-breaking
    
    best_idx = 0
    best_score = -1.0
    best_remaining_work = float('inf')
    best_op_idx = float('inf')
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        
        # Minimum duration for this operation
        if machine_options:
            min_dur = min(machine_options.values())
        else:
            min_dur = 0.0
        
        # Sum of minimum durations for all remaining operations in this job
        # remaining_ops is the list of ops still to be done (including this one?)
        # Typically remaining_ops includes the current op and all successors
        remaining_ops = op.get("remaining_ops", [])
        
        # Estimate critical path: sum of min durations of all remaining ops
        # This is a lower bound on the time needed to finish this job
        critical_path = 0.0
        for ro in remaining_ops:
            mo = ro.get("machine_options", {})
            if mo:
                critical_path += min(mo.values())
            else:
                critical_path += 0.0
        
        # The operation's own min duration is already included in critical_path
        # if remaining_ops includes it. Let's be safe: critical_path is the
        # sum over all remaining ops, which should include the current one.
        
        # Score: higher critical path means more urgent
        score = critical_path
        
        # Tie-breaking: prefer smaller remaining_work, then smaller op index
        rw = op.get("remaining_work", 0)
        op_idx = op.get("op", i)
        
        if (score > best_score or
            (score == best_score and rw < best_remaining_work) or
            (score == best_score and rw == best_remaining_work and op_idx < best_op_idx)):
            best_score = score
            best_remaining_work = rw
            best_op_idx = op_idx
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
