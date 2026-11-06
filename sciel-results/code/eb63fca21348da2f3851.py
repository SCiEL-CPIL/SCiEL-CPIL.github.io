"""Optimized dispatch rule for open-shop scheduling using LPT heuristic."""
# EVOLVE-BLOCK-START
import time
import sys

# Global state for time guard
_start_time = None
_DEADLINE = 115.0  # Safety margin before 120s limit

def _init_state():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _check_time():
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE

def dispatch(ready_operations, time):
    """
    Dispatch rule that selects which ready operation to start next.
    
    Uses the Longest Processing Time (LPT) heuristic, which prioritizes
    the job with the most remaining work. This is theoretically strong
    for open-shop scheduling to minimize makespan.
    
    Ties are broken by selecting the operation with the smallest job ID,
    then the smallest operation ID, to ensure determinism.
    """
    _init_state()
    
    if not ready_operations:
        return 0
    
    # Time guard: if we're running out of time, use the fastest possible check
    # (which is already very fast, but just in case)
    if _check_time():
        # Return the index of the first operation with max remaining_work
        max_work = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            work = op["remaining_work"]
            if work > max_work:
                max_work = work
                best_idx = i
        return int(best_idx)

    # LPT Heuristic: Maximize remaining_work
    # Tie-breaker: Minimize job ID, then Minimize op ID
    # We want the index i that minimizes the tuple (-remaining_work, job, op)
    # Or equivalently, find the max remaining_work, then among those, min job, then min op.
    
    best_idx = 0
    best_work = ready_operations[0]["remaining_work"]
    best_job = ready_operations[0]["job"]
    best_op = ready_operations[0]["op"]
    
    for i in range(1, len(ready_operations)):
        op = ready_operations[i]
        work = op["remaining_work"]
        job = op["job"]
        op_id = op["op"]
        
        # Primary: Maximize work
        if work > best_work:
            best_work = work
            best_job = job
            best_op = op_id
            best_idx = i
        # Secondary: Minimize job (if work is tied)
        elif work == best_work:
            if job < best_job:
                best_job = job
                best_op = op_id
                best_idx = i
            elif job == best_job and op_id < best_op:
                best_op = op_id
                best_idx = i
                
    return int(best_idx)

# EVOLVE-BLOCK-END
