"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random
import math
import heapq

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0  # Safety margin

def _compute_critical_path(op, current_time):
    """
    Compute the critical path length for an operation.
    Critical path = min duration of current op + sum of min durations of remaining ops.
    This represents the earliest possible completion time if we start this op now
    and process all remaining ops on their fastest machines.
    """
    mo = op.get("machine_options", {})
    rw = op.get("remaining_work", 0)
    remaining_ops = op.get("remaining_ops", 0)
    
    if mo:
        durations = list(mo.values())
        min_d = min(durations) if durations else 0
    else:
        min_d = 0
    
    # Critical path estimate:
    # 1. Current operation's minimum duration
    # 2. Plus remaining work (which approximates sum of remaining op durations)
    # We use remaining_work as a proxy for the total remaining processing time
    critical_path = min_d + rw
    
    return critical_path

def _select_machine(op, current_time):
    """
    Select the machine that minimizes the completion time for this operation.
    Returns the machine with the shortest duration.
    """
    mo = op.get("machine_options", {})
    if not mo:
        return None
    
    # Find machine with minimum duration
    best_machine = None
    min_duration = float('inf')
    
    for machine, duration in mo.items():
        if duration < min_duration:
            min_duration = duration
            best_machine = machine
    
    return best_machine

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: MWR
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    if not ready_operations:
        return 0
    
    # If only one op, return it
    if len(ready_operations) == 1:
        return 0
    
    # Critical path-based dispatch:
    # Prioritize operations with the highest critical path length.
    # This prevents idle time caused by waiting for subsequent critical operations.
    # Combined with machine selection that minimizes completion time to balance load.
    
    best_score = -1
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        # Compute critical path for this operation
        cp = _compute_critical_path(op, time)
        
        # Get the machine that would minimize completion time
        machine = _select_machine(op, time)
        
        # Score: primarily driven by critical path (higher is better)
        # Add a small bonus for operations that can be completed quickly on their best machine
        # This helps break ties and slightly favors faster completions
        mo = op.get("machine_options", {})
        if mo:
            min_d = min(mo.values())
        else:
            min_d = 0
        
        # Score = critical_path - small_penalty * (1 / (min_d + 1))
        # This heavily favors high critical path but slightly prefers shorter current ops
        score = cp - 0.1 * (1.0 / (min_d + 1.0))
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = machine
    
    # Return index and optionally the selected machine
    # The interface allows returning just the index or (index, machine)
    # Let's return just the index for simplicity, as the simulator can handle machine selection
    # or we could return the tuple if the simulator supports it
    return int(best_idx)

# EVOLVE-BLOCK-END
