"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: we must finish each dispatch call quickly.
# The evaluator kills after 150s wall-clock total for all calls.
# We'll track a global start time and if we're getting close to the limit,
# fall back to a simple heuristic.
_START_TIME = _time.time()
_DEADLINE = 140.0  # safety margin of 10s


def _simple_dispatch(ready_operations, time):
    """Fallback simple dispatch: most remaining work first."""
    if not ready_operations:
        return 0
    return int(np.argmax([op["remaining_work"] for op in ready_operations]))


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Use a combination of heuristics:
    1. Prioritize operations with the most remaining work (MWL)
    2. Break ties by choosing the operation that can be done on the machine 
       with the least current load (to balance machine utilization)
    3. Consider the minimum duration among machine options (shortest processing time)
    
    The score is mean makespan, so we want to minimize the time until all jobs are done.
    Key insight: In open-shop, a job can be processed on any machine for any operation.
    We should prioritize:
    - Jobs with more remaining work (they have more to do)
    - Operations that can be done quickly (shorter processing times reduce makespan)
    - Balance across machines to avoid bottlenecks
    
    A good composite score: prioritize by remaining_work / min_duration ratio,
    which favors operations that have lots of work but can be done quickly.
    """
    if not ready_operations:
        return 0
    
    # Check deadline
    if _time.time() > _DEADLINE:
        return _simple_dispatch(ready_operations, time)
    
    n = len(ready_operations)
    
    # For each ready operation, compute a priority score
    # Higher score = higher priority
    scores = []
    for op in ready_operations:
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            scores.append(0)
            continue
        
        # Minimum duration among available machines
        min_dur = min(machine_options.values())
        # Average duration
        avg_dur = sum(machine_options.values()) / len(machine_options)
        # Number of machine options (flexibility)
        n_machines = len(machine_options)
        
        # If no remaining work specified, estimate from remaining_ops
        if remaining_work <= 0 and remaining_ops > 0:
            remaining_work = remaining_ops * avg_dur
        
        if remaining_work <= 0:
            scores.append(0)
            continue
        
        # Composite score:
        # 1. Remaining work (more work = higher priority) - normalized
        # 2. Shortest processing time (shorter = higher priority) - inverse
        # 3. Flexibility (more options = slightly better)
        
        # Primary: remaining work
        # Secondary: 1/min_duration (favors short operations)
        # Tertiary: flexibility bonus
        
        # Use a weighted combination
        # Weight 1: remaining work (dominant)
        # Weight 2: 1/min_dur (secondary)
        # Weight 3: n_machines (tertiary, small)
        
        score = remaining_work * 1.0 + (1.0 / max(min_dur, 0.01)) * 0.5 + n_machines * 0.1
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
