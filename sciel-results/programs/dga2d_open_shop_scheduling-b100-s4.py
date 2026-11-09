"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


# Module-level deadline guard
_START_TIME = None
_TIME_LIMIT = 140.0  # 10s safety margin before 150s kill
_DEADLINE_REACHED = False


def _check_deadline():
    global _START_TIME, _DEADLINE_REACHED
    if _START_TIME is None:
        _START_TIME = _time.monotonic()
    if _time.monotonic() - _START_TIME > _TIME_LIMIT:
        _DEADLINE_REACHED = True
    return _DEADLINE_REACHED


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Use a weighted combination of:
    1. Remaining work (total) - prefer jobs with more remaining work
    2. Minimum machine option duration - prefer operations that can finish quickly
    3. Number of remaining ops - prefer jobs with fewer remaining ops (to complete jobs)
    
    The key insight for open-shop is that we want to:
    - Start operations on machines that are free (but we can't control which machine is free here)
    - Prioritize jobs with more remaining work to keep them progressing
    - But also consider the specific machine assignment to minimize idle time
    
    Since we only pick which ready operation to start (the simulation handles machine assignment),
    we should focus on which operation to dispatch. A good heuristic is to pick the operation
    that has the highest "urgency" - combining remaining work and the minimum processing time
    among its machine options (since shorter operations can be slotted in more easily).
    
    Actually, re-reading: we return an index of the ready operation to start. The simulation
    will then assign it to one of its machine_options. So we should pick the operation that
    leads to the best overall schedule.
    
    Heuristic: Score = remaining_work * 1.0 + min_machine_duration * 0.1 - remaining_ops * 0.5
    This favors jobs with lots of remaining work (to keep them busy), slightly penalizes
    long operations (since they block machines), and slightly favors jobs close to completion.
    
    Let me think more carefully. In open-shop, all jobs must visit all machines. The makespan
    is determined by the longest machine workload and the longest job workload. To minimize
    makespan, we want to balance machine loads and job completion times.
    
    A simple but effective heuristic: pick the operation with the minimum processing time
    among its available machines (shortest processing time first on the selected machine).
    But since we're choosing the operation, not the machine, let's think differently.
    
    Actually, the best approach might be to pick the operation that has the smallest minimum
    processing time (to keep machines busy with short tasks and avoid blocking), or the
    operation from the job with the most remaining work (to ensure no job falls behind).
    
    Let's use a combined score:
    - Primary: remaining_work (higher is better)
    - Secondary: -min_duration (lower min duration is better, so we negate for max)
    - Tertiary: -remaining_ops (fewer remaining ops is better)
    """
    if not ready_operations:
        return 0
    
    if _check_deadline():
        # Fallback: just return the first one
        return 0
    
    n = len(ready_operations)
    
    # Compute scores
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Get minimum duration among machine options
        machine_options = op["machine_options"]
        if isinstance(machine_options, dict):
            min_duration = min(machine_options.values()) if machine_options else 0
        else:
            min_duration = 0
        
        # Weighted score: emphasize remaining work, penalize long min duration,
        # and slightly favor jobs with fewer remaining ops
        scores[i] = (
            remaining_work * 1.0
            - min_duration * 0.3
            - remaining_ops * 0.5
        )
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
