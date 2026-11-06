"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = None
_START = None

def _ensure_deadline():
    global _DEADLINE, _START
    if _DEADLINE is None:
        _START = _time.monotonic()
        _DEADLINE = _START + 145.0  # safety margin before 150s kill
    return _DEADLINE

def _time_left():
    return max(0.0, _DEADLINE - _time.monotonic())


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Strategy: Multi-criteria weighted heuristic balancing:
    1. Remaining work (longest job first to reduce tail)
    2. Machine scarcity (prefer operations on machines with fewer alternatives)
    3. Processing time on the chosen machine (longer first to avoid short ops blocking)
    
    Returns: index of the ready operation to start
    """
    deadline = _ensure_deadline()
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Fast path: if we're close to deadline, use simple heuristic
    if _time_left() < 0.5:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        # 1. Remaining work of the job (higher = more urgent to finish)
        rem_work = op["remaining_work"]
        
        # 2. Machine scarcity: how many machine options does this op have?
        #    Fewer options = more constrained = higher priority
        num_options = len(op["machine_options"])
        scarcity = 1.0 / max(num_options, 1)
        
        # 3. Average processing time across machine options
        durations = list(op["machine_options"].values())
        avg_duration = sum(durations) / len(durations)
        
        # 4. Minimum processing time (best case duration)
        min_duration = min(durations)
        
        # 5. Number of remaining operations for this job
        rem_ops = op["remaining_ops"]
        
        # Composite score:
        # - Weight remaining work heavily (dominant factor for makespan)
        # - Add scarcity bonus (constrained operations should go first)
        # - Add a small bonus for longer average duration (longer jobs first)
        # - Penalize if many remaining ops (spread work out)
        
        score = (
            1.0 * rem_work +
            0.3 * scarcity * rem_work +
            0.1 * avg_duration +
            0.05 * min_duration -
            0.02 * rem_ops
        )
        scores[i] = score
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
