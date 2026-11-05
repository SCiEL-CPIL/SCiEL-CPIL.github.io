"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Uses a composite heuristic:
    - Primary: remaining work (critical path priority)
    - Secondary: prefer shorter processing times to reduce machine blocking
    - Tertiary: slight preference for operations with fewer machine options (constrained first)
    - Quaternary: prefer operations with fewer remaining ops (closer to completion)
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    scores = []
    for op in ready_operations:
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        remaining_ops = op["remaining_ops"]
        
        # Find the minimum processing time across available machines
        min_duration = min(machine_options.values()) if machine_options else 1
        
        # Number of machine options (fewer = more constrained)
        num_options = len(machine_options)
        
        # Composite score:
        # - High remaining work is good (priority to critical path)
        # - Shorter min duration is good (frees machine sooner)
        # - Fewer machine options is slightly better (constrained operations first)
        # - Fewer remaining ops is slightly better (closer to completion)
        # 
        # Weight: remaining work dominates, but we normalize to avoid scale issues
        score = remaining_work * 1.0
        score -= min_duration * 0.1  # Small penalty for long processing
        score += (5 - num_options) * 0.05  # Small bonus for constrained ops
        score += (10 - remaining_ops) * 0.02  # Small bonus for jobs closer to completion
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
