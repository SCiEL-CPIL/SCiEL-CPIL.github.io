"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: kill if we exceed 145s of wall-clock (5s safety margin)
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0


def dispatch(ready_operations, time):
    # If we're near the deadline, fall back to simple rule
    if _time.time() > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Score each ready operation using a composite heuristic:
    # 1. Most remaining work (higher is better)
    # 2. Fewest machine options (more constrained first)
    # 3. Shortest processing time on best machine (to reduce idle time)
    # 4. Fewest remaining ops (to finish jobs)
    
    scores = []
    for op in ready_operations:
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Number of machine options (fewer = more constrained)
        num_machines = len(machine_options)
        
        # Best (shortest) processing time among options
        best_pt = min(machine_options.values())
        
        # Composite score:
        # - remaining_work: higher is better (more urgent)
        # - 1/num_machines: higher when fewer options (more constrained)
        # - 1/best_pt: higher when shorter processing (reduces makespan)
        # - 1/max(remaining_ops, 1): higher when fewer ops left
        
        score = (
            remaining_work * 1.0 +
            (1.0 / max(num_machines, 1)) * 5.0 +
            (1.0 / max(best_pt, 0.01)) * 2.0 +
            (1.0 / max(remaining_ops, 1)) * 1.0
        )
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
