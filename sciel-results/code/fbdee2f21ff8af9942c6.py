"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven job-shop simulation.
    
    Uses a composite priority rule that considers:
    - Remaining work (longer jobs get priority)
    - Remaining operations count (jobs closer to completion)
    - Minimum available duration on machines (prefer shorter processing times)
    - Machine load balancing
    
    Returns: index of the ready operation to start
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = np.zeros(n)
    
    # Feature 1: remaining_work (higher = more work left, prioritize)
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    max_rw = remaining_works.max() if max(remaining_works) > 0 else 1.0
    scores += 0.3 * (remaining_works / max_rw)
    
    # Feature 2: remaining_ops (more ops remaining = prioritize to avoid blocking)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    max_ro = remaining_ops.max() if max(remaining_ops) > 0 else 1.0
    scores += 0.2 * (remaining_ops / max_ro)
    
    # Feature 3: minimum duration across available machines (shorter = less blocking)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else float('inf')
        for op in ready_operations
    ], dtype=np.float64)
    max_md = min_durations.max() if min_durations.max() > 0 else 1.0
    # We prefer shorter durations, so invert
    scores += 0.3 * (1.0 - (min_durations / max_md))
    
    # Feature 4: op number (later operations in a job might be more critical)
    ops = np.array([op["op"] for op in ready_operations], dtype=np.float64)
    max_op = ops.max() if max(ops) > 0 else 1.0
    scores += 0.1 * (ops / max_op)
    
    # Feature 5: Job completion urgency (fewer remaining ops = closer to done)
    # This helps complete jobs sooner, freeing up their remaining resources
    job_completion = np.array([1.0 / (op["remaining_ops"] + 1.0) for op in ready_operations], dtype=np.float64)
    scores += 0.1 * job_completion
    
    # Return index of best score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
