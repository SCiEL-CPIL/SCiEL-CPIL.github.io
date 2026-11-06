"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    start_time = time_module.perf_counter()
    deadline = start_time + 1.0  # Safety margin: leave ample time within 150s budget
    
    n = len(ready_operations)
    
    # Extract features for each ready operation
    remaining_work = np.array([op["remaining_work"] for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    
    # For each operation, get the minimum duration among machine options
    min_durations = np.array([min(op["machine_options"].values()) for op in ready_operations])
    max_durations = np.array([max(op["machine_options"].values()) for op in ready_operations])
    
    # Heuristic score: prioritize operations with:
    # 1. Most remaining work (to finish jobs with lots of work)
    # 2. Fewest remaining operations (to complete jobs soon)
    # 3. Shorter processing times (to reduce machine load)
    
    # Normalize features
    rw_max = np.max(remaining_work) if np.max(remaining_work) > 0 else 1
    ro_max = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1
    md_max = np.max(min_durations) if np.max(min_durations) > 0 else 1
    
    rw_norm = remaining_work / rw_max
    ro_norm = remaining_ops / ro_max
    md_norm = min_durations / md_max
    
    # Weighted combination: prioritize work remaining, then ops remaining, then shorter duration
    scores = 0.6 * rw_norm + 0.3 * ro_norm - 0.1 * md_norm
    
    # Check if we're approaching deadline
    elapsed = time_module.perf_counter() - start_time
    if elapsed > deadline * 0.8:
        # Return best score found so far
        return int(np.argmax(scores))
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
