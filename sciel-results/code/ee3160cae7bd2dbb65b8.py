"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module


def dispatch(ready_operations, time):
    """
    Dispatches the next operation to start in an open-shop scheduling simulation.
    
    Args:
        ready_operations: List of dicts, each representing a ready operation.
        time: Current simulation time.
        
    Returns:
        Index of the operation to start.
    """
    if not ready_operations:
        return 0

    # Extract metrics
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    
    # Calculate inverse of minimum duration for each operation
    # min_duration is the shortest time to complete this op on any available machine
    min_durations = []
    for op in ready_operations:
        durations = list(op["machine_options"].values())
        if durations:
            min_durations.append(1.0 / max(min(durations), 1e-9))
        else:
            min_durations.append(0.0)
    min_durations = np.array(min_durations)

    # Normalize components to prevent scale dominance
    # Using simple normalization: (x - min) / (max - min + eps)
    def normalize(arr):
        if len(arr) == 0:
            return arr
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        if range_val < 1e-9:
            return np.zeros_like(arr)
        return (arr - min_val) / range_val

    nw = normalize(remaining_works)
    no = normalize(remaining_ops)
    ns = normalize(min_durations)

    # Weighted composite
    # Previous best: 0.7, 0.2, 0.1
    # Hypothesis: Shifting weight from work to speed/ops helps reduce idle time
    weights = (0.6, 0.2, 0.2)
    
    scores = weights[0] * nw + weights[1] * no + weights[2] * ns
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
