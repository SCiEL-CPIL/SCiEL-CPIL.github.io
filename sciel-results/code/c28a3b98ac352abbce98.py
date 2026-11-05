"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin


def _deadline_exceeded():
    return _time.time() >= _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: A hybrid ratio-based heuristic that combines:
    1. WSPT core: min_duration / (remaining_work + 1) -- primary signal
    2. Depth urgency: remaining_ops to avoid starving deep jobs
    3. Normalization to make features comparable across instances
    
    The WSPT ratio (inspiration) is the strongest single signal: it favors
    short ops on jobs with lots of work left. We refine it with depth
    normalization (current program) to break ties and handle edge cases
    where remaining_work is small.
    """
    if _deadline_exceeded():
        # Fallback: just pick the first one
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Precompute features for each ready operation
    scores = np.empty(n)
    
    # First pass: compute raw features
    raw_work = np.empty(n)
    raw_depth = np.empty(n)
    raw_min_dur = np.empty(n)
    raw_ratio = np.empty(n)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None or remaining_work < 0:
            remaining_work = 0.0
        remaining_ops = op.get("remaining_ops", 0)
        if remaining_ops is None or remaining_ops < 0:
            remaining_ops = 0
        machine_options = op.get("machine_options", {})
        
        if machine_options:
            durations = list(machine_options.values())
            min_duration = min(durations) if durations else 1.0
        else:
            min_duration = 1.0
        
        raw_work[i] = remaining_work
        raw_depth[i] = remaining_ops
        raw_min_dur[i] = min_duration
        # Core WSPT ratio: lower is better
        raw_ratio[i] = min_duration / (remaining_work + 1.0)
    
    # Normalize the ratio to [0, 1] (inverted so higher = better)
    ratio_min = raw_ratio.min()
    ratio_max = raw_ratio.max()
    ratio_range = max(ratio_max - ratio_min, 1e-6)
    
    # Normalize depth to [0, 1]
    depth_min = raw_depth.min()
    depth_max = raw_depth.max()
    depth_range = max(depth_max - depth_min, 1e-6)
    
    # Normalize work to [0, 1]
    work_min = raw_work.min()
    work_max = raw_work.max()
    work_range = max(work_max - work_min, 1e-6)
    
    for i in range(n):
        # Inverted normalized ratio: higher = shorter relative to work = better
        ratio_score = 1.0 - (raw_ratio[i] - ratio_min) / ratio_range
        
        # Normalized depth: higher = deeper job = prioritize to avoid starvation
        depth_score = (raw_depth[i] - depth_min) / depth_range
        
        # Normalized work: higher = more work left = prioritize
        work_score = (raw_work[i] - work_min) / work_range
        
        # Combine: WSPT ratio is the dominant signal, with depth and work
        # as secondary refinements. The ratio already captures work, so
        # we keep its weight high and use depth/work mainly for tie-breaking.
        score = (
            2.0 * ratio_score +
            1.0 * depth_score +
            0.5 * work_score
        )
        
        scores[i] = score
    
    # Return the index with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
