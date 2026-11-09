"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_START = _time.time()
_MAX_WALL = 140.0  # Safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def dispatch(ready_operations, time):
    # Check deadline first; if we're over, pick the first available (index 0)
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each op, find the minimum duration across its machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    # Number of machine options available (urgency: fewer options = should schedule sooner)
    num_machines_available = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)

    # Compute score: we want to MAXIMIZE this to pick the best op
    # 
    # Primary signal: remaining_work (LRPT) - jobs with more remaining work should go first
    # Secondary: remaining_ops - jobs with more operations left
    # Tertiary: urgency - operations with fewer machine options should be scheduled sooner
    # Tie-breaker: shorter min_duration to reduce makespan contribution
    
    # Use raw values scaled appropriately to avoid normalization artifacts
    # remaining_work is typically in the range of total job processing time
    # remaining_ops is small integer
    # num_machines_available is small integer
    
    # Weighted sum with emphasis on remaining work
    # Invert num_machines_available: fewer options = higher urgency = higher score
    # Use 1/(num_machines + 1) as urgency factor (higher when fewer machines)
    urgency = 1.0 / (num_machines_available + 1.0)
    
    # Normalize urgency to [0, 1] for consistent weighting
    u_min = np.min(urgency)
    u_max = np.max(urgency)
    if u_max - u_min < 1e-12:
        urgency_norm = np.zeros_like(urgency)
    else:
        urgency_norm = (urgency - u_min) / (u_max - u_min)
    
    # For min_duration, lower is better (shorter processing = less makespan impact)
    # Invert: higher score for shorter durations
    md_min = np.min(min_durations)
    md_max = np.max(min_durations)
    if md_max - md_min < 1e-12:
        md_inv = np.zeros_like(min_durations)
    else:
        md_inv = 1.0 - (min_durations - md_min) / (md_max - md_min)
    
    # Scale remaining_work and remaining_ops to comparable ranges
    # remaining_work is the dominant factor
    rw_max = np.max(remaining_works)
    if rw_max < 1e-12:
        rw_norm = np.zeros_like(remaining_works)
    else:
        rw_norm = remaining_works / rw_max
    
    ro_max = np.max(remaining_ops)
    if ro_max < 1e-12:
        ro_norm = np.zeros_like(remaining_ops)
    else:
        ro_norm = remaining_ops / ro_max
    
    # Composite score
    # LRPT (remaining_work) is the strongest predictor for makespan
    # Urgency (fewer machine options) is important to avoid blocking
    # Remaining ops provides secondary priority
    # Shorter duration helps as a tie-breaker
    score = 1.0 * rw_norm + 0.4 * urgency_norm + 0.3 * ro_norm + 0.1 * md_inv

    # Pick the argmax
    best_idx = int(np.argmax(score))

    return best_idx
# EVOLVE-BLOCK-END
