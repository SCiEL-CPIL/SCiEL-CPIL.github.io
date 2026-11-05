"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard
_DEADLINE_START = _time.time()
_DEADLINE_LIMIT = 135.0  # safety margin before 150s kill


def _check_deadline():
    return (_time.time() - _DEADLINE_START) > _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one option, return it
    if n == 1:
        return 0
    
    # Deadline check - if we're close to timeout, fall back to simple rule
    if _check_deadline():
        # Simple MWR fallback
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # Get minimum and average duration for each operation
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=float)
    
    avg_durations = np.array([
        np.mean(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=float)
    
    # Count machine options (fewer options = more constrained)
    option_counts = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=float)
    
    # Normalize features to [0, 1]
    def normalize(arr):
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-9:
            return np.ones_like(arr) * 0.5
        return (arr - min_val) / (max_val - min_val)
    
    norm_work = normalize(remaining_works)
    norm_ops = normalize(remaining_ops)
    norm_dur = normalize(min_durations)
    norm_avg_dur = normalize(avg_durations)
    norm_options = normalize(option_counts)
    
    # Look-ahead: estimate critical path reduction for each operation
    # The idea is to simulate completing each operation and see how much
    # the maximum remaining work across all jobs decreases
    
    # For each ready operation, estimate the reduction in makespan lower bound
    # Makespan lower bound ≈ max over all jobs of (remaining_work)
    # Completing an operation reduces that job's remaining_work by its duration
    
    # We need to track which job each operation belongs to
    job_ids = np.array([op["job"] for op in ready_operations], dtype=int)
    
    # Estimate the makespan reduction for each operation
    # If we complete operation i, job job_ids[i] loses min_durations[i] of work
    # The reduction in makespan is limited by how much that job contributes to the max
    
    # For a more accurate estimate, we consider:
    # new_remaining_work[i] = remaining_works[i] - min_durations[i]
    # The makespan lower bound is approximately max(remaining_works)
    # After completing op i, the new max might be lower if job_ids[i] had the max
    
    # Compute the reduction in the maximum remaining work
    # This is a heuristic: we assume that reducing the work of the job with the most
    # remaining work helps the most
    
    # For each operation, compute the estimated reduction in makespan
    # This is the difference between current max remaining work and the new max
    # after removing this operation's duration from its job
    
    # Group operations by job to understand job-level remaining work
    # But we only have ready operations, not all jobs' remaining work
    
    # Simpler approach: use a weighted combination that emphasizes:
    # 1. Operations on jobs with high remaining work (critical path)
    # 2. Short processing times (SPT to reduce machine contention)
    # 3. Fewer remaining operations (urgency)
    
    # Non-linear urgency × blocking composite score
    # Urgency: how critical is this operation (high remaining work, few remaining ops)
    # Blocking: how much this operation blocks others (long duration, many options)
    
    # Compute urgency score: high remaining work and few remaining ops
    urgency = norm_work * (1.0 - 0.5 * norm_ops)
    
    # Compute blocking score: long durations and few options are more blocking
    # We want to prioritize operations that are NOT blocking (short, flexible)
    blocking = norm_dur * 0.6 + norm_avg_dur * 0.4
    
    # Composite score: urgency × (1 - blocking) favors urgent, non-blocking operations
    composite = urgency * (1.0 - 0.3 * blocking)
    
    # Add a small SPT bonus for short operations
    spt_bonus = (1.0 - norm_dur) * 0.1
    
    # Add constraint bonus for operations with fewer machine options
    constraint_bonus = (1.0 - norm_options) * 0.05
    
    # Final score
    score = composite + spt_bonus + constraint_bonus
    
    # Add small tie-breaker based on job number for determinism
    job_ids_float = job_ids.astype(float)
    score += 1e-6 * (1.0 - normalize(job_ids_float))
    
    return int(np.argmax(score))
# EVOLVE-BLOCK-END
