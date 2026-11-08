"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Pre-computed constants for fast evaluation
# We will use a multi-criteria heuristic that balances:
# 1. Remaining work (urgency to complete)
# 2. Remaining operations (flow time / critical path)
# 3. Shortest processing time (SPT) to free up machines
# 4. Machine load balance (avoid bottlenecks)

# We'll use a weighted sum with normalized features.
# To keep it fast, we precompute weights and use simple arithmetic.

# Let's define a scoring function that considers:
# - remaining_work: higher is worse (more work left)
# - remaining_ops: higher is worse (more ops left, longer flow)
# - min_duration: lower is better (SPT)
# - machine_availability: we don't have this directly, but we can estimate

# Actually, let's look at the available fields:
# job, op, machine_options {machine: duration}, remaining_ops, remaining_work

# A good heuristic for job-shop is often:
# - CR (Critical Ratio): due_date / remaining_time, but we don't have due dates
# - SPT: shortest processing time
# - WSPT: work remaining / processing time
# - FIF: first-in-first-out (not applicable here)
# - Longest remaining path: remaining_ops * avg_duration

# Let's use a combination:
# score = w1 * (remaining_work / max_remaining_work) + 
#         w2 * (remaining_ops / max_remaining_ops) + 
#         w3 * (min_duration / max_min_duration)
# We want to MINIMIZE this score, so we pick the op with the lowest score.

# But wait, remaining_work and remaining_ops are "remaining" so higher means more work left.
# For SPT, lower duration is better.
# So we want to prioritize ops with:
# - Lower remaining_work? No, that's not right. We want to finish jobs, so maybe higher remaining_work is more urgent? 
#   Actually, in job-shop, we want to minimize makespan. A common approach is to prioritize the operation that is on the critical path.
# - The critical path is related to remaining_ops and the durations.

# Let's try a different approach: 
# 1. Calculate the "urgency" as remaining_work / remaining_ops (average work per op)
# 2. Calculate the "speed" as 1 / min_duration (higher is better)
# 3. Combine them

# Actually, let's look at the seed program: it uses argmax of remaining_work.
# The knowledge says: "Normalized feature scaling prevents WSPT ratio from dominating... the shift from urgency-to-complete to keep-in-motion (more remaining ops) aligns with higher-scoring inspirations"

# So let's use:
# - remaining_ops: higher is better (keep in motion)
# - remaining_work: higher is better (urgency)
# - min_duration: lower is better (SPT)

# Let's normalize each feature across the ready_operations and combine them.

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, find the minimum duration among its machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 1.0
        for op in ready_operations
    ], dtype=np.float64)

    # Normalize features
    nw = _normalize(remaining_work)
    no = _normalize(remaining_ops)
    nd = _normalize(min_durations)

    # We want to MAXIMIZE a score that favors:
    # - Higher remaining_work (urgency)
    # - Higher remaining_ops (keep in motion)
    # - Lower min_duration (SPT) -> so we use 1 - nd

    # Let's use a weighted sum. The weights can be tuned.
    # Based on the knowledge, remaining_ops is important.
    w_work = 0.3
    w_ops = 0.4
    w_spt = 0.3

    score = w_work * nw + w_ops * no + w_spt * (1.0 - nd)

    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
