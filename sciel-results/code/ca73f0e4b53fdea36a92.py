"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    # Deadline guard: if we ever approach the wall-clock limit, fall back to a fast rule.
    # (This is a safety net; the main logic below is O(n) and very fast.)
    start = _time.perf_counter()
    DEADLINE = 145.0  # seconds; keep margin below 150s

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Fast path: if only one ready op, return it.
    if n == 1:
        return 0

    # Build arrays for vectorized scoring.
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each ready operation, compute a smooth surrogate of the minimum machine duration
    # using log-sum-exp (soft-min). This captures the distribution of machine options.
    durations = []
    for op in ready_operations:
        opts = op["machine_options"]
        if opts:
            durations.append(np.array(list(opts.values()), dtype=np.float64))
        else:
            durations.append(np.array([1.0], dtype=np.float64))

    # Stack durations (pad to max length for vectorized log-sum-exp)
    max_len = max(len(d) for d in durations)
    dur_matrix = np.full((n, max_len), np.inf, dtype=np.float64)
    for i, d in enumerate(durations):
        dur_matrix[i, :len(d)] = d

    # Soft-min via log-sum-exp: log(sum(exp(-d))) then negate... actually:
    # soft_min = -log(sum(exp(-d_i)))
    # Use the stable form: let m = min(d), then soft_min = m - log(sum(exp(-(d_i - m))))
    # But since we want a smooth surrogate that's close to min but considers all options,
    # we can use: soft_min = -logsumexp(-d) = m - log(sum(exp(-(d - m))))
    
    # For numerical stability:
    neg_d = -dur_matrix  # shape (n, max_len)
    # Where dur_matrix is inf, neg_d is -inf, exp(-inf)=0, so log-sum-exp handles it.
    # But let's use the stable log-sum-exp:
    max_neg_d = np.max(neg_d, axis=1, keepdims=True)  # = -min(d)
    # Handle case where all are -inf (shouldn't happen)
    max_neg_d = np.where(np.isinf(max_neg_d) & (max_neg_d < 0), 0.0, max_neg_d)
    
    shifted = neg_d - max_neg_d
    shifted = np.where(np.isinf(shifted) & (shifted < 0), -1e18, shifted)  # avoid exp(-inf) issues
    sum_exp = np.sum(np.exp(shifted), axis=1)
    logsumexp_neg = max_neg_d.squeeze() + np.log(sum_exp)
    soft_min = -logsumexp_neg  # This is the soft-min of durations

    # Score: combine remaining_work (high is good) with soft_min duration (low is good)
    # We want to prioritize operations that have more work remaining AND can be done quickly.
    # Normalize to comparable scales.
    rw_max = np.max(remaining_work) if n > 0 else 1.0
    sm_max = np.max(soft_min) if n > 0 else 1.0
    rw_min = np.min(remaining_work) if n > 0 else 0.0
    sm_min = np.min(soft_min) if n > 0 else 0.0

    # Normalize remaining_work to [0, 1], higher is better
    rw_norm = (remaining_work - rw_min) / (rw_max - rw_min + 1e-9)

    # Normalize soft_min to [0, 1], lower is better -> invert
    sm_norm = (soft_min - sm_min) / (sm_max - sm_min + 1e-9)
    sm_inv = 1.0 - sm_norm

    # Weighted combination: emphasize remaining work more, but factor in machine availability
    # Also consider remaining_ops count as a tiebreaker (fewer remaining ops = closer to completion)
    ro_max = np.max(remaining_ops) if n > 0 else 1.0
    ro_min = np.min(remaining_ops) if n > 0 else 0.0
    ro_norm = (remaining_ops - ro_min) / (ro_max - ro_min + 1e-9)
    ro_inv = 1.0 - ro_norm  # fewer remaining ops is better

    # Primary: remaining work, secondary: fast machine, tertiary: fewer remaining ops
    score = 1.0 * rw_norm + 0.3 * sm_inv + 0.1 * ro_inv

    # Check deadline
    if _time.perf_counter() - start > DEADLINE - 0.5:
        # Fallback: simple argmax of remaining work
        return int(np.argmax(remaining_work))

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
