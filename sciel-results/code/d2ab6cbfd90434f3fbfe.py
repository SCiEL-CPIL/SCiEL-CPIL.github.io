"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = _time.time() + 140.0  # safety margin before 150s kill


def _deadline_exceeded():
    return _time.time() > _DEADLINE


def _soft_min(values, temperature=0.1):
    """Approximate min using log-sum-exp with negative values (soft-min)."""
    if len(values) == 0:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    v_min = v.min()
    exp_vals = np.exp(-(v - v_min) / temperature)
    return v_min + temperature * np.log(exp_vals.sum())


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Fast path for single op
    if n == 1:
        return 0

    # Extract key metrics
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Soft-min duration (smooth surrogate for min) and avg duration across machine options
    soft_min_durs = np.empty(n, dtype=np.float64)
    avg_durs = np.empty(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        durs = np.array(list(op["machine_options"].values()), dtype=np.float64)
        soft_min_durs[i] = _soft_min(durs, temperature=0.1)
        avg_durs[i] = np.mean(durs)

    # Normalization to [0, 1] (higher = more urgent / more efficient)
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    # Urgency: more remaining work and more remaining ops => more urgent
    urgency_work = _normalize(remaining_works)
    urgency_ops = _normalize(remaining_ops)
    urgency = 0.7 * urgency_work + 0.3 * urgency_ops

    # Efficiency: shorter operations reduce idle time; invert so shorter = higher
    eff_soft = _normalize(-soft_min_durs)  # shorter soft-min duration -> higher
    eff_avg = _normalize(-avg_durs)
    efficiency = 0.6 * eff_soft + 0.4 * eff_avg

    # Combined score: urgency-weighted with efficiency as secondary
    # Weight urgency more heavily (1.0) and efficiency less (0.4)
    scores = 1.0 * urgency + 0.4 * efficiency

    # Pick the operation with the highest combined score
    best_idx = int(np.argmax(scores))

    # Optionally choose the best machine (shortest duration) for that op
    # The interface allows returning just index or (index, machine).
    # Returning just the index is safest and matches the seed.
    return best_idx
# EVOLVE-BLOCK-END
