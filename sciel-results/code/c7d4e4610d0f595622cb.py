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

    # For each op, compute a smooth surrogate for the minimum machine duration.
    # Also track the argmin machine for potential tie-breaking or future use.
    # Using log-sum-exp for soft-min: soft_min(a) = -t * log(sum(exp(-a_i/t)))
    # This is smooth and differentiable, unlike hard min.
    def _soft_min_and_argmin(vals, temp=0.5):
        if not vals:
            return 0.0, 0
        a = np.asarray(vals, dtype=np.float64)
        # soft-min value
        z = -a / temp
        zmax = np.max(z)
        s = np.log(np.sum(np.exp(z - zmax))) + zmax
        soft_val = -s * temp
        # argmin of the original values (hard argmin for machine selection)
        argmin_idx = int(np.argmin(a))
        return soft_val, argmin_idx

    min_durations = np.zeros(n, dtype=np.float64)
    best_machines = [0] * n

    for i, op in enumerate(ready_operations):
        if op["machine_options"]:
            vals = list(op["machine_options"].values())
            soft_val, argmin_idx = _soft_min_and_argmin(vals, temp=0.5)
            min_durations[i] = soft_val
            # Map back to actual machine key
            keys = list(op["machine_options"].keys())
            best_machines[i] = keys[argmin_idx]
        else:
            min_durations[i] = 0.0
            best_machines[i] = 0

    # Number of machine options available
    num_machines_available = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)

    # Composite score: we want to MAXIMIZE this to pick the best op
    # Normalize features to [0, 1] range for stable weighting
    def _norm(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    nw = _norm(remaining_works)
    no = _norm(remaining_ops)
    # Invert min_duration: lower duration is better, so we want high score for low duration
    md_inv = 1.0 - _norm(min_durations)
    nm = _norm(num_machines_available)

    # Weights: prioritize remaining work (LRPT) as the dominant signal, then
    # remaining ops, with a moderate local machine-fit term and a small flexibility term.
    # The soft-min term is now smoother, which may help in edge cases where hard min
    # causes discontinuities in the scoring landscape.
    score = 1.2 * nw + 0.4 * no + 0.35 * md_inv + 0.05 * nm

    # Pick the argmax
    best_idx = int(np.argmax(score))

    return best_idx
# EVOLVE-BLOCK-END
