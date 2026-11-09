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
        # Return index and the best machine for the single op
        op = ready_operations[0]
        if op["machine_options"]:
            best_machine = min(op["machine_options"], key=op["machine_options"].get)
            return (0, best_machine)
        return 0

    # Extract features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each op, compute soft-min duration via log-sum-exp (smooth surrogate for min)
    # This captures the "best case" processing time in a differentiable way
    # Also track the machine that gives the hard min for actual assignment
    n_machines_max = max(len(op["machine_options"]) for op in ready_operations) if ready_operations else 0

    # Build a matrix of durations (padded with +inf for missing entries)
    # Shape: (n, max_machines)
    dur_matrix = np.full((n, n_machines_max), np.inf, dtype=np.float64)
    best_machines = [None] * n
    min_durations = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            # Get all durations
            durations = np.array(list(mo.values()), dtype=np.float64)
            mkeys = list(mo.keys())
            j = len(durations)
            dur_matrix[i, :j] = durations
            # Hard min for actual machine assignment
            min_idx = np.argmin(durations)
            best_machines[i] = mkeys[min_idx]
            min_durations[i] = durations[min_idx]
            # Soft-min via log-sum-exp: -log(sum(exp(-d/T))) approximates min for small T
            # Use T=1.0 for a smooth approximation
            T = 1.0
            exp_neg = np.exp(-durations / T)
            soft_min = -T * np.log(np.sum(exp_neg))
            min_durations[i] = soft_min  # Override with soft-min for scoring

    # Number of machine options available
    num_machines_available = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)

    # Composite score: we want to MAXIMIZE this to pick the best op
    # Strategy:
    # 1. High remaining_work is good (LRPT) - Strongest signal for makespan
    # 2. High remaining_ops is good
    # 3. Low min_duration is good (frees machine sooner)
    # 4. More machine options means more flexibility
    
    # Normalize features to [0, 1] range for stable weighting
    def _norm(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    nw = _norm(remaining_works)
    no = _norm(remaining_ops)
    # Invert min_duration: lower duration is better
    md_inv = 1.0 - _norm(min_durations)
    nm = _norm(num_machines_available)

    # Adjusted weights: Emphasize Remaining Work most strongly
    # Use soft-min based duration for smoother scoring
    score = 2.5 * nw + 0.6 * no + 0.3 * md_inv + 0.15 * nm

    # Pick the argmax
    best_idx = int(np.argmax(score))

    # Return the index and the best machine (SPT) for that operation
    # This ensures we don't waste time on a longer machine if a shorter one is available
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END
