"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Hard wall-clock guard: the evaluator kills at 150s; keep a safety margin.
_DEADLINE_START = _time.monotonic()
_DEADLINE_LIMIT = 145.0  # seconds


def _deadline_exceeded():
    return (_time.monotonic() - _DEADLINE_START) > _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    # Safety guard: if we are about to exceed the wall-clock budget,
    # fall back to a cheap deterministic rule immediately.
    if _deadline_exceeded():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # --- Compute per-operation scores using bottleneck mitigation ---
    # Strategy: prioritize operations that minimize estimated completion time
    # relative to global maximum machine load, routing work to least congested resources.
    
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each ready operation, find the minimum duration across its machine options
    min_durations = np.array([
        min(op["machine_options"].values()) for op in ready_operations
    ], dtype=np.float64)

    # Estimated completion time if we start this operation now on its fastest machine
    est_completion = time + min_durations

    # Find the global maximum estimated completion time across all ready ops
    max_est_completion = est_completion.max() if n > 0 else time

    # Normalize features to [0, 1] range for stable weighting
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    nw_norm = _normalize(remaining_work)
    ro_norm = _normalize(remaining_ops)

    # Bottleneck ratio: how close this op's completion is to the global max
    # Lower ratio means this op finishes earlier relative to the bottleneck
    # We want to pick ops with LOW ratio to avoid extending the critical path
    if max_est_completion > time:
        bottleneck_ratio = est_completion / max_est_completion
    else:
        bottleneck_ratio = np.ones(n) * 0.5

    # Invert: higher score for ops that finish earlier (lower ratio)
    # Use 1 - ratio so that shorter completion times get higher scores
    early_completion_score = 1.0 - bottleneck_ratio

    # Composite score: higher is better (we pick the max)
    # Weights: emphasize remaining work (critical path), reward early completion
    w_rw = 0.50
    w_ro = 0.20
    w_early = 0.30

    scores = w_rw * nw_norm + w_ro * ro_norm + w_early * early_completion_score

    best_idx = int(np.argmax(scores))

    # Optionally choose the best machine (shortest duration) for that operation
    best_op = ready_operations[best_idx]
    best_machine = min(best_op["machine_options"], key=lambda m: best_op["machine_options"][m])

    # Return (index, machine) to let the simulator use the optimal machine
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
