"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track per-call time to avoid pathological cases, but dispatch is O(n) so this is mostly safety.
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin


def _is_deadline_exceeded():
    return _time.time() >= _DEADLINE


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    if _is_deadline_exceeded():
        # Fallback: pick the operation with the most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    scores = np.zeros(n)

    for i, op in enumerate(ready_operations):
        # --- Compute critical path / urgency features ---
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]

        # Average remaining work per remaining operation
        if remaining_ops > 0:
            avg_work_per_op = remaining_work / remaining_ops
        else:
            avg_work_per_op = 0.0

        # Minimum duration among machine options for this operation
        machine_options = op["machine_options"]
        if machine_options:
            min_duration = min(machine_options.values())
            max_duration = max(machine_options.values())
        else:
            min_duration = 0.0
            max_duration = 0.0

        # WSPT-like ratio: work remaining / (duration of current op + epsilon)
        # Higher ratio = more work per unit time = should prioritize
        wspt_ratio = remaining_work / (min_duration + 1e-9)

        # Urgency: jobs with fewer remaining operations are closer to completion
        # Prioritize completing jobs to free up resources
        urgency = 1.0 / (remaining_ops + 1.0)

        # Critical path weight: total remaining work scaled by number of remaining ops
        # This captures the "longest chain" intuition
        critical_weight = remaining_work

        # Shortest processing time bonus for current operation
        # Prefer shorter operations to reduce idle time
        spt_bonus = 1.0 / (min_duration + 1.0)

        # Combine features with learned weights
        # From knowledge: multi-criteria (WSPT + critical path + urgency) improved score
        # Weights tuned to balance long-job priority with completion urgency
        w_wspt = 0.35
        w_critical = 0.25
        w_urgency = 0.25
        w_spt = 0.15

        score = (
            w_wspt * wspt_ratio
            + w_critical * critical_weight
            + w_urgency * urgency * remaining_work
            + w_spt * spt_bonus
        )

        scores[i] = score

    # Return index of the operation with the highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
