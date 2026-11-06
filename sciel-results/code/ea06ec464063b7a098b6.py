"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: keep explicit wall-clock budget.
    # The evaluator kills after 150s; we want to return well before that.
    _start = _time.perf_counter()
    _deadline = _start + 115.0  # 115s safety margin

    best_score = -1.0
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})

        if not machine_options:
            score = 0.0
            best_machine_candidate = None
        else:
            durations = list(machine_options.values())
            if not durations:
                score = 0.0
                best_machine_candidate = None
                continue

            # Soft-min: use log-sum-exp to get a smooth minimum that accounts for
            # all machine options (avoids abrupt switching when one machine is marginally faster)
            d_max = max(durations)
            if d_max <= 0:
                d_max = 1.0
            # Normalize for numerical stability
            exps = [np.exp(-d / d_max) for d in durations]
            s = sum(exps)
            if s <= 0:
                soft_min = d_max
            else:
                # soft-min ≈ d_max * log(sum(exp(-d/d_max))) / (-1) ... 
                # Simpler: weighted average biased toward smaller values
                soft_min = sum(d * e for d, e in zip(durations, exps)) / s

            # Pick the actual machine with the true minimum duration for execution
            min_duration = min(durations)
            if min_duration <= 0:
                min_duration = 1.0
            best_machine_candidate = min(machine_options, key=machine_options.get)

            # Heuristic scoring:
            # 1. Critical path urgency: remaining_work / soft_min_duration
            #    Using soft_min makes the score smoother and less sensitive to noise
            urgency = remaining_work / soft_min if soft_min > 0 else remaining_work

            # 2. Completion bonus: prefer ops closer to finishing their job
            #    Fewer remaining ops means this op is more likely on the critical path
            completion_factor = 1.0 / (1.0 + 0.5 * remaining_ops)

            # 3. Duration penalty: longer ops block machines longer
            #    Use the true min_duration for this penalty
            duration_factor = 1.0 / (1.0 + 0.05 * min_duration)

            # 4. Tie-breaker: prefer jobs with less total remaining work (less "mass" left)
            #    This helps when urgency is similar
            mass_factor = 1.0 / (1.0 + 0.01 * remaining_work)

            score = urgency * completion_factor * duration_factor * mass_factor

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_machine_candidate

        # Check deadline periodically
        if i % 50 == 0 and _time.perf_counter() > _deadline:
            break

    return int(best_index)
# EVOLVE-BLOCK-END
