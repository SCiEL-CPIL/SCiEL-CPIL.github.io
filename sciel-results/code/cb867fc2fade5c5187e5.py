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

    best_score = -float('inf')
    best_index = 0

    for i, op in enumerate(ready_operations):
        # Deadline guard: if we're running out of time, return best so far
        if _time.perf_counter() > _deadline:
            break

        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})

        if not machine_options:
            score = 0.0
        else:
            # Find the minimum duration among eligible machines
            min_duration = min(machine_options.values())
            if min_duration <= 0:
                min_duration = 1e-9

            # Number of eligible machines: constraint tightness proxy
            # Fewer options = higher urgency (bottleneck mitigation)
            num_machines = len(machine_options)
            constraint_term = 1.0 / (num_machines + 0.5)

            # Work urgency: operations with more remaining work are more critical
            work_term = remaining_work / min_duration

            # SPT term: prefer shorter processing times
            spt_term = 1.0 / min_duration

            # Depth term: favor operations closer to completion (reduce tail)
            depth_term = 1.0 / (remaining_ops + 1)

            # Composite heuristic:
            # Primary: work urgency (high work, short time) - the strongest signal
            # Secondary: constraint tightness (fewest options = most urgent)
            # Tertiary: SPT (shortest processing time)
            # Quaternary: depth (closer to completion)
            score = (
                1.0 * work_term +
                2.0 * constraint_term +
                0.3 * spt_term +
                0.2 * depth_term
            )

        if score > best_score:
            best_score = score
            best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
