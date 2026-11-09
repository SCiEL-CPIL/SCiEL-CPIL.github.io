"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random

_DEADLINE = _time.time() + 145.0  # safety margin before 150s kill
_RNG = _random.Random(42)


def _fast_dispatch(ready_operations, time):
    """Lightweight heuristic: combine remaining work, machine load, and operation count."""
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Estimate machine loads from ready operations
    machine_loads = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_loads[m] = machine_loads.get(m, 0.0) + 1.0

    scores = []
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_opts = op["machine_options"]

        # Find best machine for this operation (shortest duration, least loaded)
        best_machine_score = float('inf')
        for m, dur in machine_opts.items():
            load = machine_loads.get(m, 0.0)
            # Combine duration and load: prefer shorter, less loaded
            m_score = dur * (1.0 + 0.3 * load)
            if m_score < best_machine_score:
                best_machine_score = m_score

        # Score: higher is better
        # Weight remaining work heavily, penalize long best-machine time
        score = remaining_work * 10.0 - best_machine_score * 2.0 + remaining_ops * 1.0
        scores.append(score)

    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    return best_idx


def _lookahead_dispatch(ready_operations, time):
    """One-step lookahead: simulate starting each candidate, score resulting state."""
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    if _time.time() > _DEADLINE:
        return _fast_dispatch(ready_operations, time)

    # Estimate machine availability after each candidate
    machine_loads = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_loads[m] = machine_loads.get(m, 0.0) + 1.0

    best_idx = 0
    best_score = float('-inf')

    for i in range(n):
        op = ready_operations[i]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_opts = op["machine_options"]

        # Pick best machine for this op
        best_m = None
        best_m_dur = float('inf')
        for m, dur in machine_opts.items():
            load = machine_loads.get(m, 0.0)
            m_score = dur * (1.0 + 0.5 * load)
            if m_score < best_m_dur:
                best_m_dur = m_score
                best_m = m

        # Simulate: remove this op, update machine loads
        new_loads = dict(machine_loads)
        if best_m in new_loads:
            new_loads[best_m] -= 1.0
            if new_loads[best_m] <= 0:
                del new_loads[best_m]

        # Score the resulting state: sum of best-machine scores for remaining ops
        state_score = 0.0
        for j, op2 in enumerate(ready_operations):
            if j == i:
                continue
            r_work = op2["remaining_work"]
            r_ops = op2["remaining_ops"]
            m_opts2 = op2["machine_options"]
            # Best machine for op2 in new state
            bm_score = float('inf')
            for m2, d2 in m_opts2.items():
                l2 = new_loads.get(m2, 0.0)
                s2 = d2 * (1.0 + 0.5 * l2)
                if s2 < bm_score:
                    bm_score = s2
            state_score += r_work * 10.0 - bm_score * 2.0 + r_ops * 1.0

        # The chosen op contributes its own "completion benefit"
        completion_benefit = remaining_work * 10.0
        total_score = state_score + completion_benefit

        if total_score > best_score:
            best_score = total_score
            best_idx = i

    return best_idx


def dispatch(ready_operations, time):
    """Dispatch rule: pick which ready operation to start next.
    
    Returns index of the ready operation to start.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # For small sets, use lookahead; for larger, use fast heuristic
    if n <= 8:
        return _lookahead_dispatch(ready_operations, time)
    else:
        return _fast_dispatch(ready_operations, time)
# EVOLVE-BLOCK-END
