"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 110.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.
    
    Strategy:
    1. Calculate 'Urgency' for each ready operation based on remaining work.
    2. Calculate 'Cost' for each operation-machine pair, penalizing machines
       with high contention (many ready ops can use it) and long durations.
    3. Select the operation with the highest Urgency/Cost ratio.
    4. Select the machine that minimizes the Cost for that operation.
    """
    if _elapsed() > _DEADLINE:
        # Hard safety fallback: Shortest Processing Time (SPT)
        if not ready_operations:
            return 0
        best_i = 0
        best_d = float('inf')
        for i, op in enumerate(ready_operations):
            opts = op["machine_options"]
            if not opts:
                continue
            min_d = min(opts.values())
            if min_d < best_d:
                best_d = min_d
                best_i = i
        return best_i

    if not ready_operations:
        return 0
    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    # Precompute machine contention: count how many ready ops can use each machine
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_count[m] = machine_count.get(m, 0) + 1

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    # Contention penalty weight. Higher means we avoid crowded machines more.
    CONTENTION_WEIGHT = 0.2

    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        
        # If no remaining work, skip (shouldn't happen for ready ops)
        if rw <= 0:
            continue

        # Urgency: Prioritize jobs with more remaining work
        urgency = rw

        # Find the best machine for this operation considering contention
        min_cost = float('inf')
        best_m_for_op = None
        
        for m, d in op["machine_options"].items():
            count = machine_count.get(m, 1)
            # Cost = Duration * (1 + Weight * (Count - 1))
            # If count is 1, factor is 1. If count is 2, factor is 1 + Weight.
            factor = 1.0 + CONTENTION_WEIGHT * (count - 1)
            cost = float(d) * factor
            
            if cost < min_cost:
                min_cost = cost
                best_m_for_op = m

        if min_cost == float('inf') or min_cost == 0:
            continue

        # Score: High urgency, low cost
        score = urgency / (min_cost + 1e-9)
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    # Fallback if no valid score was found
    if best_machine is None:
        best_index = 0
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_machine = min(opts, key=lambda k: opts[k])
        else:
            best_machine = None

    if best_machine is not None:
        return (best_index, best_machine)
    return best_index
# EVOLVE-BLOCK-END
