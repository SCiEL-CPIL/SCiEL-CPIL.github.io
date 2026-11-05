"""Open-Shop Scheduling dispatch rule using a load-aware scoring approach."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_DEADLINE = 120.0
_SAFETY_MARGIN = 5.0


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Returns (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Deadline guard
    if _elapsed() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: Max remaining work, then Min duration
        best_idx = 0
        best_rw = -1
        best_min_dur = float('inf')
        for i, op in enumerate(ready_operations):
            rw = op["remaining_work"]
            durations = op["machine_options"]
            min_d = min(durations.values()) if durations else float('inf')
            if rw > best_rw or (rw == best_rw and min_d < best_min_dur):
                best_rw = rw
                best_min_dur = min_d
                best_idx = i
        
        op = ready_operations[best_idx]
        durations = op["machine_options"]
        best_m = min(durations, key=lambda m: durations[m]) if durations else 0
        return (best_idx, best_m)

    if n == 1:
        op = ready_operations[0]
        durations = op["machine_options"]
        best_m = min(durations, key=lambda m: durations[m]) if durations else 0
        return (0, best_m)

    # 1. Calculate total pending load for each machine from the ready set
    # This represents the "congestion" on each machine.
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            machine_load[m] = machine_load.get(m, 0.0) + float(d)

    # Parameters
    # LAMBDA: Weight of the effective cost penalty.
    # BETA: Weight of the load factor in the effective cost.
    # We want to balance criticality (remaining_work) against the cost of scheduling
    # (which is higher if the machine is already heavily loaded with other ready ops).
    LAMBDA = 1.0
    BETA = 0.5

    best_score = -float('inf')
    best_idx = 0
    best_machine = 0

    for i, op in enumerate(ready_operations):
        rw = float(op["remaining_work"])
        durations = op["machine_options"]
        
        if not durations:
            continue

        # Find the best machine for this operation based on effective cost
        # Cost = duration * (1 + beta * load)
        # We want to minimize this cost.
        min_cost = float('inf')
        best_m_for_op = 0
        
        for m, d in durations.items():
            load = machine_load.get(m, 0.0)
            # Load includes this operation's duration.
            # If we schedule this op, it adds to the load, but for comparison,
            # we look at the current "pressure" on the machine.
            cost = float(d) * (1.0 + BETA * load)
            if cost < min_cost:
                min_cost = cost
                best_m_for_op = m

        # Calculate score for this operation
        # Score = remaining_work - LAMBDA * min_cost
        # High remaining work is good (critical).
        # Low effective cost is good (not adding to a bottleneck or taking a long time on a busy machine).
        score = rw - LAMBDA * min_cost
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
