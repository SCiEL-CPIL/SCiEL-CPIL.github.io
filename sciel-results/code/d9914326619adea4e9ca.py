"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE = 110.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def _softmin(values, temperature=1.0):
    """Smooth minimum using log-sum-exp trick."""
    if not values:
        return 0.0
    min_val = min(values)
    if min_val <= 0:
        return min_val
    # Shift to avoid overflow
    shifted = [v / min_val for v in values]
    log_sum = math.log(sum(math.exp(-v / temperature) for v in values))
    return -temperature * log_sum


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.
    
    Strategy:
    1. For each ready operation, compute urgency as remaining_work * (1 + remaining_ops * 0.1)
       to account for critical path length.
    2. Estimate machine congestion by summing durations of other ready operations
       that can also use that machine.
    3. Calculate a score for each (operation, machine) pair:
       score = urgency / (duration + congestion_penalty * congestion)
    4. Select the (operation, machine) pair with the highest score.
    5. This balances job urgency (including path length) against machine load.
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

    n = len(ready_operations)
    
    # Precompute remaining work, remaining ops, and machine options for all ready ops
    # op_data[i] = (remaining_work, remaining_ops, {machine: duration})
    op_data = []
    for op in ready_operations:
        rw = float(op.get("remaining_work", 0.0))
        ro = int(op.get("remaining_ops", 1))
        opts = op.get("machine_options", {})
        op_data.append((rw, ro, opts))

    # Precompute machine contention/backlog for each machine
    # backlog[m] = sum of durations of all ready ops that can use m
    machine_load = {}
    for rw, ro, opts in op_data:
        for m, d in opts.items():
            machine_load[m] = machine_load.get(m, 0.0) + float(d)

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    # Congestion penalty weight. 
    # Higher weight penalizes busy machines more.
    CONGESTION_WEIGHT = 0.5
    # Critical path weight: adds urgency for jobs with more remaining operations
    CRITICAL_PATH_WEIGHT = 0.1

    for i in range(n):
        rw, ro, opts = op_data[i]
        
        if rw <= 0:
            continue

        # Compute urgency: remaining work scaled by critical path length
        urgency = rw * (1.0 + CRITICAL_PATH_WEIGHT * ro)

        # Find the best machine for this operation
        best_m_for_op = None
        best_pair_score = -float('inf')
        
        for m, d in opts.items():
            d = float(d)
            if d == 0:
                continue
            
            # Get the total load on this machine
            load = machine_load.get(m, d)
            
            # Effective cost: Duration + Penalty * (Load - Duration)
            # This penalizes machines that are heavily contended by other ready ops.
            # (Load - Duration) is the backlog of *other* ops.
            other_load = load - d
            
            effective_cost = d + CONGESTION_WEIGHT * other_load
            
            if effective_cost <= 0:
                continue

            # Score: Urgency / Effective Cost
            score = urgency / effective_cost
            
            if score > best_pair_score:
                best_pair_score = score
                best_m_for_op = m

        if best_m_for_op is not None and best_pair_score > best_score:
            best_score = best_pair_score
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
