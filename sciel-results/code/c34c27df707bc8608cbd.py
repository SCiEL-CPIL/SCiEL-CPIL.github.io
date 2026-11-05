"""Improved dispatch rule for Flexible Job-Shop Scheduling."""

import math
import time as _time

# Module-level state for tracking machine availability estimates
_mach_avail = {}  # machine_id -> estimated available time
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = _time.time()
    return _time.time() - _start_time

def dispatch(ready_operations, time):
    """
    Picks the ready operation to start next and the best machine for it.
    
    Strategy (refined within the same family):
    1. For each ready operation, compute a smooth soft-min duration across machine options
       using log-sum-exp for a differentiable surrogate of min.
    2. Track estimated machine availability (load balancing) to adjust machine selection.
    3. Score = (Remaining_Work / (soft_min_duration + epsilon)) * congestion_factor
    4. Select the operation with the highest score.
    5. Choose the machine that minimizes (duration + estimated_wait) among eligible machines.
    """
    global _mach_avail, _start_time

    if not ready_operations:
        return 0

    # Deadline guard: if we're getting close to the 150s limit, fall back to simple MWRF
    if _elapsed() > 140:
        # Simple fallback: most work remaining
        best_i = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            rw = op.get('remaining_work', 0)
            if rw > best_rw:
                best_rw = rw
                best_i = i
        mo = ready_operations[best_i].get('machine_options', {})
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (best_i, best_m)
        return best_i

    n = len(ready_operations)
    epsilon = 1e-6
    soft_min_T = 0.5  # Temperature for log-sum-exp soft-min

    # Update machine availability estimates
    # When we pick a machine for an operation, we add its duration to the machine's avail
    # For now, we use the current time as a baseline and track cumulative assignments

    # Pre-compute for each operation: soft-min duration and machine options
    op_soft_min_durs = [0.0] * n
    op_machine_options = [None] * n

    for i, op in enumerate(ready_operations):
        machine_options = op.get('machine_options', {})
        op_machine_options[i] = machine_options
        if not machine_options:
            op_soft_min_durs[i] = float('inf')
            continue

        # Log-sum-exp soft-min: soft_min = -T * log(sum(exp(-d/T)))
        # This gives a smooth approximation to min(d)
        neg_d_over_T = [(-d / soft_min_T) for d in machine_options.values()]
        max_neg = max(neg_d_over_T)
        sum_exp = sum(math.exp(x - max_neg) for x in neg_d_over_T)
        soft_min = -soft_min_T * (max_neg + math.log(sum_exp))
        op_soft_min_durs[i] = soft_min

    # Count machine preferences for congestion (using argmin machine)
    machine_congestion = {}
    for i, op in enumerate(ready_operations):
        mo = op_machine_options[i]
        if not mo:
            continue
        min_machine = min(mo.keys(), key=lambda m: mo[m])
        machine_congestion[min_machine] = machine_congestion.get(min_machine, 0) + 1

    # Score each operation and select the best machine
    best_index = 0
    best_score = -1.0
    best_machine = None
    best_min_dur = float('inf')

    congestion_weight = 0.25

    for i, op in enumerate(ready_operations):
        remaining_work = op.get('remaining_work', 0)
        machine_options = op_machine_options[i]

        if not machine_options:
            continue

        soft_min_dur = op_soft_min_durs[i]
        if soft_min_dur == float('inf'):
            continue

        # Find the min-duration machine for congestion calculation
        min_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        cong = machine_congestion.get(min_machine, 1)
        congestion_factor = 1.0 / (1.0 + congestion_weight * (cong - 1))

        score = (remaining_work / (soft_min_dur + epsilon)) * congestion_factor

        if score > best_score:
            best_score = score
            best_index = i
            # Choose machine: minimize (duration + estimated_wait_on_machine)
            # Use current estimated availability as a proxy for wait
            best_machine = None
            best_mach_cost = float('inf')
            for m_id, dur in machine_options.items():
                est_wait = _mach_avail.get(m_id, 0.0)
                # If machine is already busy, wait = max(0, est_avail - time)
                if est_wait > time:
                    wait = est_wait - time
                else:
                    wait = 0.0
                cost = dur + wait
                if cost < best_mach_cost:
                    best_mach_cost = cost
                    best_machine = m_id
            best_min_dur = soft_min_dur
        elif abs(score - best_score) < 1e-9:
            # Tie-break: prefer shorter soft-min duration
            if soft_min_dur < best_min_dur - 1e-9:
                best_index = i
                best_min_dur = soft_min_dur
                best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])

    # Update machine availability estimate for the chosen machine
    if best_machine is not None:
        mo = ready_operations[best_index].get('machine_options', {})
        if best_machine in mo:
            dur = mo[best_machine]
            current_avail = _mach_avail.get(best_machine, 0.0)
            _mach_avail[best_machine] = max(current_avail, time) + dur

    return (best_index, best_machine)
