# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

_dispatch_start_time = _time.time()
_rng = np.random.default_rng(42)

# Global state to track estimated machine availability across dispatch calls
_mach_avail = {}
_dispatch_count = 0
_best_strategy = None  # 'min_completion', 'max_work', 'balanced'


def _elapsed():
    return _time.time() - _dispatch_start_time


def _update_mach_avail(ready_operations, time, chosen_idx=None, chosen_machine=None, chosen_dur=None):
    """Update our estimate of machine availability.
    
    When we dispatch an operation to a machine, we know that machine will be
    busy until time + duration. We track this to make better future decisions.
    """
    global _mach_avail, _dispatch_count
    _dispatch_count += 1
    
    if chosen_idx is not None and chosen_machine is not None and chosen_dur is not None:
        # We just assigned this operation; update machine availability
        _mach_avail[chosen_machine] = time + chosen_dur
    
    # For machines not yet tracked, estimate their availability
    # A machine that has never been seen is likely free or nearly free
    # We can't know for sure, so we assume it's available at current time
    # unless we have better info


def _estimate_mach_avail(ready_operations, time):
    """Estimate machine availability for all machines mentioned in ready operations.
    
    Returns a dict mapping machine -> estimated available time.
    """
    global _mach_avail
    
    # Collect all machines mentioned
    all_machines = set()
    for op in ready_operations:
        for m in op["machine_options"]:
            all_machines.add(m)
    
    # For each machine, use our tracked value if available, otherwise estimate
    # If a machine was last assigned to finish at T, it's available at max(T, time)
    # If never seen, assume it's available at time (optimistic) or slightly before
    avail = {}
    for m in all_machines:
        if m in _mach_avail:
            avail[m] = max(_mach_avail[m], time)
        else:
            # Never seen this machine; assume it's free
            avail[m] = time
    return avail


def _score_operation_lookahead(op, mach_avail_est, time):
    """Score an operation by simulating its assignment to the best eligible machine.
    
    Returns (best_machine, score) where lower score is better (minimizes makespan impact).
    """
    mo = op["machine_options"]
    if not mo:
        return None, float('inf')
    
    best_machine = None
    best_score = float('inf')
    
    for m, dur in mo.items():
        est_avail = mach_avail_est.get(m, time)
        completion = est_avail + dur
        
        # Score: we want to minimize the completion time, but also consider
        # the remaining work of this job (jobs with more remaining work should
        # be prioritized to avoid stragglers)
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Primary: minimize completion time (load balancing)
        # Secondary: favor jobs with more remaining work (urgency)
        # Tertiary: favor jobs closer to completion (reduce stragglers)
        urgency_bonus = -0.3 * remaining_work  # more remaining work = higher priority
        completion_bonus = 0.1 * remaining_ops  # more ops left = slightly higher priority
        
        total_score = completion + urgency_bonus + completion_bonus
        
        if total_score < best_score:
            best_score = total_score
            best_machine = m
    
    return best_machine, best_score


def _optimize_strategy(ready_operations, time):
    """Determine the best strategy based on the current state.
    
    We use a simple heuristic: if there are many ready operations, prioritize
    load balancing (min completion). If few, prioritize urgency (max remaining work).
    """
    n = len(ready_operations)
    if n <= 2:
        return 'urgency'  # few options, pick the most urgent
    elif n <= 5:
        return 'balanced'  # moderate, balance both
    else:
        return 'min_completion'  # many options, focus on load balancing


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    if _elapsed() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick the operation with the most remaining work
        best_idx = 0
        best_work = ready_operations[0]["remaining_work"]
        for i, op in enumerate(ready_operations):
            if op["remaining_work"] > best_work:
                best_work = op["remaining_work"]
                best_idx = i
        mo = ready_operations[best_idx]["machine_options"]
        if mo:
            best_machine = min(mo, key=mo.get)
            return (best_idx, best_machine)
        return best_idx

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        mo = op["machine_options"]
        if mo:
            # Pick the machine with the shortest duration
            best_machine = min(mo, key=mo.get)
            _update_mach_avail(ready_operations, time, 0, best_machine, mo[best_machine])
            return (0, best_machine)
        return 0

    # Estimate machine availability
    mach_avail_est = _estimate_mach_avail(ready_operations, time)
    
    # Score each operation using lookahead
    best_idx = 0
    best_machine = None
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            continue
        
        machine, score = _score_operation_lookahead(op, mach_avail_est, time)
        
        if score < best_score:
            best_score = score
            best_idx = i
            best_machine = machine
    
    # If no valid machine found, fallback to max remaining work
    if best_machine is None:
        best_idx = 0
        best_work = ready_operations[0]["remaining_work"]
        for i, op in enumerate(ready_operations):
            if op["remaining_work"] > best_work:
                best_work = op["remaining_work"]
                best_idx = i
        mo = ready_operations[best_idx]["machine_options"]
        if mo:
            best_machine = min(mo, key=mo.get)
    
    # Update machine availability tracking
    mo = ready_operations[best_idx]["machine_options"]
    if best_machine is not None and best_machine in mo:
        _update_mach_avail(ready_operations, time, best_idx, best_machine, mo[best_machine])
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
