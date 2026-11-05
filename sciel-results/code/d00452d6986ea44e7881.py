# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

_dispatch_start_time = _time.time()

# Global state to track estimated machine availability across dispatch calls
_mach_avail = {}
_dispatch_count = 0


def _elapsed():
    return _time.time() - _dispatch_start_time


def _update_mach_avail(ready_operations, time, chosen_idx=None, chosen_machine=None, chosen_dur=None):
    """Update our estimate of machine availability."""
    global _mach_avail, _dispatch_count
    _dispatch_count += 1
    
    if chosen_idx is not None and chosen_machine is not None and chosen_dur is not None:
        _mach_avail[chosen_machine] = time + chosen_dur


def _estimate_mach_avail(ready_operations, time):
    """Estimate machine availability for all machines mentioned in ready operations."""
    global _mach_avail
    
    all_machines = set()
    for op in ready_operations:
        for m in op["machine_options"]:
            all_machines.add(m)
    
    avail = {}
    for m in all_machines:
        if m in _mach_avail:
            avail[m] = max(_mach_avail[m], time)
        else:
            avail[m] = time
    return avail


def _compute_flow_score(op, mach_avail_est, time, all_ready_ops):
    """
    Compute a flow-based score for an operation.
    
    The idea: model the scheduling as a flow network where:
    - Source -> Job nodes (capacity = remaining work)
    - Job -> Operation nodes (capacity = 1 per operation)
    - Operation -> Machine nodes (capacity = duration on that machine)
    - Machine -> Sink (capacity = estimated available time)
    
    The bottleneck capacity of the shortest augmenting path from the job source
    to the least-loaded machine sink determines the score.
    
    For a ready operation, we compute:
    1. For each eligible machine, the "residual capacity" = estimated free time
    2. The "flow value" through this operation = min(remaining_work, min_residual)
    3. Score = completion_time - flow_value (lower is better)
    
    Additionally, we consider downstream impact: operations that unlock more
    downstream work should be prioritized.
    """
    mo = op["machine_options"]
    if not mo:
        return float('inf')
    
    remaining_work = op["remaining_work"]
    remaining_ops = op["remaining_ops"]
    
    best_score = float('inf')
    
    for m, dur in mo.items():
        est_avail = mach_avail_est.get(m, time)
        completion = est_avail + dur
        
        # Flow-based bottleneck: the minimum residual capacity along the path
        # Residual capacity of the machine = how much more work it can take before
        # becoming the critical resource
        # We approximate this as the estimated free time relative to the duration
        residual = max(0, est_avail - time)  # how long until this machine is free
        bottleneck = min(remaining_work, residual + dur)
        
        # The flow value represents how much "progress" this operation unlocks
        # Higher flow value means this operation is more critical to the makespan
        flow_value = bottleneck
        
        # Downstream impact: operations with more remaining ops in their job
        # unlock more downstream work, so they should be prioritized
        downstream_weight = 1.0 + 0.2 * remaining_ops
        
        # Score: we want to minimize completion time but also maximize the
        # flow value (i.e., prioritize operations that are bottlenecks)
        # Lower score is better
        score = completion / downstream_weight - 0.5 * flow_value
        
        if score < best_score:
            best_score = score
    
    return best_score


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
            best_machine = min(mo, key=mo.get)
            _update_mach_avail(ready_operations, time, 0, best_machine, mo[best_machine])
            return (0, best_machine)
        return 0

    # Estimate machine availability
    mach_avail_est = _estimate_mach_avail(ready_operations, time)
    
    # Compute flow-based scores for each operation
    best_idx = 0
    best_machine = None
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            continue
        
        score = _compute_flow_score(op, mach_avail_est, time, ready_operations)
        
        if score < best_score:
            best_score = score
            best_idx = i
            # Find the best machine for this operation
            best_m_for_op = None
            best_m_score = float('inf')
            for m, dur in mo.items():
                est_avail = mach_avail_est.get(m, time)
                completion = est_avail + dur
                remaining_work = op["remaining_work"]
                remaining_ops = op["remaining_ops"]
                residual = max(0, est_avail - time)
                bottleneck = min(remaining_work, residual + dur)
                flow_value = bottleneck
                downstream_weight = 1.0 + 0.2 * remaining_ops
                m_score = completion / downstream_weight - 0.5 * flow_value
                if m_score < best_m_score:
                    best_m_score = m_score
                    best_m_for_op = m
            best_machine = best_m_for_op
    
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
