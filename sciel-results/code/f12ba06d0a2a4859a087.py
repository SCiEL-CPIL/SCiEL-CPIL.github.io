import numpy as np
import time as time_module

# ------------------------------------------------------------------
# Open-Shop dispatch rule: LPT-inspired Greedy with Look-Ahead
#
# Strategy:
# 1. Primary heuristic: Longest Processing Time (LPT) approximation.
#    Prioritize operations with the largest minimum processing time.
#    This prevents late-stage bottlenecks by scheduling long, inflexible
#    tasks early when machine availability is high.
# 2. Tie-breaking: Higher remaining_work is preferred to keep machines busy.
# 3. Refinement: Greedy look-ahead simulation for top-K candidates.
#    The simulation assigns each operation to the machine that yields
#    the earliest completion time, providing a better estimate of makespan.
# ------------------------------------------------------------------

_DEADLINE = 150.0
_START_TIME = time_module.time()
_TIME_MARGIN = 2.0

def _elapsed():
    return time_module.time() - _START_TIME

def _remaining_time():
    return _DEADLINE - _TIME_MARGIN - _elapsed()

def _get_min_duration(op):
    """Get the minimum processing time among available machines."""
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0
    return min(mo.values())

def _count_machine_loads(ready_operations):
    """
    Build a dict mapping machine -> count of ready ops that use it.
    Used as a secondary congestion proxy.
    """
    loads = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo.keys():
            loads[m] = loads.get(m, 0) + 1
    return loads

def _score_op_lpt(op, machine_loads=None):
    """
    Score a single ready operation using LPT-inspired logic.
    Higher score = more preferred.
    
    Primary: max_min_duration (LPT)
    Secondary: remaining_work (Workload)
    Tertiary: -machine_load (Congestion avoidance)
    """
    min_dur = _get_min_duration(op)
    rw = op.get("remaining_work", 0.0)
    
    # Calculate min load among available machines for this op
    mo = op.get("machine_options", {})
    min_load = 0
    if machine_loads is not None and mo:
        loads = [machine_loads.get(m, 0) for m in mo.keys()]
        min_load = min(loads) if loads else 0

    # Weights:
    # LPT is the dominant factor for makespan minimization in this context.
    # We normalize duration and work to be on similar scales if possible, 
    # but since we don't know the scale, we rely on relative ordering.
    # A large min_dur is very good.
    
    # Heuristic weights
    W_LPT = 1.0
    W_RW = 0.1
    W_LOAD = 0.05
    
    score = (W_LPT * min_dur 
             + W_RW * rw 
             - W_LOAD * min_load)
             
    return score

def _simulate_makespan_lpt(ready_operations, chosen_idx, time, machine_loads):
    """
    Estimate the makespan if we schedule `chosen_idx` now, and then greedily
    schedule the remaining ready operations using an LPT/ECT hybrid.
    
    1. Schedule chosen_idx on its best machine (min completion time).
    2. For remaining ops, repeatedly pick the op that can finish earliest 
       (ECT) on its best available machine.
    """
    n = len(ready_operations)
    if n == 0:
        return 0.0
    
    # 1. Determine machine for the chosen op.
    op_chosen = ready_operations[chosen_idx]
    mo_chosen = op_chosen.get("machine_options", {})
    
    # Pick machine that minimizes finish time: virtual_avail[m] + duration
    # Initially, all machines are available at `time` except we don't know
    # the true avail, but we assume `time` is the current global time.
    # However, in the simulation context, we assume `time` is when we start.
    # So finish = time + duration. We pick min duration.
    
    best_m_chosen = None
    min_finish = float('inf')
    for m, d in mo_chosen.items():
        finish = time + d
        if finish < min_finish:
            min_finish = finish
            best_m_chosen = m
            
    if best_m_chosen is None:
        # Fallback
        best_m_chosen = list(mo_chosen.keys())[0] if mo_chosen else 0
        min_finish = time + mo_chosen.get(best_m_chosen, 0.0)
        
    # Initialize virtual availability for all machines
    all_machines = set()
    for op in ready_operations:
        for m in op.get("machine_options", {}).keys():
            all_machines.add(m)
            
    virtual_avail = {m: time for m in all_machines}
    virtual_avail[best_m_chosen] = min_finish
    
    remaining_indices = [i for i in range(n) if i != chosen_idx]
    max_completion = min_finish
    
    # 2. Greedily schedule remaining ops.
    # Strategy: Earliest Completion Time (ECT) on best available machine.
    while remaining_indices:
        best_i = -1
        best_m = -1
        best_finish = float('inf')
        
        for i in remaining_indices:
            op = ready_operations[i]
            mo = op.get("machine_options", {})
            for m, d in mo.items():
                start_time = virtual_avail[m]
                finish = start_time + d
                if finish < best_finish:
                    best_finish = finish
                    best_i = i
                    best_m = m
                    
        if best_i == -1:
            break
            
        # Update state
        virtual_avail[best_m] = best_finish
        if best_finish > max_completion:
            max_completion = best_finish
        remaining_indices.remove(best_i)
        
    return max_completion

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    if _remaining_time() < 0.5:
        # Fallback to simple LPT
        return _simple_lpt_heuristic(ready_operations, time)
    
    n = len(ready_operations)
    
    # 1. Compute machine loads
    machine_loads = _count_machine_loads(ready_operations)
    
    # 2. Score all ops with LPT heuristic
    scores = []
    for i, op in enumerate(ready_operations):
        s = _score_op_lpt(op, machine_loads)
        scores.append((s, i))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    if n <= 2:
        return scores[0][1]
    
    # 3. Look-ahead refinement for top-K candidates
    top_k = min(3, n)
    
    best_idx = scores[0][1]
    best_mk = float('inf')
    
    for k in range(top_k):
        idx = scores[k][1]
        
        if _remaining_time() < 1.0:
            break
            
        est_mk = _simulate_makespan_lpt(ready_operations, idx, time, machine_loads)
        
        # We want to minimize makespan.
        if est_mk < best_mk - 1e-6:
            best_mk = est_mk
            best_idx = idx
        elif abs(est_mk - best_mk) < 1e-6:
            # Tie-breaker: keep the one with higher primary LPT score
            # Since we iterate from highest score down, the first one found 
            # with this mk is the best. We don't change best_idx.
            pass
            
    return best_idx

def _simple_lpt_heuristic(ready_operations, time):
    """
    Fast fallback: Pick op with max(min_duration), tie-break by remaining_work.
    """
    best_idx = 0
    best_score = float('-inf')
    
    for i, op in enumerate(ready_operations):
        min_dur = _get_min_duration(op)
        rw = op.get("remaining_work", 0.0)
        
        # Simple LPT score
        s = 1.0 * min_dur + 0.1 * rw
        
        if s > best_score:
            best_score = s
            best_idx = i
            
    return best_idx
