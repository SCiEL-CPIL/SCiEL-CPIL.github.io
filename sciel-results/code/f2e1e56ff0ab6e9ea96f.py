"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

# Machine load tracking: accumulate estimated work assigned to each machine
_machine_load = {}

def _bump_machine_load(machine, duration):
    _machine_load[machine] = _machine_load.get(machine, 0.0) + duration

def dispatch(ready_operations, time):
    """Pick the ready operation to start next, optionally with machine choice.
    
    Strategy: Combined score using remaining work (WSPT), tightness, and
    machine load balancing. Returns (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick least-loaded eligible machine, tiebreak by shortest duration
        mo = op["machine_options"]
        best_m = min(mo, key=lambda m: (_machine_load.get(m, 0.0) + mo[m], mo[m]))
        _bump_machine_load(best_m, mo[best_m])
        return (0, best_m)
    
    # Check deadline
    if _time.time() > _DEADLINE:
        # Fallback: just pick max remaining work
        best_idx = int(np.argmax([op["remaining_work"] for op in ready_operations]))
        op = ready_operations[best_idx]
        mo = op["machine_options"]
        best_m = min(mo, key=lambda m: (_machine_load.get(m, 0.0) + mo[m], mo[m]))
        _bump_machine_load(best_m, mo[best_m])
        return (best_idx, best_m)
    
    # For each ready op, evaluate all candidate machines
    # Score combines:
    #   - WSPT weight: remaining_work / min_duration (higher = more urgent)
    #   - Machine load: prefer less-loaded machines
    #   - Critical path: ops with fewer remaining_ops are closer to completion
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        min_dur = min(mo.values())
        
        # WSPT-like weight: remaining work relative to shortest processing time
        wspt = remaining_work / (min_dur + 1e-9)
        
        # Critical path weight: operations with fewer remaining ops are on the critical path
        # Higher weight for ops that will complete sooner (fewer remaining ops)
        critical_weight = 1.0 / (remaining_ops + 1.0)
        
        # For each candidate machine, compute a combined score
        for m, dur in mo.items():
            ml = _machine_load.get(m, 0.0)
            # Machine utilization penalty: heavier load = less preferred
            # Use relative load to avoid scale issues
            load_penalty = ml / (ml + dur + 1e-9)
            
            # Combined score:
            #   5 * wspt (workload urgency)
            #   3 * critical_weight (proximity to job completion)
            #   -2 * load_penalty (avoid overloading machines)
            #   -0.5 * dur/min_dur (prefer shorter durations on this machine)
            score = (5.0 * wspt + 
                     3.0 * critical_weight - 
                     2.0 * load_penalty - 
                     0.5 * (dur / (min_dur + 1e-9)))
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = m
    
    # Update machine load for the chosen assignment
    op = ready_operations[best_idx]
    _bump_machine_load(best_machine, op["machine_options"][best_machine])
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
