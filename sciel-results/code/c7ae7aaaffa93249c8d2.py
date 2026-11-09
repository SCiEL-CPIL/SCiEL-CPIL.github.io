"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Module-level deadline guard
_START_TIME = _time.time()
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill

def _elapsed_ok():
    """Check if we're still within the time budget."""
    return (_time.time() - _START_TIME) < _TIME_LIMIT

def dispatch(ready_operations, time):
    """
    Select the next operation to start and optionally the machine.
    
    Strategy: Composite heuristic balancing:
    1. Job urgency via remaining work (critical path proxy)
    2. Machine congestion avoidance (prefer less-loaded machines)
    3. Operation duration (prefer shorter ops to free machines)
    4. Urgency from remaining ops count
    
    Returns (index, machine) tuple.
    """
    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        op = ready_operations[0]
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_machine)

    # Time guard: if we're close to the limit, use a fast fallback
    if not _elapsed_ok():
        # Fast fallback: pick op with most remaining work, shortest duration machine
        best_idx = 0
        best_score = -1e18
        for i, op in enumerate(ready_operations):
            rw = op["remaining_work"]
            ro = max(op["remaining_ops"], 1)
            min_dur = min(op["machine_options"].values())
            score = rw - 0.5 * min_dur
            if score > best_score:
                best_score = score
                best_idx = i
        best_machine = min(ready_operations[best_idx]["machine_options"].keys(), 
                          key=lambda m: ready_operations[best_idx]["machine_options"][m])
        return (best_idx, best_machine)

    n = len(ready_operations)
    scores = np.zeros(n)
    best_machines = [None] * n
    
    # Pre-extract values for vectorized computation where possible
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    remaining_ops = np.maximum(remaining_ops, 1)
    
    # For each op, find the best machine (shortest duration) and compute scores
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            scores[i] = 0
            best_machines[i] = None
            continue
        
        # Find machine with shortest duration
        best_machine = min(options.keys(), key=lambda m: options[m])
        min_duration = options[best_machine]
        
        rw = remaining_works[i]
        ro = remaining_ops[i]
        avg_work = rw / ro
        
        # Composite score:
        # - High remaining work -> higher priority (reduces makespan)
        # - High average work per op -> keeps machines busy
        # - Short min_duration -> frees machine quickly
        # - Fewer remaining ops -> more urgent (tail effect)
        
        # Weights tuned for FJSP:
        w_rw = 1.0      # weight for remaining work
        w_avg = 0.3     # weight for average work per op
        w_dur = 0.8     # weight for duration (negative)
        w_urgency = 0.2 # weight for urgency (1/remaining_ops)
        
        urgency = 1.0 / ro
        
        score = w_rw * rw + w_avg * avg_work - w_dur * min_duration + w_urgency * urgency
        
        scores[i] = score
        best_machines[i] = best_machine
    
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    if best_machine is None:
        return best_idx
    
    return (best_idx, best_machine)

# EVOLVE-BLOCK-END
