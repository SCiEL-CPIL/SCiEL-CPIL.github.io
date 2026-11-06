"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill


def _time_left():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def dispatch(ready_operations, time):
    """
    Load-Balancing Water-Filling heuristic for open-shop scheduling.
    
    For each ready operation, we consider assigning it to the least-loaded
    compatible machine. We score each candidate by how much it increases the
    maximum machine load (L-infinity norm increase). We pick the operation
    that minimizes this increase, with tie-breaking by remaining work (more
    work first) and then by job id for stability.
    
    We maintain a persistent estimate of machine loads across dispatch calls
    to enable water-filling behavior.
    """
    if not ready_operations:
        return 0
    
    # Check deadline - if we're running low on time, fall back to simple heuristic
    if _time_left() < 5.0:
        # Simple fallback: most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Get all unique machines across all ready operations
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    
    if not all_machines:
        return 0
    
    # Try to use persistent machine load estimates
    # We store these as attributes on the function for persistence
    if not hasattr(dispatch, '_machine_loads'):
        dispatch._machine_loads = {}
    
    machine_loads = dispatch._machine_loads
    
    # For each candidate operation, compute the best machine assignment
    # and the resulting increase in max load
    n_ops = len(ready_operations)
    scores = []
    
    # Current max load among all machines (considering only machines that
    # appear in at least one ready op, but also track global max)
    # We'll compute per-candidate the new max load if we assign to least-loaded
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            scores.append((float('inf'), -op["remaining_work"], op["job"]))
            continue
        
        # Find the least-loaded machine among options
        # Use current estimated loads
        best_machine = None
        best_load = float('inf')
        for m, dur in machine_options.items():
            current_load = machine_loads.get(m, 0)
            if current_load < best_load:
                best_load = current_load
                best_machine = m
        
        # The new load on this machine
        new_load = best_load + machine_options[best_machine]
        
        # Compute the resulting max load across all machines
        # We need to know the current max load across ALL machines
        # For efficiency, we track a global max
        if not hasattr(dispatch, '_global_max_load'):
            dispatch._global_max_load = 0
        
        # The max load after assignment is max(current global max, new_load)
        # But we should consider all machines, not just those in ready ops
        # For simplicity, use the tracked global max
        current_global_max = dispatch._global_max_load
        resulting_max = max(current_global_max, new_load)
        
        # Score: the increase in max load (lower is better)
        # If resulting_max > current_global_max, the increase is resulting_max - current_global_max
        # Otherwise, the increase is 0
        increase = max(0, resulting_max - current_global_max)
        
        # Tie-breaker: prefer operations with more remaining work (to clear them)
        # Secondary tie-breaker: shorter duration on assigned machine
        dur = machine_options[best_machine]
        
        scores.append((increase, -op["remaining_work"], dur, op["job"]))
    
    # Pick the operation with the minimum score (lowest increase, then most work)
    best_idx = int(np.argmin(scores, key=lambda x: x))
    
    # Update persistent state: simulate the assignment
    op = ready_operations[best_idx]
    machine_options = op["machine_options"]
    
    # Find the least-loaded machine (same logic as above)
    best_machine = None
    best_load = float('inf')
    for m, dur in machine_options.items():
        current_load = machine_loads.get(m, 0)
        if current_load < best_load:
            best_load = current_load
            best_machine = m
    
    new_load = best_load + machine_options[best_machine]
    
    # Update machine load
    machine_loads[best_machine] = new_load
    
    # Update global max
    dispatch._global_max_load = max(dispatch._global_max_load, new_load)
    
    return best_idx
# EVOLVE-BLOCK-END
