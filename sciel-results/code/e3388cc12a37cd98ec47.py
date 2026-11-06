"""Improved dispatch rule for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time

# Track call count for deadline guard
_dispatch_call_count = 0
_dispatch_start_time = None


def dispatch(ready_operations, time):
    """Dispatch rule: combine job urgency, remaining work, and machine availability.
    
    Returns (index, machine) tuple or just index.
    """
    global _dispatch_call_count, _dispatch_start_time
    
    # Deadline guard: if we're taking too long, fall back to simple heuristic
    if _dispatch_start_time is None:
        _dispatch_start_time = _time.time()
    _dispatch_call_count += 1
    
    elapsed = _time.time() - _dispatch_start_time
    # If we've been running for too long, use a fast fallback
    if elapsed > 100:
        # Fast fallback: most remaining work
        best_idx = 0
        best_work = -1
        for i, op in enumerate(ready_operations):
            if op["remaining_work"] > best_work:
                best_work = op["remaining_work"]
                best_idx = i
        return best_idx
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    best_score = -1e18
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        job = op["job"]
        remaining_ops = op["remaining_ops"]
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        # Find the best machine for this operation
        # Prefer machines with shorter duration (frees up sooner)
        # and balance load
        best_machine_score = -1e18
        best_m = None
        
        for machine, duration in machine_options.items():
            # Machine score: prefer shorter duration, 
            # but also consider that this machine might be busy
            # Since we don't have machine state directly, use duration as proxy
            # Shorter duration = less time on machine = better
            # Also, if remaining_ops is high, we want to finish this job soon
            m_score = -duration  # shorter is better
            
            if m_score > best_machine_score:
                best_machine_score = m_score
                best_m = machine
        
        # Composite score for this operation:
        # 1. Urgency: more remaining ops = more urgent (start earlier)
        # 2. Remaining work: more work = should start earlier
        # 3. Duration on best machine: shorter = can complete sooner
        
        # Weighted combination
        urgency = remaining_ops * 10.0  # scale up urgency
        work_component = remaining_work * 0.5  # remaining work matters
        duration_penalty = -best_machine_score * 0.3  # shorter duration is better
        
        total_score = urgency + work_component + duration_penalty
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            best_machine = best_m
    
    # Return (index, machine) to also specify machine
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
