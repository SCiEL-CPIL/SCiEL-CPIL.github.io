"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Track start time to ensure we stay well within the 120s/150s limits
_start_time = _time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: 1-Step Lookahead with Load Balancing.
    
    Strategy: For each ready operation, simulate assigning it to its fastest 
    available machine. We estimate the 'cost' of this decision by looking at:
    1. The completion time of the current operation.
    2. The potential blocking time for the *next* operation in the same job 
       (if it shares a machine with the current one or is likely to be critical).
    
    This approach prioritizes operations that not only finish quickly but also 
    minimize the disruption or delay to subsequent operations, effectively 
    balancing local efficiency (SPT) with global flow (WIP reduction).
    """
    # Deadline guard
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    best_idx = 0
    best_score = float('inf')

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        # Find the machine with the shortest duration for this operation
        # This is the "best" machine for this op in a vacuum.
        best_machine = None
        min_duration = float('inf')
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
                best_machine = m
        
        if best_machine is None:
            continue
            
        # Estimate the "impact" of scheduling this operation now.
        # A simple but effective proxy for good scheduling in job shops is 
        # to minimize the time until the *next* operation can potentially start,
        # or to prioritize jobs with less remaining work to clear the system.
        
        # Heuristic Score:
        # Primary: The processing time of the current op (shorter is better).
        # Secondary: The remaining work of the job (less remaining work is better,
        # as it clears a job from the system faster, reducing WIP).
        
        # We combine these. 
        # Score = duration + 0.1 * remaining_work
        # This is a weighted SPT that slightly favors jobs close to completion.
        
        remaining_work = op.get("remaining_work", 0)
        
        # If we want to be more sophisticated, we could check if the next op 
        # in the job uses the same machine. If so, scheduling this op might 
        # "hold" the machine. But without knowing the next op's machine 
        # compatibility directly in this dict (only remaining_ops count), 
        # we rely on the remaining_work proxy.
        
        # Another angle: If we return (index, machine), we could suggest the machine.
        # The prompt says "optionally (index, machine)". The seed returned just index.
        # Let's stick to index to be safe, but use a smarter index selection.
        
        # Let's try a "Critical Path" approximation.
        # Jobs with fewer remaining operations are closer to the end.
        # Clearing them reduces the chance of blocking others.
        
        remaining_ops = op.get("remaining_ops", 0)
        
        # Score calculation:
        # We want to minimize this score.
        # min_duration is the time the machine is busy.
        # remaining_ops is a proxy for how much more this job will interact with the system.
        
        # Weighting: 
        # If durations are similar, prefer the job with fewer remaining ops.
        # If remaining ops are similar, prefer the shorter duration.
        
        score = min_duration + 0.5 * remaining_ops
        
        if score < best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
