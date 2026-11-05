"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = 140.0  # 10s safety margin before 150s

def dispatch(ready_operations, time):
    # Check deadline
    if _time.time() > _DEADLINE:
        # Fallback: Return first index to prevent crash/timeout score 0
        return 0

    if not ready_operations:
        return 0

    # Strategy: Critical Path + Congestion Avoidance
    # 1. Prioritize operations with high remaining work (Critical Path) to ensure long jobs are started early.
    # 2. For the selected operation, select the machine that minimizes the estimated completion time.
    #    Since we don't know machine availability directly, we estimate load by the current time + duration.
    #    However, a better proxy for "least loaded" in a static dispatch without state is often 
    #    the machine with the shortest duration for this specific op, BUT combined with 
    #    the urgency of the job.
    
    # Let's refine the heuristic:
    # Primary Key: Remaining Work (Higher is better -> Urgency)
    # Secondary Key: Min Duration (Lower is better -> Ease)
    # Tertiary Key: Index (Stability)
    
    # Wait, the current program uses SPT (Min Duration) as primary. 
    # The inspiration (score 1.0) used Argmax Remaining Work.
    # The current program (score 0.87) uses SPT.
    # Usually, in Job Shop, Critical Path (Longest Remaining) is a strong heuristic.
    # Let's try a weighted combination or a lexicographic approach favoring Remaining Work.
    
    # Let's try: Maximize Remaining Work. Tie-break with Min Duration.
    # If we return (index, machine), we can explicitly pick the best machine.
    # Which machine is best? The one with the shortest duration for this op.
    
    best_score = -1
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        machine_opts = op["machine_options"]
        
        # Find the best machine for this op (shortest duration)
        min_dur = float('inf')
        best_m_for_op = None
        for m, d in machine_opts.items():
            if d < min_dur:
                min_dur = d
                best_m_for_op = m
        
        # Heuristic Score:
        # We want high rem_work.
        # We want low min_dur (as a tie-breaker or secondary factor).
        # Let's use a composite score: rem_work - 0.1 * min_dur
        # Or simply lexicographic: (rem_work, -min_dur)
        
        # Let's try a weighted sum to allow trade-offs.
        # If rem_work is very large, we prefer it even if duration is slightly longer.
        score = rem_work - 0.5 * min_dur
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op

    # Return (index, machine) to explicitly guide the simulator
    if best_machine is not None:
        return (int(best_idx), best_machine)
    else:
        return int(best_idx)
# EVOLVE-BLOCK-END
