"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def dispatch(ready_operations, time):
    """
    Dispatches the next operation to start.
    Uses a Critical Path heuristic combined with Shortest Processing Time tie-breaking.
    Returns (index, machine) to allow explicit machine selection.
    """
    if not ready_operations:
        return 0, 0 # Fallback, though shouldn't happen if ready is non-empty

    # Calculate critical path length for each unique job present in ready_operations
    # Critical path is approximated by the sum of minimum possible durations for remaining ops.
    job_cp = {}
    for op in ready_operations:
        job_id = op["job"]
        if job_id not in job_cp:
            # Sum of min durations for all remaining operations of this job
            cp_len = 0
            for rem_op in op["remaining_ops"]:
                # rem_op structure is assumed similar to op, having 'machine_options'
                # If 'machine_options' is not in rem_op, we might need to handle that, 
                # but usually the simulator provides consistent structures.
                # The prompt says op has 'machine_options'. It implies remaining_ops are also ops.
                try:
                    min_dur = min(rem_op["machine_options"].values())
                    cp_len += min_dur
                except (KeyError, TypeError, ValueError):
                    # Fallback if structure is different or empty
                    cp_len += 1.0 
            job_cp[job_id] = cp_len

    best_score = -1
    best_index = 0
    best_machine = None
    best_min_dur = float('inf')

    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        cp_len = job_cp.get(job_id, 0)
        
        # Find the best machine for this operation (shortest duration)
        machine_options = op["machine_options"]
        if not machine_options:
            # Fallback if no options, which shouldn't happen
            continue
            
        min_dur = float('inf')
        best_m_for_op = None
        for m, d in machine_options.items():
            if d < min_dur:
                min_dur = d
                best_m_for_op = m

        # Heuristic:
        # Primary: Higher Critical Path (more work remaining on the critical path)
        # Secondary: Shorter Processing Time (SPT) to reduce contention
        # We can combine them. A common approach is to prioritize the job with the longest critical path.
        # If we just pick max CP, we might ignore SPT.
        # Let's use a score: CP * (1 + 1/min_dur) or similar?
        # Or simply: Sort by CP descending, then min_dur ascending.
        
        # To make it a single scalar for easy comparison in a loop:
        # Score = CP + epsilon * (1/min_dur) ?
        # Actually, CP is in time units. 1/min_dur is in 1/time.
        # Let's stick to lexicographic order for robustness:
        # 1. Max CP
        # 2. Min Duration
        
        # We can simulate this by checking conditions
        current_score = cp_len
        
        if current_score > best_score:
            best_score = current_score
            best_index = i
            best_machine = best_m_for_op
            best_min_dur = min_dur
        elif current_score == best_score:
            # Tie-breaker: Shorter processing time is better
            if min_dur < best_min_dur:
                best_index = i
                best_machine = best_m_for_op
                best_min_dur = min_dur
            elif min_dur == best_min_dur:
                # Further tie-breaker: Prefer the one with fewer machine options (more constrained)?
                # Or just leave it.
                pass

    return (best_index, best_machine)

# EVOLVE-BLOCK-END
