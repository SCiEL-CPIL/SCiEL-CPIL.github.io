"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Strategy: Lookahead Critical Path (LCP)
    # We estimate the criticality of a job by the sum of the minimum processing
    # times of all its remaining operations. This captures the "longest path"
    # aspect of the job shop problem, which is a stronger predictor of makespan
    # than just the total remaining work (which can be inflated by non-critical
    # parallel paths or long but non-bottleneck operations).
    
    # Calculate criticality score for each ready operation
    scores = []
    for op in ready_operations:
        remaining_ops = op.get("remaining_ops", [])
        
        # If remaining_ops is empty or missing, fallback to remaining_work
        if not remaining_ops:
            score = op.get("remaining_work", 0)
        else:
            # Sum the minimum duration across all machine options for each remaining op
            # This represents the shortest possible time left for this job
            min_durations = []
            for rop in remaining_ops:
                machine_options = rop.get("machine_options", {})
                if machine_options:
                    min_durations.append(min(machine_options.values()))
                else:
                    # Fallback if machine_options is empty
                    min_durations.append(0)
            
            score = sum(min_durations)
        
        scores.append(score)

    # Select the operation with the highest criticality score
    # If there's a tie, we could break it by shortest processing time on the assigned machine,
    # but argmax picks the first max. Let's add a secondary criterion for robustness.
    # Actually, simple argmax is often sufficient and fast. 
    # To be more robust, let's break ties by choosing the one with the shortest immediate duration.
    
    best_idx = 0
    best_score = scores[0]
    best_duration = float('inf')
    
    for i, op in enumerate(ready_operations):
        s = scores[i]
        # Get the minimum duration for this operation (assuming we can pick the best machine)
        machine_options = op.get("machine_options", {})
        if machine_options:
            min_dur = min(machine_options.values())
        else:
            min_dur = 0
            
        if s > best_score or (s == best_score and min_dur < best_duration):
            best_score = s
            best_duration = min_dur
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
