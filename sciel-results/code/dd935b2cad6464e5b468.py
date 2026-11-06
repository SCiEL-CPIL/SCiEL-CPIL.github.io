"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling using a new approach:
    Priority based on total remaining workload of the job and machine contention.
    
    Strategy:
    - Prioritize operations belonging to jobs with more total remaining work (to prevent
      long jobs from being delayed and increasing makespan)
    - Among those, prefer operations with shorter processing times on their assigned machines
      to free up machine capacity quickly
    - Add a small bonus for operations that are the last remaining op of their job
    
    Returns the index of the best operation to start next.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Calculate scores for each ready operation
    scores = np.zeros(n)
    
    # Pre-compute statistics for normalization
    remaining_works = [op["remaining_work"] for op in ready_operations]
    max_rw = max(remaining_works) if remaining_works else 1
    if max_rw == 0:
        max_rw = 1
    
    min_durs = []
    for op in ready_operations:
        if op["machine_options"]:
            min_durs.append(min(op["machine_options"].values()))
        else:
            min_durs.append(0)
    max_md = max(min_durs) if min_durs else 1
    if max_md == 0:
        max_md = 1
    
    # Track job-level remaining work to prioritize jobs with more work left
    # We don't have direct job-level totals, but remaining_work per op is a proxy
    # Actually, let's use a different approach:
    # Score = weight1 * (remaining_work / max_rw) + weight2 * (1 - min_dur / max_md) + weight3 * (1 / (1 + remaining_ops))
    
    for i, op in enumerate(ready_operations):
        # Component 1: Remaining work of this operation (higher is better)
        # This helps ensure that operations with significant remaining work get scheduled
        rw_score = op["remaining_work"] / max_rw
        
        # Component 2: Processing time on the best available machine (lower is better)
        if op["machine_options"]:
            md = min(op["machine_options"].values())
        else:
            md = 0
        dur_score = 1.0 - (md / max_md)
        
        # Component 3: Completion bonus - operations that are the last for their job
        # get a boost to complete jobs sooner
        rem_ops = op["remaining_ops"]
        completion_bonus = 0.3 if rem_ops <= 1 else 0.0
        
        # Weighted combination with emphasis on remaining work and short duration
        scores[i] = 0.5 * rw_score + 0.3 * dur_score + completion_bonus
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
