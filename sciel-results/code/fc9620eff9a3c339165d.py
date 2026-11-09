"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard to ensure we finish within the 120s limit.
# We use a safety margin of 10s.
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 110.0  # 10s safety margin


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy:
    - For each ready operation, consider all its machine options.
    - We want to minimize the makespan. A good heuristic is to prioritize operations
      that are "critical" - i.e., those with high remaining work, or those that would
      cause a machine to be busy for a long time if started now.
    - Another approach: For each operation, compute the minimum duration across its
      machine options. Prioritize operations with the longest minimum duration (Longest
      Processing Time first, but considering the best machine). This tends to reduce
      idle time and balance load.
    - Also consider remaining_work: operations with more total remaining work are more
      critical to the job's completion.
    
    Let's combine:
    1. For each op, min_duration = min(machine_options.values())
    2. Score = remaining_work * 0.5 + min_duration * 0.5
       This balances job criticality and immediate machine load.
    
    Actually, let's think about this more carefully.
    In open shop, each job has operations that can be processed in any order, and each
    operation can be processed on any of its eligible machines.
    
    Key insight: We want to avoid situations where a machine is idle while an operation
    that could use it is waiting. But we also want to finish jobs quickly.
    
    A strong heuristic for open shop is:
    - Prioritize operations with the largest minimum processing time across their
      machine options. This is because these operations are the most "constraining"
      - they will keep a machine busy for a long time regardless of which machine is
      chosen. By scheduling them early, we reduce the chance of having to wait for them
      later.
    
    Let's also factor in remaining_work to bias toward jobs that have more work left,
    as finishing those jobs sooner can free up resources.
    
    Score = alpha * min_duration + beta * remaining_work
    
    Let's try alpha=1.0, beta=0.1 to primarily use min_duration with a small bias
    toward remaining work.
    """
    if not ready_operations:
        return 0
    
    # Check deadline - if we're close to the deadline, return a quick heuristic
    if _time.time() > _DEADLINE - 5.0:
        # Fallback: just pick the one with max remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    scores = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        if not machine_options:
            scores.append(-1.0)
            continue
        
        # Minimum duration across available machines
        min_dur = min(machine_options.values())
        
        # Remaining work for this job
        rem_work = op["remaining_work"]
        
        # Combined score: prioritize long minimum durations, with a small bias
        # toward operations with more remaining work
        score = 1.0 * min_dur + 0.1 * rem_work
        scores.append(score)
    
    # Pick the operation with the highest score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
