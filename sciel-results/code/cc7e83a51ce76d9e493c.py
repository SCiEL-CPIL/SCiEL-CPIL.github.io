"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule that selects the next ready operation to start.
    
    Strategy: Use a composite heuristic that balances:
    1. Critical Path / Longest Remaining Processing Time (LRPT)
    2. Minimum Processing Time (MPT) to reduce immediate machine load
    3. Machine availability/load balancing (approximated by choosing shorter ops on busy machines)
    
    For each ready operation, we compute a score based on:
    - The minimum possible duration on any available machine (to minimize immediate blocking)
    - The remaining work (to prioritize jobs with more work left)
    - A tie-breaker favoring shorter operations to keep the system flowing
    
    We use a weighted combination: prioritize operations that have high remaining work 
    but can be processed quickly on their best machine option.
    """
    if not ready_operations:
        return 0
    
    scores = []
    for op in ready_operations:
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        # Find the minimum duration among available machines for this operation
        if machine_options:
            min_duration = min(machine_options.values())
            # Also consider the average duration as a secondary factor
            avg_duration = np.mean(list(machine_options.values()))
        else:
            min_duration = remaining_work
            avg_duration = remaining_work
        
        # Heuristic: 
        # We want to prioritize operations that:
        # 1. Have high remaining work (critical path)
        # 2. Can be completed quickly (min duration) to free up resources
        # 
        # Score = remaining_work / (min_duration + epsilon)
        # This favors operations with lots of work remaining that can be done quickly
        # Actually, let's think differently:
        # 
        # A good dispatch rule is "Longest Processing Time" for the critical path,
        # but combined with "Shortest Processing Time" for immediate execution.
        #
        # Let's use: score = remaining_work * 0.7 + (1 / (min_duration + 1)) * 0.3
        # This balances long-term job completion with short-term throughput.
        
        # Alternative: Use a ratio that favors operations where the next step is short
        # relative to remaining work, indicating the job is close to completion on this path
        # OR operations with lots of remaining work that have a fast next step.
        
        # Let's try: prioritize by remaining_work, break ties by min_duration (shorter is better)
        # Or: score = remaining_work - alpha * min_duration
        
        # Actually, a well-known effective rule is "SPT" (Shortest Processing Time) 
        # or "LRPT" (Longest Remaining Processing Time). 
        # Let's combine: score = remaining_work / (min_duration + 1e-6)
        # This gives higher score to jobs with more remaining work that can be done faster.
        
        score = remaining_work / (min_duration + 1e-6)
        scores.append(score)
    
    scores = np.array(scores)
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
