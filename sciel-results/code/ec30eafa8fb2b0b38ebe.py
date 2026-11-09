"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random
import math
import heapq

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0  # Safety margin

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: MWR
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    if not ready_operations:
        return 0
    
    # If only one op, return it
    if len(ready_operations) == 1:
        return 0
    
    # Calculate scores for all ops
    # We want to minimize makespan. 
    # Key insights:
    # 1. Jobs with more remaining work (remaining_work) should be prioritized to avoid them becoming critical path bottlenecks late in the schedule.
    # 2. Operations with shorter processing times on their best machine can help keep machines busy and reduce idle time, but if the remaining work is huge, we might want to start it even if it's long, to get it done.
    # 3. A common heuristic is "Longest Remaining Processing Time" (LRPT) which prioritizes jobs with the most remaining work.
    # 4. Another is "Critical Ratio" (remaining work / remaining time), but we don't have a deadline.
    # 5. Let's try a composite score:
    #    score = remaining_work / (1 + min_duration * 0.1)
    #    This strongly favors remaining_work but slightly penalizes very long operations.
    
    # However, the previous score was `rw / (min_d + 1.0)`.
    # Let's try a different angle: 
    # Prioritize the job that has the most remaining work, but if two jobs have similar remaining work, pick the one that can be processed faster (shorter min_duration) to free up resources sooner?
    # Or, pick the one with the longest min_duration if it has the most remaining work?
    
    # Let's look at the "inspiration 3" which scored 1.0281. It used `_fast_dispatch` and `_heuristic_score`.
    # The current program scored 1.0953.
    # The parent scored 1.0820.
    # So the current program is worse than the parent.
    # The parent used state tracking (`_machine_load`, `_job_remaining`, etc.).
    # The current program removed that state tracking and used a simple ratio.
    # This suggests that state tracking or a more sophisticated heuristic is better.
    
    # Let's try to combine the strengths:
    # 1. Use remaining_work as the primary driver.
    # 2. Use remaining_ops as a secondary driver (jobs with more ops left are more complex and might need earlier start to avoid bottlenecks).
    # 3. Use min_duration as a tie-breaker or small penalty.
    
    # Let's try: score = remaining_work * 100 + remaining_ops * 10 - min_duration * 0.1
    # This heavily weights remaining_work, then remaining_ops, then penalizes long operations.
    
    best_score = -float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
        else:
            min_d = 1.0
            
        if min_d < 0:
            min_d = 0
            
        # Score formulation
        # We want to maximize this.
        # High rw is good.
        # High ro is good (more complexity left).
        # Low min_d is slightly good (faster completion of this step).
        
        score = rw * 100.0 + ro * 10.0 - min_d * 0.1
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return int(best_idx)

# EVOLVE-BLOCK-END
