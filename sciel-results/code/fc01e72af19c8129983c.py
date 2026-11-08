import time
import numpy as np

def dispatch(ready_operations, time):
    """
    Dispatches the next operation based on a dynamic priority rule.
    
    Strategy:
    - Calculate a priority score for each ready operation.
    - The score combines the minimum possible processing time (SPT component)
      and the remaining work of the job (Critical Path/Workload component).
    - We use a weighted sum: score = w1 * min_duration + w2 * remaining_work.
    - Lower score is better.
    - This attempts to balance keeping machines utilized with short jobs (SPT)
      and ensuring jobs with significant remaining work are not starved,
      which can help reduce the makespan by preventing bottlenecks later.
    """
    if not ready_operations:
        return -1

    # Safety margin: If we are running long, just pick the first one to avoid timeout
    # However, this function is called many times. The overhead is minimal.
    # We assume the caller handles the global timeout, but we keep this fast.
    
    best_index = -1
    best_machine = -1
    best_score = float('inf')
    
    # Weights: 
    # If w1 is high, it behaves like SPT.
    # If w2 is high, it prioritizes jobs with more remaining work (like FAS/CR).
    # A balanced approach often works well. Let's try w1=1.0, w2=0.5
    # This means a job with 2x remaining work but same duration is slightly worse.
    # Or perhaps normalize? 
    # Let's stick to simple additive first. 
    # Actually, remaining_work can be large. 
    # Let's try a ratio or just simple sum. 
    # In Job Shop, remaining work is a strong predictor of makespan contribution.
    
    W_TIME = 1.0
    W_WORK = 0.1  # Scale down work to make it comparable to duration if durations are small
    
    for i, op in enumerate(ready_operations):
        # Find the machine with the shortest duration for this operation
        min_dur = float('inf')
        min_mach = -1
        for m, d in op['machine_options'].items():
            if d < min_dur:
                min_dur = d
                min_mach = m
        
        # Get remaining work
        rem_work = op.get('remaining_work', 0)
        
        # Calculate score
        # We want to minimize the time spent and prioritize jobs that have a lot of work left
        # to avoid them becoming bottlenecks.
        # If rem_work is 0, it's the last op, so pure SPT is good.
        # If rem_work is high, we might want to start it earlier? 
        # Actually, usually Longest Processing Time (LPT) or Most Remaining Work is used 
        # for critical jobs. 
        # Let's try: Score = min_dur + alpha * rem_work.
        # If alpha is positive, we penalize jobs with lots of remaining work? 
        # No, usually we want to *prioritize* jobs with lots of remaining work to finish them.
        # So we want *lower* score for higher remaining work? 
        # Or is it that we want to finish the job? 
        # If we pick a job with high remaining work, we are investing in it.
        # If we pick a job with low remaining work, we clear it quickly.
        # 
        # Let's look at standard heuristics:
        # - SPT: Shortest Processing Time (min duration)
        # - FAS: First Available (arbitrary)
        # - CR: Critical Ratio (due date / remaining time) - no due dates here.
        # - LPT: Longest Processing Time.
        # - MOST: Most Remaining Work.
        #
        # The current program uses SPT. It scored 0.82.
        # Let's try a blend: Score = min_dur * (1 + rem_work / 100)
        # This slightly penalizes jobs with lots of remaining work, favoring SPT but not exclusively.
        # Alternatively, Score = min_dur + rem_work.
        
        # Let's try a different angle: 
        # Prioritize the machine that is free? No, we don't have machine availability in the dict.
        # We only have ready_operations. The simulation likely ensures that if an op is ready,
        # the machine is free OR the op is waiting for the machine.
        # The interface says: "picks which ready operation to start next and, optionally, which eligible machine runs it".
        # This implies we can choose the machine.
        
        # If we can choose the machine, we should pick the one that minimizes the completion time?
        # But we don't know when the machine becomes free.
        # Wait, if the op is "ready", does that mean the machine is free?
        # In many FJS simulations, "ready" means all predecessors are done AND the machine is free.
        # If so, we just pick the best op/machine pair.
        
        # Let's try a heuristic that favors shorter durations but breaks ties by remaining work.
        # Actually, let's try: Score = min_dur. Tie break: higher remaining_work.
        # The current code just uses min_dur. 
        # Let's add a small penalty for low remaining work? 
        # No, let's try: Score = min_dur + 0.1 * (max_remaining_work - op['remaining_work'])
        # This would prioritize jobs with *more* remaining work in case of ties.
        
        # Let's try a more aggressive change: 
        # Use the "Most Remaining Work" (MRW) as the primary driver, 
        # but only if the duration is not significantly longer than the minimum available.
        
        # Simpler: Score = min_dur / (1 + rem_work)
        # This favors short durations and high remaining work.
        
        rem_work_val = op.get('remaining_work', 1)
        if rem_work_val <= 0:
            rem_work_val = 1 # Avoid division by zero or extreme values
        
        score = min_dur / (1.0 + rem_work_val * 0.1)
        
        if score < best_score:
            best_score = score
            best_index = i
            best_machine = min_mach

    return (best_index, best_machine)
