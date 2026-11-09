"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a Critical Ratio heuristic
    with machine contention awareness.
    
    Strategy:
    1. Estimate machine contention: Count how many ready operations can use each machine.
       Machines with higher counts are assumed to be more "loaded" or constrained.
    2. For each operation, determine the "effective duration" by finding the machine
       that minimizes (duration + contention_penalty * contention_count).
    3. Calculate the Critical Ratio: Remaining_Work / (1 + Effective_Duration).
    4. Select the operation with the highest Critical Ratio. This prioritizes
       operations that have a lot of remaining work and can be completed quickly
       on a less contested machine.
    """
    if not ready_operations:
        return 0

    n_ops = len(ready_operations)
    
    # Step 1: Calculate machine contention (proxy for load)
    machine_contention = {}
    for op in ready_operations:
        for m in op["machine_options"].keys():
            machine_contention[m] = machine_contention.get(m, 0) + 1
    
    # Step 2: For each op, find the best machine considering contention
    # and compute effective duration and critical ratio.
    best_idx = 0
    best_score = -float('inf')
    
    # Weight for contention penalty. 
    # A higher weight means we avoid "popular" machines more aggressively.
    contention_weight = 0.5 
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        # Find the machine that minimizes the "cost" of using it
        # Cost = duration + contention_weight * contention_count
        min_cost = float('inf')
        for m, dur in machine_options.items():
            count = machine_contention.get(m, 0)
            cost = dur + contention_weight * count
            if cost < min_cost:
                min_cost = cost
        
        # Effective duration is the minimum cost found (which includes duration)
        # Note: min_cost is slightly larger than pure duration due to penalty,
        # but it serves as a proxy for "time to completion including waiting".
        # To be more accurate, we could just use the duration of the best machine,
        # but the cost helps select the machine.
        
        # Let's refine: Find the machine with min cost, then use its duration for the ratio?
        # Or use the cost itself? 
        # If we use cost, it penalizes ops on busy machines, lowering their priority.
        # This is desirable: if an op *must* use a busy machine, its "effective" time is longer,
        # so its critical ratio drops, unless it has huge remaining work.
        
        # Critical Ratio = Remaining Work / (1 + Effective Cost)
        # We want to MAXIMIZE this ratio.
        score = remaining_work / (1.0 + min_cost)
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return best_idx
# EVOLVE-BLOCK-END
