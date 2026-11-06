"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
import random

def dispatch(ready_operations, time):
    """
    Dispatch rule using Stochastic Local Search.
    
    Strategy:
    1. Define a robust base score for each ready operation. We use a composite 
       metric: 
       - Primary: Minimize the "critical path" contribution. We estimate this by 
         looking at the remaining work of the job and the processing time on the 
         least loaded machine it can use.
       - Secondary: Favor operations that have fewer options (bottleneck ops) to 
         resolve conflicts early.
    2. Instead of always picking the argmax, we use a simulated annealing-like 
       acceptance criterion. 
       - Calculate a "temperature" that starts high and decays over the simulation 
         time (or simply use a fixed low temperature for stochasticity).
       - With a small probability, we pick a sub-optimal candidate to explore 
         different scheduling paths. This helps escape local optima where a 
         greedy choice leads to a bottleneck later.
    
    This differs from static MWR or simple load balancing by introducing 
    stochasticity and a search-like exploration of the dispatch space.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Deadline guard
    start_time = _time.time()
    timeout = 0.05  # 50ms limit for this dispatch call to be safe
    
    # 1. Calculate machine loads based on current ready operations
    # We estimate the "potential" load if all ready ops were assigned to their 
    # minimum duration machine. This is a proxy for congestion.
    machine_loads = {}
    for op in ready_operations:
        options = op["machine_options"]
        for machine, duration in options.items():
            if machine not in machine_loads:
                machine_loads[machine] = 0.0
            machine_loads[machine] += duration

    min_load = min(machine_loads.values()) if machine_loads else 1.0
    max_load = max(machine_loads.values()) if machine_loads else 1.0
    load_range = max_load - min_load if max_load > min_load else 1.0

    scores = []
    for i, op in enumerate(ready_operations):
        if _time.time() - start_time > timeout:
            # Fallback to simple MWR if we are taking too long
            return int(np.argmax([o["remaining_work"] for o in ready_operations]))
            
        options = op["machine_options"]
        rem_work = op["remaining_work"]
        
        # Heuristic 1: Machine Freedom
        # Find the machine with the lowest load among those this op can use.
        min_load_for_op = min(machine_loads.get(m, 0.0) for m in options.keys())
        
        # Normalize freedom: 1.0 if on least loaded machine, 0.0 if on most loaded
        if load_range > 0:
            freedom = 1.0 - (min_load_for_op - min_load) / load_range
        else:
            freedom = 1.0

        # Heuristic 2: Urgency / Criticality
        # Jobs with more remaining work are more critical.
        # Normalize remaining work relative to the max remaining work in the pool.
        max_rem_work = max(o["remaining_work"] for o in ready_operations)
        if max_rem_work > 0:
            urgency = rem_work / max_rem_work
        else:
            urgency = 1.0

        # Heuristic 3: Bottleneck Potential
        # Operations with fewer machine options are more constrained.
        num_options = len(options)
        # Fewer options -> higher constraint score. 
        # Max options in pool:
        max_options = max(len(o["machine_options"]) for o in ready_operations)
        if max_options > 0:
            constraint = 1.0 / (num_options + 1)  # 1/1=1, 1/2=0.5, etc.
        else:
            constraint = 1.0

        # Combine scores
        # We want high urgency, high freedom, high constraint.
        # Weights: Urgency is primary, Freedom is secondary, Constraint is tertiary.
        # Using a weighted sum.
        w_urgency = 0.6
        w_freedom = 0.3
        w_constraint = 0.1
        
        score = w_urgency * urgency + w_freedom * freedom + w_constraint * constraint
        scores.append(score)

    # 2. Stochastic Selection
    # We identify the top K candidates (K=3 or 5) and select from them.
    # This is a form of "stochastic hill climbing" or "simulated annealing" 
    # where we accept the best move with high probability, but occasionally 
    # accept a slightly worse one to explore.
    
    # Sort indices by score descending
    sorted_indices = np.argsort(scores)[::-1]
    
    # Temperature parameter: 
    # We can use a fixed small temperature to allow some randomness.
    # Or we can decay it based on 'time' (simulation time).
    # Let's use a fixed probability of picking the 2nd or 3rd best if they are close.
    
    # Determine top 3 candidates
    top_k = min(3, n)
    candidates = sorted_indices[:top_k]
    
    # Calculate probabilities based on scores (Softmax-like)
    # To avoid underflow and handle the "stochastic" part:
    # If the best score is significantly better, pick it. 
    # If they are close, pick randomly among the top few.
    
    best_score = scores[candidates[0]]
    
    # Simple heuristic for stochasticity:
    # With 10% probability, if the 2nd best is within 10% of the best, pick the 2nd best.
    # This introduces minimal exploration without hurting performance too much.
    
    if n > 1 and random.random() < 0.1:
        # Check if 2nd best is close
        if len(candidates) > 1:
            second_score = scores[candidates[1]]
            # Relative difference
            if best_score > 0 and (best_score - second_score) / best_score < 0.1:
                # Pick the second best
                return int(candidates[1])
    
    # Otherwise, pick the best
    return int(candidates[0])

# EVOLVE-BLOCK-END
