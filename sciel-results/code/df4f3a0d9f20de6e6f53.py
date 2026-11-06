"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Look-ahead regret-based heuristic for flexible job-shop scheduling.
    
    For each ready operation, compute a composite score that combines:
    1. Regret: difference between best and second-best machine assignment costs
    2. Urgency: based on remaining work and position in job sequence
    3. Machine congestion: estimated future load on candidate machines
    
    The cost of assigning an operation to a machine is:
      duration + alpha * estimated_future_load(machine)
    
    where estimated_future_load accounts for both current ready operations
    and remaining work of jobs that may need that machine.
    
    We select the operation with the maximum composite score, which balances
    decision urgency (high regret) with job criticality.
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Compute total remaining work per machine across all ready operations
    # and use remaining_work as a proxy for future demand
    machine_load = {}
    total_remaining_work = 0.0
    
    for op in ready_operations:
        total_remaining_work += op["remaining_work"]
        for machine, duration in op["machine_options"].items():
            if machine not in machine_load:
                machine_load[machine] = 0.0
            machine_load[machine] += duration
    
    # Scale factor to normalize machine load estimates
    if total_remaining_work > 0:
        load_scale = 1.0 / (total_remaining_work + 1.0)
    else:
        load_scale = 1.0
    
    # For each operation, compute a composite score
    scores = []
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Compute cost for each machine: duration + weighted future load
        costs = []
        durations = []
        for machine, duration in options.items():
            future_load = machine_load.get(machine, 0.0)
            # Weight future load to reflect congestion
            cost = duration + 0.3 * future_load * load_scale
            costs.append(cost)
            durations.append(duration)
        
        # Compute regret (difference between best and second-best)
        if len(costs) <= 1:
            regret = 0.0
        else:
            sorted_costs = sorted(costs)
            regret = sorted_costs[1] - sorted_costs[0]
        
        # Normalize regret by the minimum cost to make it scale-invariant
        min_cost = min(costs) if costs else 1.0
        normalized_regret = regret / (min_cost + 1e-6)
        
        # Urgency component: prefer operations with more remaining work
        # and fewer remaining operations (closer to completion)
        urgency = remaining_work / (remaining_ops + 1.0)
        
        # Combine regret and urgency with weights
        # Higher regret means the machine choice is more critical
        # Higher urgency means the job needs attention sooner
        score = 0.6 * normalized_regret + 0.4 * urgency
        
        scores.append(score)
    
    # Select the operation with maximum composite score
    max_score = max(scores)
    
    # Find all operations with maximum score (within tolerance for floating point)
    tolerance = 1e-9
    candidates = [i for i in range(n_ops) if scores[i] >= max_score - tolerance]
    
    if len(candidates) == 1:
        return int(candidates[0])
    
    # Tie-breaking: prefer operation with more remaining work, then shorter min duration
    def tiebreak_score(i):
        op = ready_operations[i]
        remaining_work = op["remaining_work"]
        min_duration = min(op["machine_options"].values())
        # Higher remaining work is better, shorter min duration is better
        return (remaining_work, -min_duration)
    
    candidates.sort(key=tiebreak_score, reverse=True)
    return int(candidates[0])
# EVOLVE-BLOCK-END
