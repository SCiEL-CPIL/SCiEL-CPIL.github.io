"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Regret-based heuristic for flexible job-shop scheduling.
    
    For each ready operation, compute the regret: the difference between the
    best and second-best machine assignment costs. The cost of assigning an
    operation to a machine is estimated as:
      duration + estimated_future_load(machine)
    
    where estimated_future_load is approximated by the total remaining work
    that must be processed on that machine (a proxy for congestion).
    
    We select the operation with the maximum regret, as delaying a decision
    with high regret (i.e., where the best and second-best options differ
    significantly) is most likely to worsen the schedule.
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Compute total remaining work per machine across all ready operations
    # This serves as a proxy for future machine load/congestion
    machine_load = {}
    for op in ready_operations:
        for machine, duration in op["machine_options"].items():
            # Add the duration of this operation as a proxy for load
            # We also consider remaining_work as a signal of job urgency
            if machine not in machine_load:
                machine_load[machine] = 0.0
            machine_load[machine] += duration
    
    # For each operation, compute the regret
    best_costs = []
    for op in ready_operations:
        options = op["machine_options"]
        # Compute cost for each machine: duration + estimated future load
        costs = []
        for machine, duration in options.items():
            future_load = machine_load.get(machine, 0.0)
            cost = duration + 0.5 * future_load  # Weight future load
            costs.append(cost)
        
        if len(costs) <= 1:
            # Only one option, no regret
            regret = 0.0
        else:
            # Sort costs to find best and second-best
            sorted_costs = sorted(costs)
            regret = sorted_costs[1] - sorted_costs[0]
        
        best_costs.append(regret)
    
    # Select the operation with maximum regret
    # Tie-break: among tied regret values, prefer the one with more remaining work
    # and then the one with shorter minimum duration (urgency)
    max_regret = max(best_costs)
    
    # Find all operations with maximum regret
    candidates = [i for i in range(n_ops) if best_costs[i] == max_regret]
    
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
