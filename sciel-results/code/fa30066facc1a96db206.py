"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math


def dispatch(ready_operations, time):
    """
    Dispatch rule using Weighted Shortest Job First (WSJF) approximation.
    
    Heuristic:
    Priority = Remaining Operations / Minimum Duration of Current Operation
    
    Rationale:
    - Jobs with more remaining operations are more likely to be on the critical path
      and should be prioritized to avoid tail effects.
    - Operations with shorter minimum processing times should be started first to
      quickly free up machines and allow subsequent operations to begin.
    - This ratio balances job urgency (complexity) with immediate throughput.
    
    Additionally, we select the machine with the minimum duration for the chosen
    operation to minimize the immediate makespan contribution.
    """
    if not ready_operations:
        return 0
    
    best_idx = 0
    best_score = float('-inf')
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        if not machine_options:
            continue
            
        # Get the minimum processing time among all eligible machines
        min_duration = min(machine_options.values())
        
        # Avoid division by zero
        safe_min_duration = max(min_duration, 0.001)
        safe_remaining_ops = max(remaining_ops, 1)
        
        # WSJF Score: Higher is better
        # We want to prioritize jobs with many remaining ops (urgent)
        # and operations that are short (fast to clear).
        score = safe_remaining_ops / safe_min_duration
        
        if score > best_score:
            best_score = score
            best_idx = i
            # Store the machine that provides this minimum duration
            best_machine = min(machine_options, key=machine_options.get)
    
    # Return tuple (index, machine) to explicitly choose the best machine
    return (int(best_idx), best_machine)
# EVOLVE-BLOCK-END
