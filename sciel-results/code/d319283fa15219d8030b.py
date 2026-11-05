"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    best_index = 0
    best_score = -float('inf')
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        # Find the machine with the shortest duration for this operation
        min_duration = float('inf')
        min_machine = None
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
                min_machine = m
        
        # Heuristic Score:
        # 1. Prioritize jobs with more remaining work (urgency)
        # 2. Prefer operations with shorter processing times (efficiency)
        # 3. Slight preference for operations with fewer machine options (constraint)
        # Weights are tuned to balance urgency and efficiency.
        num_options = len(machine_options)
        
        # Normalize remaining_work slightly to prevent it from dominating too much if values are large
        # But since we don't know the scale, we use a linear combination.
        score = remaining_work * 10.0 - min_duration * 1.0 - num_options * 0.1
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = min_machine
    
    # Return the index and the chosen machine to allow for optimal machine selection
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
