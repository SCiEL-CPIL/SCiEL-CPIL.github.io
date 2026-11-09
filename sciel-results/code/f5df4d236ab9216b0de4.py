"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Module-level state for machine load tracking
_machine_load = {}
_last_time = 0.0


def dispatch(ready_operations, time):
    global _machine_load, _last_time
    
    # Update machine loads based on elapsed time (simple decay/reset not needed,
    # we track cumulative assigned work)
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # For each ready operation, compute a composite priority score
    # and select the best (operation, machine) pair
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        job = op["job"]
        op_num = op["op"]
        machine_options = op["machine_options"]
        remaining_ops = op["remaining_ops"]
        remaining_work = op["remaining_work"]
        
        # Find the best machine for this operation
        best_machine_for_op = None
        best_machine_score = -float('inf')
        
        for machine, duration in machine_options.items():
            # Estimate machine load: use tracked load + this duration
            current_load = _machine_load.get(machine, 0.0)
            # Penalty for assigning to a busy machine
            # Use a smooth penalty: load / (load + duration) as congestion indicator
            congestion = current_load / (current_load + duration + 1e-9)
            
            # Score: prefer less congested machines, shorter durations
            machine_score = -congestion * duration - 0.1 * current_load
            
            if machine_score > best_machine_score:
                best_machine_score = machine_score
                best_machine_for_op = machine
        
        if best_machine_for_op is None:
            # Fallback: pick machine with shortest duration
            best_machine_for_op = min(machine_options, key=lambda m: machine_options[m])
            best_machine_score = -machine_options[best_machine_for_op]
        
        # Composite priority for this operation:
        # 1. Remaining work (higher = more urgent to start)
        # 2. Fewer remaining ops (closer to completion)
        # 3. Minimum duration on best machine (shorter = less blocking)
        # 4. Machine score (less congested = better)
        
        min_duration = min(machine_options.values())
        
        # Weighted composite score
        # Normalize-ish: remaining_work is in hours, remaining_ops is count
        score = (
            1.0 * remaining_work +          # primary: most work remaining
            2.0 * (5 - remaining_ops) +     # bonus for being later in job (closer to done)
            -0.5 * min_duration +            # slight preference for shorter ops
            0.5 * best_machine_score         # machine congestion penalty
        )
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op
    
    # Update machine load for the chosen machine
    chosen_op = ready_operations[best_idx]
    if best_machine is not None and best_machine in chosen_op["machine_options"]:
        _machine_load[best_machine] = _machine_load.get(best_machine, 0.0) + chosen_op["machine_options"][best_machine]
    else:
        # Fallback
        m = min(chosen_op["machine_options"], key=lambda m: chosen_op["machine_options"][m])
        _machine_load[m] = _machine_load.get(m, 0.0) + chosen_op["machine_options"][m]
    
    # Return (index, machine) to allow machine selection
    return (best_idx, best_machine) if best_machine is not None else best_idx
# EVOLVE-BLOCK-END
