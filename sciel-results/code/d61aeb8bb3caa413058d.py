"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as time_module
import numpy as np

# Global state to track machine loads if we could persist it, but dispatch is stateless per call in the interface description.
# However, we can track machine congestion based on the current set of ready_operations.

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Calculate congestion per machine based on current ready operations
    machine_congestion = {}
    for op in ready_operations:
        # Assume the operation will be assigned to its most efficient machine (min duration) 
        # or we need to pick a machine. The interface says machine_options {machine: duration}.
        # Usually, in these tasks, the simulator picks the machine or we must return (index, machine).
        # The prompt says: "dispatch(ready_operations, time) -> index of the ready operation to start (optionally (index, machine))"
        # If we return just index, the simulator likely uses a default or first available.
        # If we return (index, machine), we control the routing.
        # Let's assume we can return (index, machine) to optimize routing.
        
        # Determine the best machine for this operation (shortest duration)
        best_machine = None
        best_dur = float('inf')
        for m, d in op["machine_options"].items():
            if d < best_dur:
                best_dur = d
                best_machine = m
        
        if best_machine not in machine_congestion:
            machine_congestion[best_machine] = 0
        machine_congestion[best_machine] += 1

    best_idx = 0
    best_machine = None
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        # Find the best machine for this operation considering current congestion
        # We want to minimize: duration + penalty * congestion_on_that_machine
        # This encourages spreading work across machines.
        
        min_score_for_op = float('inf')
        chosen_machine = None
        
        for m, d in op["machine_options"].items():
            cong = machine_congestion.get(m, 0)
            # Penalty factor: if a machine is congested, adding another op increases makespan risk.
            # Simple penalty: d + cong * 0.1 * d  (10% penalty per concurrent ready op on that machine)
            score = d * (1.0 + 0.1 * cong)
            
            if score < min_score_for_op:
                min_score_for_op = score
                chosen_machine = m
        
        # We also want to prioritize jobs with more remaining work to avoid starvation?
        # Or just minimize local delay?
        # Let's combine: Primary goal = Minimize local delay (score). Secondary = Maximize remaining work (tie-breaker).
        # Actually, pure load balancing (min score) is a strong structural change from MWRT.
        
        if min_score_for_op < best_score:
            best_score = min_score_for_op
            best_idx = i
            best_machine = chosen_machine
        elif min_score_for_op == best_score:
            # Tie-breaker: pick the one with more remaining work to keep jobs moving
            if op["remaining_work"] > ready_operations[best_idx]["remaining_work"]:
                best_idx = i
                best_machine = chosen_machine

    # Return index and machine to allow the simulator to route efficiently
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
