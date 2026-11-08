"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track machine loads across dispatch calls
# This allows the heuristic to be aware of contention without modifying the interface
_machine_loads = {}

def dispatch(ready_operations, time):
    """
    Dispatch rule: Load-Aware Shortest Processing Time (SPT) with Machine Balancing.
    
    For each ready operation, we:
    1. Identify the eligible machine with the lowest current load.
    2. Estimate the completion time contribution as the duration on that machine.
    3. Select the operation with the minimum such estimated time.
    4. Return the index of the selected operation and the chosen machine.
    """
    if not ready_operations:
        return -1

    # Initialize machine loads for any new machines encountered
    # We assume machines are integers or hashable identifiers
    for op in ready_operations:
        for m in op['machine_options']:
            if m not in _machine_loads:
                _machine_loads[m] = 0.0

    best_index = -1
    best_machine = -1
    best_score = float('inf')

    # Deadline guard: Although the logic is O(N*M) which is very fast for 10 jobs/5 machines,
    # we include a guard as per directive.
    start_time = time.time()
    deadline = start_time + 1.0  # 1 second margin, well within 120s limit

    for i, op in enumerate(ready_operations):
        # Check deadline
        if time.time() > deadline:
            break

        # Find the machine with the lowest current load among options
        min_load_machine = -1
        min_load = float('inf')
        min_duration = float('inf')
        
        for m, dur in op['machine_options'].items():
            current_load = _machine_loads.get(m, 0.0)
            # Heuristic: We want to pick the machine that is least busy.
            # If loads are equal, pick the one with shortest duration.
            if current_load < min_load or (current_load == min_load and dur < min_duration):
                min_load = current_load
                min_duration = dur
                min_load_machine = m

        if min_load_machine != -1:
            # Score: Primary is duration on least-loaded machine.
            # Secondary tie-breaker: Prefer operations with more remaining work 
            # to ensure jobs don't stall, but SPT is primary for makespan.
            # Actually, for makespan, SPT is generally strong. 
            # Let's stick to SPT on the least loaded machine.
            score = min_duration
            
            # Optional: Small penalty if the machine is already very loaded relative to others?
            # For now, keep it simple: Minimize duration on least-loaded machine.
            
            if score < best_score or (score == best_score and op['remaining_work'] > ready_operations[best_index]['remaining_work'] if best_index != -1 else False):
                best_score = score
                best_index = i
                best_machine = min_load_machine

    if best_index == -1:
        # Fallback if no valid operation found or deadline hit early
        return 0

    # Update the load of the chosen machine
    # The duration added is the duration of the operation on that machine
    # Note: In a real simulation, this load would be "time until completion" or "queue length".
    # Since we don't know the exact simulation step size, we approximate load by adding the duration.
    # This is a heuristic proxy for "how busy is this machine".
    chosen_op = ready_operations[best_index]
    chosen_duration = chosen_op['machine_options'][best_machine]
    _machine_loads[best_machine] += chosen_duration

    # Return index and machine as tuple (index, machine)
    # The spec says: "return index of the ready operation to start (optionally (index, machine))"
    # Returning a tuple is safer if the evaluator expects machine selection.
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
