import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE = 148.0  # Safety margin before 150s kill

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0, 0 # Fallback, shouldn't happen

    # Deadline guard: if we're running out of time, return the first operation on its first machine
    if _time.time() - _start_time > _DEADLINE:
        return 0, list(ready_operations[0]['machine_options'].keys())[0]

    # Calculate contention for each machine among ready operations
    machine_contention = {}
    for op in ready_operations:
        for m in op['machine_options']:
            machine_contention[m] = machine_contention.get(m, 0) + 1

    best_score = -float('inf')
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        # Constraint tightness: number of machine options (fewer = more constrained)
        num_options = len(op['machine_options'])
        
        # Use remaining_work as a measure of urgency/criticality
        remaining_work = op.get('remaining_work', 1.0)
        
        # Ensure we don't divide by zero or have negative work
        if remaining_work <= 0:
            remaining_work = 1.0
        
        # Constraint tightness factor: operations with fewer options are more critical
        # This prevents future starvation by prioritizing constrained operations
        constraint_factor = 1.0 + (5.0 - num_options) * 0.3  # Scale based on typical problem size

        for m, duration in op['machine_options'].items():
            if duration <= 0:
                continue
            
            contention = machine_contention.get(m, 0)
            
            # Heuristic: 
            # Numerator: How much work is left (urgency) * constraint tightness
            # Denominator: Time it takes on this machine, penalized by how busy the machine is
            # This favors operations with high remaining work and few options on fast, less busy machines.
            
            # The factor 0.5 for contention is a guess. 
            # Higher contention means more likely to be queued or delayed.
            load_factor = 1.0 + 0.5 * contention
            effective_duration = duration * load_factor
            
            # Score combines urgency, constraint tightness, and machine efficiency
            score = (remaining_work * constraint_factor) / effective_duration
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = m

    # Fallback if no valid option found
    if best_machine is None:
        return 0, list(ready_operations[0]['machine_options'].keys())[0]

    return best_idx, best_machine
