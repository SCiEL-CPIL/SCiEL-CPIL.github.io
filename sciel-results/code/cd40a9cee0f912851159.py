import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Pre-compute machine usage frequency in the current ready set
    # to help with load balancing / contention estimation
    machine_counts = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_counts[m] = machine_counts.get(m, 0) + 1

    best_idx = -1
    best_score = -1
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            continue
            
        # Calculate flexibility (number of machine options)
        flexibility = len(machine_options)
        
        # Find the minimum duration and the machines that offer it
        min_duration = float('inf')
        best_machines = []
        
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
                best_machines = [m]
            elif d == min_duration:
                best_machines.append(m)
        
        # If no machines found (should not happen if machine_options not empty)
        if not best_machines:
            continue

        # Heuristic: Most Constrained First + Shortest Processing Time
        # Prioritize operations with fewer machine choices (higher constraint)
        # and shorter processing times.
        # score = 1 / (flexibility * (min_duration + epsilon))
        # Using +1.0 to avoid division by zero and smooth the curve.
        
        score = 1.0 / (float(flexibility) * (float(min_duration) + 1.0))
        
        if score > best_score:
            best_score = score
            best_idx = i
            
            # Select machine among those with min_duration
            # Prefer machine with lowest contention (fewest other ready ops can use it)
            min_contention = float('inf')
            selected_machine = None
            
            for m in best_machines:
                contention = machine_counts.get(m, 0)
                if contention < min_contention:
                    min_contention = contention
                    selected_machine = m
                elif contention == min_contention:
                    # Tie-breaker: lower machine ID
                    if selected_machine is None or m < selected_machine:
                        selected_machine = m
            
            best_machine = selected_machine

    if best_idx == -1:
        return 0

    return (best_idx, best_machine)
