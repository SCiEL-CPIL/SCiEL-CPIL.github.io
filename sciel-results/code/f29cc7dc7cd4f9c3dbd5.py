import numpy as np

def dispatch(ready_operations, time):
    """
    Dispatches the next operation to start in an open-shop simulation.
    
    Strategy:
    1. Job Selection: Most Work Remaining (MWR).
       - Select the operation with the highest `remaining_work`.
       - Tie-breaker: Shortest Minimum Duration (SPT-like) to clear machines quickly.
    2. Machine Selection: Shortest Processing Time (SPT).
       - Select the machine from `machine_options` with the minimum duration.
    
    Returns:
        (index, machine): The index of the operation and the selected machine.
    """
    if not ready_operations:
        # Fallback if no operations are ready, though the simulator should handle this.
        return 0

    # 1. Find the operation with the maximum remaining work
    max_work = -1.0
    candidates = []
    
    for i, op in enumerate(ready_operations):
        work = op.get("remaining_work", 0)
        if work > max_work:
            max_work = work
            candidates = [i]
        elif work == max_work:
            candidates.append(i)
            
    # 2. If there are multiple candidates, break the tie using the shortest minimum duration
    best_idx = candidates[0]
    best_min_dur = float('inf')
    
    for i in candidates:
        op = ready_operations[i]
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
        min_dur = min(machine_options.values())
        if min_dur < best_min_dur:
            best_min_dur = min_dur
            best_idx = i
        elif min_dur == best_min_dur:
            # Further tie-breaker: Fewer machine options (more constrained)
            # Or just keep the first one. Let's keep the first one found with min_dur.
            pass

    # 3. Select the machine for the chosen operation using SPT (shortest duration)
    chosen_op = ready_operations[best_idx]
    machine_options = chosen_op.get("machine_options", {})
    
    if not machine_options:
        # Fallback: return just the index if no machine options (should not happen)
        return best_idx
        
    best_machine = min(machine_options, key=machine_options.get)
    
    return (best_idx, best_machine)
