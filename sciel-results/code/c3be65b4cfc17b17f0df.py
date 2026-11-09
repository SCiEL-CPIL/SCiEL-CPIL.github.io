"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.optimize import linear_sum_assignment

# Deadline guard
_START_TIME = _time.time()
_DEADLINE = 120.0  # seconds, with safety margin

def _elapsed_ok():
    return (_time.time() - _START_TIME) < _DEADLINE

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using an assignment-based approach.
    
    Strategy: Model as bipartite assignment where we want to minimize the
    maximum machine completion time. Use a cost matrix that accounts for:
    1. Processing time on each machine
    2. Estimated current load on each machine
    3. Constraint tightness (operations with fewer options get priority)
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Deadline check
    if not _elapsed_ok():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Collect all machine options
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    machines = sorted(all_machines)
    m = len(machines)
    machine_idx = {mach: i for i, mach in enumerate(machines)}
    
    # Build processing time matrix: proc[i][j] = time for op i on machine j
    proc = np.zeros((n, m))
    valid_mask = np.zeros((n, m), dtype=bool)
    
    for i, op in enumerate(ready_operations):
        for mach, dur in op["machine_options"].items():
            j = machine_idx[mach]
            proc[i][j] = dur
            valid_mask[i][j] = True
    
    # Estimate machine loads: distribute remaining_work across machines that can process it
    # For each machine, estimate how much work is "assigned" to it
    machine_load = np.zeros(m)
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        # Count how many machines can process this operation
        num_options = sum(1 for mach in op["machine_options"])
        if num_options > 0:
            # Distribute remaining work evenly across eligible machines as a load estimate
            load_per_machine = rw / num_options
            for mach in op["machine_options"]:
                j = machine_idx[mach]
                machine_load[j] += load_per_machine
    
    # Build cost matrix: cost[i][j] = proc[i][j] + alpha * machine_load[j]
    # We want to minimize the maximum, so we use a weighted combination
    # Higher weight on machine load to balance
    alpha = 0.5
    cost = proc.copy()
    for j in range(m):
        cost[:, j] += alpha * machine_load[j]
    
    # For invalid entries, set high cost
    BIG_COST = 1e9
    cost[~valid_mask] = BIG_COST
    
    # Try to solve assignment problem
    if _elapsed_ok():
        try:
            # linear_sum_assignment minimizes total cost
            # We want to minimize max completion, but total cost is a good proxy
            row_ind, col_ind = linear_sum_assignment(cost)
            
            # Find which row (operation) is assigned to a valid column
            # We want to pick the operation that, when started, leads to best balance
            # Actually, we need to pick ONE operation to start now.
            # The assignment tells us the optimal matching, but we can only start one.
            
            # Strategy: Pick the operation that is assigned to the machine where it
            # has the lowest "regret" or where starting it now is most beneficial.
            
            # Alternative: For each operation, compute the best machine it's assigned to,
            # then pick the operation with the highest priority based on:
            # 1. Fewest alternative machines (constraint tightness)
            # 2. Longest processing time on its assigned machine
            # 3. Machine load balance
            
            # Let's use a scoring approach:
            scores = np.zeros(n)
            for i in range(n):
                # Find assigned machine for operation i
                assigned_j = None
                for r, c in zip(row_ind, col_ind):
                    if r == i:
                        assigned_j = c
                        break
                
                if assigned_j is not None and valid_mask[i, assigned_j]:
                    # Score based on:
                    # - Number of options (fewer = higher priority)
                    num_options = int(np.sum(valid_mask[i]))
                    option_score = 1.0 / max(num_options, 1)
                    
                    # - Processing time on assigned machine
                    proc_score = proc[i, assigned_j]
                    
                    # - Machine load (lower load machine = better to start here)
                    load_score = machine_load[assigned_j]
                    
                    # - Remaining work (more work = higher priority)
                    rw_score = ready_operations[i]["remaining_work"]
                    
                    # Combine: prioritize constrained ops, long processing, low load machines
                    scores[i] = (0.3 * option_score + 
                                0.3 * (proc_score / (proc.max() + 1e-9)) + 
                                0.2 * (1.0 - load_score / (machine_load.max() + 1e-9)) + 
                                0.2 * (rw_score / (np.max([op['remaining_work'] for op in ready_operations]) + 1e-9)))
                else:
                    # Fallback for invalid assignment
                    num_options = int(np.sum(valid_mask[i]))
                    scores[i] = 1.0 / max(num_options, 1)
            
            best_idx = int(np.argmax(scores))
            
        except Exception:
            # Fallback to simple heuristic
            best_idx = int(np.argmax([op["remaining_work"] for op in ready_operations]))
    else:
        # Deadline exceeded, use simple heuristic
        best_idx = int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    return best_idx
# EVOLVE-BLOCK-END
