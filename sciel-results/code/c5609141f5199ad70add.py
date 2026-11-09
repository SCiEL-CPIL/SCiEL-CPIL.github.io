"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a load-balancing heuristic
    inspired by bipartite assignment principles.
    
    Strategy:
    1. For each ready operation, compute the best (shortest) machine option.
    2. Estimate the resulting load on each machine if we assign the operation
       to its best machine.
    3. Prefer operations that, when assigned to their best machine, result in
       the lowest maximum machine load (minimizing makespan proxy).
    4. Tie-break by remaining work (longer jobs first) to avoid starvation.
    """
    if not ready_operations:
        return 0

    # Extract machine options and remaining work for each operation
    n_ops = len(ready_operations)
    
    # For each operation, find the minimum duration and which machine(s) achieve it
    op_min_duration = []
    op_best_machines = []
    op_remaining_work = []
    
    for op in ready_operations:
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        min_dur = min(durations)
        op_min_duration.append(min_dur)
        
        # Find all machines with minimum duration
        best_machines = [m for m, d in machine_options.items() if d == min_dur]
        op_best_machines.append(best_machines)
        
        op_remaining_work.append(op["remaining_work"])
    
    op_min_duration = np.array(op_min_duration)
    op_remaining_work = np.array(op_remaining_work)
    
    # Collect all unique machines
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    all_machines = sorted(all_machines)
    n_machines = len(all_machines)
    
    # We don't have current machine loads directly, so we use a proxy:
    # Estimate load contribution. Since we only see ready operations, we
    # approximate by considering that assigning an op to a machine adds its
    # duration to that machine's load. We want to minimize the max load.
    #
    # Without actual current loads, we use a heuristic:
    # - Prefer operations with shorter min duration (they finish sooner, freeing up capacity)
    # - But also consider remaining work to ensure long jobs get started
    #
    # A better proxy: score = min_duration / (1 + remaining_work)
    # This favors operations that are both short and have a lot of remaining work
    # (i.e., critical path operations).
    #
    # Alternative: Use a combined score that balances:
    # 1. Short processing time on best machine (reduces makespan)
    # 2. High remaining work (ensures long jobs are prioritized)
    # 3. Flexibility (more machine options = more scheduling flexibility)
    
    # Compute flexibility: number of machines with minimum duration
    op_flexibility = [len(bm) for bm in op_best_machines]
    op_flexibility = np.array(op_flexibility, dtype=float)
    
    # Normalize values for scoring
    if op_min_duration.max() > op_min_duration.min():
        norm_dur = (op_min_duration - op_min_duration.min()) / (op_min_duration.max() - op_min_duration.min())
    else:
        norm_dur = np.zeros_like(op_min_duration, dtype=float)
    
    if op_remaining_work.max() > op_remaining_work.min():
        norm_work = (op_remaining_work - op_remaining_work.min()) / (op_remaining_work.max() - op_remaining_work.min())
    else:
        norm_work = np.zeros_like(op_remaining_work, dtype=float)
    
    if op_flexibility.max() > op_flexibility.min():
        norm_flex = (op_flexibility - op_flexibility.min()) / (op_flexibility.max() - op_flexibility.min())
    else:
        norm_flex = np.zeros_like(op_flexibility, dtype=float)
    
    # Composite score: we want to MINIMIZE makespan, so:
    # - Lower norm_dur is better (shorter ops reduce makespan)
    # - Higher norm_work is better (long jobs should be prioritized to avoid blocking)
    # - Higher norm_flex is better (more flexible ops can be scheduled later)
    #
    # Score = w1 * norm_dur + w2 * (1 - norm_work) + w3 * (1 - norm_flex)
    # We pick the operation with the LOWEST score.
    
    w1 = 0.4  # weight for processing time
    w2 = 0.4  # weight for remaining work (inverse)
    w3 = 0.2  # weight for flexibility (inverse)
    
    scores = w1 * norm_dur + w2 * (1 - norm_work) + w3 * (1 - norm_flex)
    
    # Pick the operation with the lowest score
    best_idx = int(np.argmin(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
