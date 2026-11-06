"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    _start = _time.perf_counter()
    _deadline = _start + 115.0  # 115s safety margin (evaluator kills at 150s)

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        # Deadline guard: if we're running out of time, return best so far
        if _time.perf_counter() > _deadline:
            break

        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})

        if not machine_options:
            score = 0.0
            best_machine_for_op = None
        else:
            # Extract durations and machines
            machines = list(machine_options.keys())
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            
            # Avoid issues with very small values
            durations = np.maximum(durations, 1e-6)
            
            # Use soft-min via log-sum-exp for smooth surrogate of min duration
            # A larger beta makes the soft-min closer to the true min
            beta = 10.0
            exp_neg = np.exp(-beta * durations)
            # Handle potential overflow/underflow by subtracting max
            max_exp_neg = np.max(exp_neg)
            if max_exp_neg > 0:
                exp_neg_normalized = exp_neg / max_exp_neg
                soft_min = -np.log(np.sum(exp_neg_normalized)) / beta + np.log(max_exp_neg) / beta
            else:
                soft_min = -np.log(np.sum(exp_neg)) / beta
            
            # Track the actual min machine for tie-breaking
            min_idx = int(np.argmin(durations))
            best_machine_for_op = machines[min_idx]
            
            # Heuristic: Prioritize operations with high remaining work
            # that can be completed quickly. Use soft_min as a smooth proxy for min_duration.
            # Add urgency bonus for operations with fewer remaining ops
            score = remaining_work / soft_min
            if remaining_ops > 0:
                score += 0.15 * (1.0 / remaining_ops)
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_machine_for_op

    # Return index and optionally the machine
    if best_machine is not None:
        return (int(best_index), best_machine)
    else:
        return int(best_index)
# EVOLVE-BLOCK-END
