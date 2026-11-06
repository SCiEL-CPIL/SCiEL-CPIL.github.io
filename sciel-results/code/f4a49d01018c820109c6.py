"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Global cache for machine loads and other state that persists across dispatch calls
# We use a simple module-level dict to track machine completion times
_machine_state = {}
_dispatch_call_count = 0


def _estimate_machine_load(machine, current_time, ready_operations):
    """Estimate the load on a machine based on known remaining work."""
    # Base load is the current time if machine is free, or its last completion
    # We don't have direct access to machine availability, so we estimate
    # from the ready operations that could be assigned to this machine
    load = current_time
    for op in ready_operations:
        if machine in op["machine_options"]:
            # This operation could use this machine, add its duration as potential load
            # Weight by how likely it is to be scheduled here (simplification: just add)
            load += op["machine_options"][machine] * 0.5
    return load


def dispatch(ready_operations, time):
    """
    Dispatch rule using a regret-based heuristic with look-ahead.
    
    For each ready operation, compute the 'regret' of delaying its assignment:
    - For each operation, find the best and second-best machine assignments
    - Regret = difference between second-best and best completion times
    - Also factor in urgency (remaining work) and machine load imbalance
    
    Returns: index of the ready operation to start next
    """
    global _dispatch_call_count
    _dispatch_call_count += 1
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = []
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        if not machine_options:
            scores.append(0.0)
            continue
        
        # Get all machine options sorted by duration
        machines = list(machine_options.keys())
        durations = [machine_options[m] for m in machines]
        
        if len(machines) == 1:
            # Only one option, no regret, score based on urgency
            best_duration = durations[0]
            # Urgency: more remaining work = more urgent
            urgency = remaining_work
            # Load factor: estimate machine load
            est_load = _estimate_machine_load(machines[0], time, ready_operations)
            completion = est_load + best_duration
            score = urgency * 2.0 + completion * 0.5
            scores.append(score)
            continue
        
        # Sort machines by duration to find best and second-best
        sorted_machines = sorted(zip(machines, durations), key=lambda x: x[1])
        best_machine, best_duration = sorted_machines[0]
        second_machine, second_duration = sorted_machines[1]
        
        # Estimate completion times for best and second-best
        est_best_load = _estimate_machine_load(best_machine, time, ready_operations)
        est_second_load = _estimate_machine_load(second_machine, time, ready_operations)
        
        best_completion = est_best_load + best_duration
        second_completion = est_second_load + second_duration
        
        # Regret: how much worse it gets if we delay this decision
        # Higher regret means the choice between machines is more critical
        regret = second_completion - best_completion
        
        # Also consider the absolute urgency
        urgency = remaining_work
        # Penalty for having many remaining operations (longer tail)
        tail_penalty = remaining_ops * 0.1
        
        # Combine: regret is the primary driver, urgency is secondary
        # We want to prioritize operations where the machine choice matters most
        # AND where the job is urgent
        score = regret * 3.0 + urgency * 1.0 + tail_penalty
        
        # Bonus for operations with short best duration (can start sooner)
        score += (1.0 / (best_duration + 1.0)) * 5.0
        
        scores.append(score)
    
    # Find the operation with the highest score
    max_score = max(scores)
    best_index = scores.index(max_score)
    
    return int(best_index)
# EVOLVE-BLOCK-END
