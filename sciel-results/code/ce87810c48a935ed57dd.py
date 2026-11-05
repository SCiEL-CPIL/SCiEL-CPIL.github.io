"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s; we want to be safe.
_DEADLINE = 145.0  # seconds wall-clock budget
_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _time.time() - _START_TIME


def _soft_min(values, beta=20.0):
    """Smooth minimum via log-sum-exp trick."""
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    arr = np.array(values, dtype=np.float64)
    # For soft min: -log(sum(exp(-beta*x)))/beta
    # Numerically stable: subtract max of -beta*x = -beta*min(x)
    m = np.min(arr)
    exps = np.exp(-beta * (arr - m))
    return m - np.log(np.sum(exps)) / beta


def _soft_max(values, beta=20.0):
    """Smooth maximum via log-sum-exp trick."""
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    arr = np.array(values, dtype=np.float64)
    m = np.max(arr)
    exps = np.exp(beta * (arr - m))
    return m + np.log(np.sum(exps)) / beta


def dispatch(ready_operations, time):
    """
    Dispatch rule: regret-based heuristic with smooth critical-path estimate.
    
    For each ready operation, compute:
      1. Best and second-best machine assignment cost (duration + estimated future load)
      2. Regret = best_cost - second_best_cost (higher regret = more urgent to decide)
      3. Critical path estimate via soft-max over remaining operations
      4. Combined score = urgency * regret + critical_path_weight
    
    Returns the index of the operation to start next.
    """
    if len(ready_operations) == 1:
        return 0
    
    # Safety: if we're close to deadline, just pick argmax remaining_work
    if _elapsed() > _DEADLINE - 2.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    scores = np.zeros(n_ops)
    
    # Pre-compute machine load estimates from remaining_work distribution
    # We'll estimate each machine's future load as proportional to total remaining work
    total_remaining_work = sum(op["remaining_work"] for op in ready_operations)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        machines = list(machine_options.keys())
        durations = [machine_options[m] for m in machines]
        
        if len(machines) <= 1:
            # Only one machine option; no regret from machine choice
            best_cost = durations[0] if durations else 0
            second_best_cost = best_cost  # no difference
            regret = 0.0
        else:
            # Estimate future machine load: assume remaining work is distributed
            # across machines roughly proportionally. For a rough estimate,
            # use total_remaining_work / num_machines as average load per machine.
            # But we don't know num_machines globally, so use the number of
            # machines in this op's options as a proxy, or just use a fixed estimate.
            # Simpler: estimate load as total_remaining_work / (some machine count).
            # We'll use a heuristic: each machine has roughly total/n_machines load.
            # Since we don't have global machine count, use len(machines) as proxy
            # or just use the average duration as a baseline.
            
            # Better approach: the "cost" of assigning to machine m is:
            #   duration_m + estimated_queue_time_on_m
            # We estimate queue time as proportional to total remaining work
            # that could be assigned to that machine. Without more info,
            # we approximate: queue_m ≈ total_remaining_work * (duration_m / sum(durations))
            # This is rough but captures the idea that longer-duration machines
            # accumulate more work.
            
            sum_durations = sum(durations)
            if sum_durations > 0:
                est_loads = [total_remaining_work * (d / sum_durations) for d in durations]
            else:
                est_loads = [0.0] * len(machines)
            
            costs = [durations[j] + est_loads[j] for j in range(len(machines))]
            
            # Sort costs to find best and second-best
            sorted_costs = sorted(costs)
            best_cost = sorted_costs[0]
            second_best_cost = sorted_costs[1] if len(sorted_costs) > 1 else best_cost
            regret = second_best_cost - best_cost
        
        # Critical path estimate: use soft-max over remaining operations' work
        # to get a smooth measure of the longest remaining path.
        remaining_works = [op_j["remaining_work"] for op_j in ready_operations]
        critical_path = _soft_max(remaining_works, beta=10.0)
        
        # Also consider this specific operation's remaining work as urgency
        urgency = op["remaining_work"]
        
        # Combine: high regret means the machine choice is critical;
        # high urgency/critical path means the job is on the critical path.
        # Weight regret heavily when there are multiple machine options.
        score = urgency * (1.0 + regret) + 0.1 * critical_path
        
        scores[i] = score
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
