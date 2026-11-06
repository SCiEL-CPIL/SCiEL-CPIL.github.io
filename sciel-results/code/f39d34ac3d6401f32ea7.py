"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE = 115.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.
    
    Strategy: Regret-based heuristic with urgency weighting.
    1. For each ready operation, compute the cost of assigning it to each eligible machine.
       Cost includes duration and an estimate of future machine load (contention).
    2. Compute the "regret" for each operation: the difference between its best and
       second-best machine assignment costs. High regret means the machine choice
       matters a lot, so we should decide it now to avoid worsening the schedule.
    3. Weight the regret by job urgency (remaining work) to balance between
       making high-stakes machine decisions and prioritizing jobs with more work left.
    4. Select the operation with the highest weighted regret, and assign it to its
       best machine.
    
    This approach prioritizes decisions where the choice of machine has the highest
    potential to worsen the schedule if delayed, rather than just prioritizing the
    most urgent or shortest job.
    """
    if _elapsed() > _DEADLINE:
        # Hard safety fallback: Shortest Processing Time (SPT)
        if not ready_operations:
            return 0
        best_i = 0
        best_d = float('inf')
        for i, op in enumerate(ready_operations):
            opts = op["machine_options"]
            if not opts:
                continue
            min_d = min(opts.values())
            if min_d < best_d:
                best_d = min_d
                best_i = i
        return best_i

    if not ready_operations:
        return 0
    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    n = len(ready_operations)
    
    # Precompute machine contention: count how many ready ops can use each machine
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_count[m] = machine_count.get(m, 0) + 1
    
    # For each operation, compute the cost of each machine assignment and find
    # the best and second-best costs to compute regret.
    
    # Cost model: duration * contention_factor
    # contention_factor = 1 + CONTENT_WEIGHT * (count - 1)
    # This approximates the delay due to other operations competing for the same machine.
    
    CONTENT_WEIGHT = 0.3
    
    # We'll store for each operation:
    #   best_cost, best_machine, second_best_cost, urgency
    # Then compute weighted_regret = urgency * (second_best_cost - best_cost)
    
    best_score = -float('inf')
    best_index = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 0.0))
        opts = op["machine_options"]
        
        if not opts:
            continue
        
        # Urgency: combine remaining work and operations
        urgency = rw + 0.1 * ro
        if urgency <= 0:
            urgency = 1e-6
        
        # Compute cost for each machine
        costs = []
        for m, d in opts.items():
            d = float(d)
            if d <= 0:
                continue
            count = machine_count.get(m, 1)
            contention_factor = 1.0 + CONTENT_WEIGHT * (count - 1)
            cost = d * contention_factor
            costs.append((cost, m))
        
        if not costs:
            continue
        
        # Sort by cost to find best and second-best
        costs.sort(key=lambda x: x[0])
        best_cost, best_m = costs[0]
        
        if len(costs) >= 2:
            second_best_cost = costs[1][0]
        else:
            # Only one machine option: regret is high (no alternative)
            # Use a penalty to encourage early decision for single-option ops
            second_best_cost = best_cost * 2.0 + 1.0
        
        regret = second_best_cost - best_cost
        
        # Weighted regret: higher urgency and higher regret -> higher priority
        weighted_regret = urgency * regret
        
        if weighted_regret > best_score:
            best_score = weighted_regret
            best_index = i
            best_machine = best_m
    
    # Fallback if no valid score was found
    if best_machine is None:
        best_index = 0
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_machine = min(opts, key=lambda k: opts[k])
        else:
            best_machine = None
    
    if best_machine is not None:
        return (best_index, best_machine)
    return best_index
# EVOLVE-BLOCK-END
