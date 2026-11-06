"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

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
    
    Strategy: Hybrid Regret and Urgency Heuristic
    1. Calculate machine contention for all ready operations.
    2. For each operation, compute the 'regret' of the machine choice:
       the difference between the best and second-best estimated completion times.
       Estimated completion time accounts for contention (queueing delay).
    3. Calculate an 'urgency' score based on the average remaining work per 
       remaining operation (remaining_work / (1 + remaining_ops)). This 
       prioritizes jobs with large remaining chunks of work.
    4. Select the operation with the highest combined score:
       score = regret + URG_WEIGHT * urgency.
    5. Assign the selected operation to its best (lowest cost) machine.
    
    This approach prioritizes decisions where the machine choice is critical 
    (high regret) while using urgency to break ties and guide selection when 
    regret is low, favoring jobs that will keep machines busy for longer.
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
    
    URG_WEIGHT = 0.1
    CONTENT_WEIGHT = 0.4
    
    best_score = -float('inf')
    best_index = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            continue
        
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 0.0))
        
        # Urgency: Average remaining work per remaining operation
        # This favors jobs with large remaining work relative to their remaining steps
        if ro > 0:
            urgency = rw / (1.0 + ro)
        else:
            urgency = rw
        
        # Compute costs for all machines
        machine_costs = []
        for m, d in opts.items():
            d = float(d)
            if d <= 0:
                continue
            
            count = machine_count.get(m, 1)
            # Contention factor: approximate queueing delay
            # If count is 1, factor is 1.0. If count is 2, factor is 1.4, etc.
            contention_factor = 1.0 + CONTENT_WEIGHT * (count - 1)
            est_completion = d * contention_factor
            
            machine_costs.append((est_completion, m))
        
        if not machine_costs:
            continue
        
        # Sort by estimated completion time
        machine_costs.sort(key=lambda x: x[0])
        best_cost, best_m = machine_costs[0]
        
        # Calculate regret: difference between second-best and best cost
        if len(machine_costs) >= 2:
            second_cost, _ = machine_costs[1]
            regret = second_cost - best_cost
        else:
            regret = 0.0
        
        # Combined score: Regret drives the decision, Urgency breaks ties and guides low-regret cases
        score = regret + URG_WEIGHT * urgency
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m
    
    # Fallback if no valid score was found
    if best_machine is None:
        best_index = 0
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_machine = min(opts, key=lambda k: float(opts[k]))
        else:
            best_machine = None
    
    if best_machine is not None:
        return (best_index, best_machine)
    return best_index
# EVOLVE-BLOCK-END
