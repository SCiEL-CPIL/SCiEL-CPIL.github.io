"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill

# Tuning parameters
_T_SOFT_MIN = 1.5  # Temperature for soft-min (machine selection)
_LAMBDA_CONG = 1.0 # Weight for congestion in machine cost
_ALPHA_BOTTLENECK = 1.0 # Scaling for bottleneck ratio


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a Dynamic Bottleneck Optimization Heuristic:
    1. Estimate machine congestion based on the number of ready operations eligible for each machine.
    2. For each operation, calculate a 'soft-min' of the effective cost (duration + congestion penalty)
       over its eligible machines. This represents the expected completion time if the operation
       were to start, considering load balancing.
    3. Calculate a 'Bottleneck Ratio' for each operation: Remaining Work / (Estimated Load on Eligible Machines).
       High ratio indicates a job that is critical and facing high congestion, so it should be prioritized
       to prevent further delays.
    4. Select the operation with the highest Bottleneck Ratio.
    5. Select the machine that minimizes the effective cost (duration + congestion penalty).
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Pre-compute features
    machine_options_list = []
    rws = []
    
    # For machine congestion analysis
    machine_congestion = {}
    
    for op in ready_operations:
        mw = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        rws.append(mw)
        machine_options_list.append(machine_options)
        
        if machine_options:
            for m in machine_options.keys():
                machine_congestion[m] = machine_congestion.get(m, 0) + 1

    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            # Find best machine considering congestion
            best_m = None
            best_cost = float('inf')
            for m, d in mo.items():
                cong = machine_congestion.get(m, 0)
                cost = d + _LAMBDA_CONG * cong
                if cost < best_cost:
                    best_cost = cost
                    best_m = m
            return (0, best_m)
        return 0

    # Calculate scores
    scores = []
    best_machines = []
    
    # We will compute the "Estimated Cumulative Load" for the denominator of the bottleneck ratio.
    # We approximate the load on a machine by the sum of min-durations of all ops eligible for it?
    # Or just the count? The count is a proxy for queue length.
    # Let's use the sum of minimum durations of all *other* ready ops eligible for the machine + the current op's duration?
    # That's complex. Let's stick to:
    # Denominator = Sum of (min_duration * congestion_count) for all machines eligible for this op?
    # No, simpler: 
    # Bottleneck Score = Remaining Work / (Sum of Effective Costs of eligible machines)
    # Where Effective Cost = duration + lambda * congestion.
    # If an op has many machines with low cost, its bottleneck is low.
    # If an op has few machines with high cost (congestion), its bottleneck is high.
    
    for i in range(n):
        mo = machine_options_list[i]
        rw = rws[i]
        
        if not mo:
            scores.append(0)
            best_machines.append(None)
            continue
            
        # Calculate effective costs for each machine
        costs = []
        machine_costs = []
        for m, d in mo.items():
            cong = machine_congestion.get(m, 0)
            cost = d + _LAMBDA_CONG * cong
            costs.append(cost)
            machine_costs.append((m, cost))
            
        if not costs:
            scores.append(0)
            best_machines.append(None)
            continue
            
        # Find the machine with the minimum effective cost
        min_cost = min(costs)
        best_m = None
        for m, c in machine_costs:
            if c == min_cost:
                best_m = m
                break
                
        # Calculate Soft-Min of costs (smooth surrogate for min)
        # This value represents the "expected" cost if we could probabilistically assign machines
        # but we are actually picking the argmin.
        # We use this to estimate the "load" contribution.
        # Actually, for the bottleneck ratio, we want to know how "congested" the options are.
        # If all options are expensive, the ratio should be high?
        # Let's define:
        # Load_Factor = Sum of exp(-cost / T) ? No.
        
        # Let's use the average cost as a proxy for the "efficiency" of the machine set.
        # Or better: The sum of costs?
        # If we use Sum(costs), an op with many machines has a high denominator -> low ratio -> lower priority?
        # That seems wrong. An op with many flexible options is *less* critical because it can go anywhere.
        # An op with only one busy machine is *more* critical.
        
        # So, we want the denominator to be SMALL when the op is critical.
        # Critical = High RW, Low Flexibility/High Congestion.
        
        # Let's try: Score = RW / (Min_Cost + Small_Epsilon)
        # If Min_Cost is high (congested), and RW is high, Score is... wait.
        # If Min_Cost is high, RW/Min_Cost is smaller. That would deprioritize it.
        # We want to PRIORITIZE it.
        
        # So maybe Score = RW * Min_Cost?
        # If RW is high and Cost is high (congestion), Score is very high.
        # This makes sense: "Urgent and Blocked" jobs get highest priority.
        
        # Let's refine:
        # Score = RW / (1 / Min_Cost) = RW * Min_Cost ?
        # Or Score = RW * (1 + Min_Cost)?
        
        # Let's look at the "Bottleneck Ratio" idea again.
        # Usually, Critical Path Length (RW) / Available Capacity.
        # If Available Capacity is low (High Congestion), the ratio is High.
        # So we want a denominator that represents "Available Capacity".
        # Low Cost = High Capacity?
        # So Denominator = 1 / Min_Cost ?
        # Then Score = RW / (1 / Min_Cost) = RW * Min_Cost.
        
        # Let's test this logic:
        # Job A: RW=100, Best Machine Cost=10 (Congested). Score = 1000.
        # Job B: RW=100, Best Machine Cost=1 (Free). Score = 100.
        # Job A is prioritized. This is good because Job A is blocked on a busy machine.
        
        # What if Job C: RW=10, Best Machine Cost=100 (Very Congested). Score = 1000.
        # Job C ties Job A. But Job A has more work.
        # Maybe we should weight RW more?
        
        # Let's use: Score = RW * Min_Cost
        # But we also need to consider that if an op has *many* options, it's less critical.
        # The Min_Cost already captures the *best* option.
        
        # Let's add a flexibility penalty?
        # If an op has 5 machines, the "effective" congestion is lower because it can switch.
        # But we are picking the *best* machine.
        
        # Let's stick to Score = RW * Min_Cost for now, but normalize?
        # Actually, let's use the Soft-Min value we calculated earlier as the "Effective Cost".
        # Soft-Min is smoother.
        
        # Calculate Soft-Min
        sum_exp = 0.0
        for c in costs:
            sum_exp += math.exp(-(c - min_cost) / _T_SOFT_MIN)
        soft_min_val = min_cost - _T_SOFT_MIN * math.log(sum_exp)
        
        # Score = RW * Soft_Min_Val
        # This rewards high remaining work and high effective cost (congestion).
        
        score = rw * soft_min_val
        scores.append(score)
        best_machines.append(best_m)

    # Find best
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
            
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END
