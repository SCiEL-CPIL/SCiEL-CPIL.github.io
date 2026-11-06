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
    
    Strategy: Refined Model-and-optimize approach with Load-Aware Congestion.
    1. Calculate urgency based on remaining work.
    2. Estimate machine congestion using the sum of minimum durations of 
       other ready operations that can use that machine.
    3. Select operation-machine pair that maximizes a combined score of
       urgency-adjusted completion time and decision regret.
    
    Changes from previous version:
    - Replaced simple count-based contention with load-based congestion.
    - Increased sensitivity to machine load to avoid bottlenecks.
    - Tuned weights based on observed improvements from stronger congestion penalties.
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
    
    # Precompute machine load estimates:
    # For each machine, sum the minimum duration of all OTHER ready ops that can use it.
    # This approximates the queue length if we were to schedule them optimally (SPT order).
    machine_load = {}
    machine_min_durs = {} # store min dur for each op-machine pair for quick access
    
    # First pass: collect min durations for all op-machine pairs
    op_min_durs = []
    for op in ready_operations:
        opts = op["machine_options"]
        min_d = min(opts.values()) if opts else 0.0
        op_min_durs.append(min_d)
        
        for m in opts:
            if m not in machine_load:
                machine_load[m] = 0.0
            machine_load[m] += min_d

    # Now, for each operation i, the load on machine m EXCLUDING i is:
    # machine_load[m] - min_duration(i, m)
    # But we want the sum of min_durations of *other* ops.
    # Actually, machine_load[m] is sum of min_d(op_j, m) for all j where m in opts_j.
    # So load_excl_i = machine_load[m] - min_d(i, m).
    
    # Precompute per-operation features
    urgency_scores = np.empty(n, dtype=float)
    regret_scores = np.empty(n, dtype=float)
    critical_path_scores = np.empty(n, dtype=float)
    
    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 0.0))
        opts = op["machine_options"]
        
        # Urgency: Remaining work is the primary driver for makespan.
        urgency_scores[i] = rw + 0.1 * ro
        
        # Critical path estimate
        if opts:
            min_dur = min(float(d) for d in opts.values())
            critical_path_scores[i] = ro * min_dur
        else:
            critical_path_scores[i] = 0.0
            
        # Compute regret
        if opts and len(opts) >= 2:
            sorted_opts = sorted(opts.items(), key=lambda x: float(x[1]))
            best_cost = float(sorted_opts[0][1])
            second_cost = float(sorted_opts[1][1])
            best_m = sorted_opts[0][0]
            
            # Regret is the difference in cost.
            # We don't need contention factor here for the base regret, 
            # but we can scale it if the best machine is heavily loaded.
            load_best = machine_load.get(best_m, 0.0) - best_cost
            # If the best machine is very loaded, the regret of *not* using it might be higher?
            # Or rather, the penalty for using it is higher.
            # Let's keep regret as pure duration difference for now, 
            # and let the congestion term handle the load penalty.
            regret_scores[i] = second_cost - best_cost
        else:
            regret_scores[i] = 0.0

    # Now evaluate each operation-machine pair
    best_score = -float('inf')
    best_index = 0
    best_machine = None
    
    # Weight parameters
    URG_WEIGHT = 1.0
    REGRET_WEIGHT = 1.0  # Increased to prioritize resolving ambiguous choices
    CONTENT_WEIGHT = 0.8 # Load weight: how much the queue delays the operation
    CP_WEIGHT = 0.5      # Weight for critical path term
    
    for i, op in enumerate(ready_operations):
        urgency = urgency_scores[i]
        if urgency <= 0:
            urgency = 1e-6
        
        regret = regret_scores[i]
        cp = critical_path_scores[i]
        
        for m, d in op["machine_options"].items():
            d = float(d)
            if d <= 0:
                continue
            
            # Calculate load on machine m excluding this operation
            # machine_load[m] includes this op's min duration.
            # We need the sum of min durations of OTHER ops.
            # min_d(i, m) is d.
            load_excl_i = machine_load.get(m, 0.0) - d
            
            # Estimated completion time
            # d + CONTENT_WEIGHT * load_excl_i
            # This assumes the other operations will be processed roughly in parallel 
            # or with some overlap, but for a conservative estimate, we add their load.
            # Using CONTENT_WEIGHT < 1.0 assumes some efficiency or that not all 
            # "ready" ops will actually run on m immediately.
            est_completion = d + CONTENT_WEIGHT * load_excl_i
            
            # Score: higher is better
            score = URG_WEIGHT * urgency / (est_completion + 1e-9)
            score += REGRET_WEIGHT * regret
            score += CP_WEIGHT * cp / (est_completion + 1e-9)
            
            if score > best_score:
                best_score = score
                best_index = i
                best_machine = m
    
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
