"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin before 150s kill

def _deadline_exceeded():
    return _time.monotonic() >= _DEADLINE

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def _soft_min(values, alpha=2.0):
    """Compute soft-min using log-sum-exp trick for smooth approximation of min."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 1.0
    if len(values) == 1:
        return values[0]
    # Soft min: -1/alpha * log(sum(exp(-alpha * x)))
    neg_alpha_vals = -alpha * values
    max_neg = np.max(neg_alpha_vals)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_alpha_vals - max_neg)))
    return -log_sum / alpha

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling using dynamic bottleneck identification.
    Selects the operation whose assignment to its least-loaded machine maximizes
    the minimum slack time across all currently ready operations.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _deadline_exceeded():
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, collect machine durations
    all_durations = []
    for op in ready_operations:
        if op["machine_options"]:
            all_durations.append(list(op["machine_options"].values()))
        else:
            all_durations.append([1.0])
    
    # Compute min, max, mean, and soft-min durations
    min_durations = np.array([min(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    max_durations = np.array([max(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    mean_durations = np.array([np.mean(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    
    # Soft-min for smooth duration estimation
    soft_min_durations = np.array([
        _soft_min(np.array(d, dtype=np.float64), alpha=3.0) if d else 1.0 
        for d in all_durations
    ], dtype=np.float64)

    # WSPT ratio: remaining_work / soft_min_duration
    wspt = remaining_work / np.maximum(soft_min_durations, 1e-9)
    
    # Normalize all features
    n_wspt = _normalize(wspt)
    nw = _normalize(remaining_work)
    no = _normalize(remaining_ops)
    nd_min = _normalize(min_durations)
    nd_max = _normalize(max_durations)
    nd_mean = _normalize(mean_durations)
    nd_soft = _normalize(soft_min_durations)

    # Dynamic bottleneck identification:
    # For each candidate operation i, estimate the "slack" it provides to other operations.
    # The idea: if we start op i, it occupies a machine for duration_i. Other ops that
    # could have used that machine are delayed. We want to pick the op that maximizes
    # the minimum slack across all other ready ops, preventing machine starvation.
    
    # Estimate machine load: for each machine, estimate how much work is pending.
    # We approximate by counting how many ready ops can use each machine and their durations.
    machine_load = {}
    for i, op in enumerate(ready_operations):
        if op["machine_options"]:
            for machine, dur in op["machine_options"].items():
                if machine not in machine_load:
                    machine_load[machine] = 0.0
                machine_load[machine] += dur * remaining_work[i] / max(remaining_ops[i], 1)
    
    # For each candidate i, compute the bottleneck score:
    # If we start op i on its best (shortest) machine, what is the impact on others?
    # We estimate the "slack" provided by starting i: it's the difference between
    # the projected completion of i and the earliest time other ops could start on
    # the same machine.
    
    # Simplified: for each candidate i, compute a "bottleneck relief" score:
    #   - How much does starting i reduce the maximum machine load?
    #   - How much slack does it create for other ops?
    
    # For each candidate i, estimate the machine it would use (the one with min duration)
    # and compute how much that machine is loaded by other ops.
    bottleneck_scores = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if not ready_operations[i]["machine_options"]:
            bottleneck_scores[i] = 0.0
            continue
        
        # Find the machine with minimum duration for op i
        best_machine = None
        best_dur = float('inf')
        for machine, dur in ready_operations[i]["machine_options"].items():
            if dur < best_dur:
                best_dur = dur
                best_machine = machine
        
        if best_machine is None:
            bottleneck_scores[i] = 0.0
            continue
        
        # Estimate the load on best_machine from other ops (excluding i)
        load_from_others = 0.0
        count_others = 0
        for j in range(n):
            if j == i:
                continue
            if ready_operations[j]["machine_options"] and best_machine in ready_operations[j]["machine_options"]:
                load_from_others += ready_operations[j]["machine_options"][best_machine]
                count_others += 1
        
        # The "slack" provided by starting i: if we start i now, it completes at
        # time + best_dur. Other ops on the same machine can start after that.
        # The bottleneck is the maximum load on any machine. Starting i on a
        # less-loaded machine is better.
        
        # Score: higher is better. We want to pick ops that:
        # 1. Have high urgency (WSPT)
        # 2. Use a machine that isn't already heavily loaded by others
        # 3. Have short duration (SPT) to free up the machine quickly
        
        # Normalize the load from others
        if count_others > 0:
            avg_load_from_others = load_from_others / count_others
        else:
            avg_load_from_others = 0.0
        
        # The "slack" is inversely related to the load from others
        # If load_from_others is high, starting i creates more conflict
        # If load_from_others is low, starting i is less disruptive
        
        # Combine with WSPT: high WSPT + low load = good
        # Normalize avg_load_from_others to [0, 1]
        all_loads = []
        for j in range(n):
            if ready_operations[j]["machine_options"]:
                for m, d in ready_operations[j]["machine_options"].items():
                    all_loads.append(d)
        if all_loads:
            max_load = max(all_loads)
            min_load = min(all_loads)
            if max_load > min_load:
                norm_load = (avg_load_from_others - min_load) / (max_load - min_load)
            else:
                norm_load = 0.5
        else:
            norm_load = 0.5
        
        # bottleneck_score = WSPT * (1 - normalized_load) + SPT_bonus
        bottleneck_scores[i] = wspt[i] * (1.0 - norm_load) + (1.0 - nd_soft[i]) * 0.3
    
    # Normalize bottleneck scores
    n_bottleneck = _normalize(bottleneck_scores)
    
    # Combine the original heuristic score with the bottleneck score
    # The bottleneck score captures machine starvation prevention
    # The original score captures urgency and SPT
    
    w_wspt = 0.35      # primary urgency signal
    w_work = 0.10      # secondary urgency
    w_ops = 0.08       # keep in motion
    w_spt = 0.15       # shortest processing time
    w_range = 0.07     # penalize high max duration
    w_bottleneck = 0.25  # dynamic bottleneck relief
    
    score = (w_wspt * n_wspt + 
             w_work * nw + 
             w_ops * no + 
             w_spt * (1.0 - nd_soft) + 
             w_range * (1.0 - nd_max) +
             w_bottleneck * n_bottleneck)

    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
