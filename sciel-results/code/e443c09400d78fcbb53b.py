import math
import time as time_module

_DEADLINE = 150.0
_START_TIME = time_module.time()
_TIME_MARGIN = 2.0

def _elapsed():
    return time_module.time() - _START_TIME

def _remaining():
    return _DEADLINE - _TIME_MARGIN - _elapsed()

def _lse2(vals):
    """Smooth max via log-sum-exp (base 2 for numerical stability)."""
    if not vals:
        return 0.0
    m = max(vals)
    if m == 0:
        return 0.0
    s = sum(math.exp(0.693147 * (v - m)) for v in vals)
    return m + math.log(s) / 0.693147

def _count_loads(ready_operations):
    """Count how many ready ops use each machine (as a proxy for load)."""
    loads = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            loads[m] = loads.get(m, 0) + 1
    return loads

def _est_finish(op, loads):
    """Estimate earliest finish time for op given current load counts.
    We assume each machine has a base availability proportional to its load.
    Use the machine that minimizes estimated finish time."""
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0
    best = float('inf')
    for m, d in mo.items():
        # Estimated availability: proportional to load count
        # Higher load means machine is busier
        est_avail = loads.get(m, 0) * 1.0  # unit time per concurrent op
        finish = est_avail + d
        if finish < best:
            best = finish
    return best

def _score(ready_operations, loads):
    """Score each operation: lower is better (minimizes estimated makespan).
    For each op, estimate the resulting load vector and compute smooth max."""
    n = len(ready_operations)
    scores = [0.0] * n
    
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            scores[i] = float('inf')
            continue
        
        # Estimate finish time for this op
        est_f = _est_finish(op, loads)
        
        # Estimate resulting loads if this op is scheduled
        # The op occupies its best machine, increasing that machine's load
        # For simplicity, assume it uses the machine that minimizes finish
        best_m = None
        best_f = float('inf')
        for m, d in mo.items():
            est_avail = loads.get(m, 0) * 1.0
            finish = est_avail + d
            if finish < best_f:
                best_f = finish
                best_m = m
        
        # New load vector: increment best_m
        new_loads = dict(loads)
        new_loads[best_m] = new_loads.get(best_m, 0) + 1
        
        # Estimate makespan: smooth max over all machines of (load * avg_duration)
        # We don't know avg_duration per machine, so use load as proxy
        # Better: estimate each machine's completion time as load * (typical duration)
        # Use the min duration in that machine's ops as a proxy for "work per op"
        
        # Compute estimated completion time for each machine
        machine_completions = []
        for m, cnt in new_loads.items():
            # Estimate: cnt ops * average duration on this machine
            # We don't have per-machine duration stats, so use a global average
            # Or just use cnt as a proxy (assuming unit durations)
            # Better: use the sum of durations of ops that could use this machine
            # But that's complex. For now, use cnt * 1.0
            machine_completions.append(cnt * 1.0)
        
        # Smooth max as makespan estimate
        makespan_est = _lse2(machine_completions) if machine_completions else 0.0
        
        # Also factor in the op's own remaining work (higher is worse if it delays others)
        rw = op.get("remaining_work", 0.0)
        
        # Score: lower is better
        # Combine makespan estimate with a small penalty for long remaining work
        scores[i] = makespan_est + 0.1 * rw
    
    return scores

def _lookahead_score(ready_operations, loads):
    """Two-step lookahead: for each op, schedule it, then greedily schedule the next best op,
    and estimate the resulting makespan."""
    n = len(ready_operations)
    scores = [0.0] * n
    
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            scores[i] = float('inf')
            continue
        
        # Schedule op i: find best machine
        best_m = None
        best_f = float('inf')
        for m, d in mo.items():
            est_avail = loads.get(m, 0) * 1.0
            finish = est_avail + d
            if finish < best_f:
                best_f = finish
                best_m = m
        
        # New loads after scheduling op i
        new_loads = dict(loads)
        new_loads[best_m] = new_loads.get(best_m, 0) + 1
        
        # Now greedily pick the next best op from the remaining
        remaining = [op for j, op in enumerate(ready_operations) if j != i]
        
        if remaining:
            # Find the op with minimum estimated finish time
            next_best_m = None
            next_best_f = float('inf')
            next_best_op = None
            for rop in remaining:
                rmo = rop.get("machine_options", {})
                if not rmo:
                    continue
                for m, d in rmo.items():
                    est_avail = new_loads.get(m, 0) * 1.0
                    finish = est_avail + d
                    if finish < next_best_f:
                        next_best_f = finish
                        next_best_m = m
                        next_best_op = rop
            
            if next_best_m is not None:
                new_loads[next_best_m] = new_loads.get(next_best_m, 0) + 1
        
        # Estimate makespan from final loads
        machine_completions = [cnt * 1.0 for cnt in new_loads.values()]
        makespan_est = _lse2(machine_completions) if machine_completions else 0.0
        
        # Factor in remaining work
        rw = op.get("remaining_work", 0.0)
        scores[i] = makespan_est + 0.1 * rw
    
    return scores

def _fast(ready_operations, loads):
    """Fast fallback: pick op with minimum estimated finish time."""
    best_idx = 0
    best_f = float('inf')
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            continue
        for m, d in mo.items():
            est_avail = loads.get(m, 0) * 1.0
            finish = est_avail + d
            if finish < best_f:
                best_f = finish
                best_idx = i
    return best_idx

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    if _remaining() < 0.5:
        loads = _count_loads(ready_operations)
        return _fast(ready_operations, loads)
    
    n = len(ready_operations)
    loads = _count_loads(ready_operations)
    
    # For small n, use two-step lookahead
    if n <= 15:
        scores = _lookahead_score(ready_operations, loads)
    else:
        scores = _score(ready_operations, loads)
    
    # Pick the operation with the lowest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] < best_score:
            best_score = scores[i]
            best_idx = i
    
    return best_idx
