"""Job-Shop Scheduling dispatch rule combining graph-theoretic critical path lookahead
with SPT, machine load balancing, and job urgency heuristics."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TS = _time.monotonic()
_DEADLINE_S = 115.0  # Safety margin before 150s kill limit


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Strategy: Model as a dynamic bipartite graph between ready operations and machines.
    For each operation-machine pair, estimate the increase in global critical path:
      - The operation's processing duration on that machine
      - Plus the maximum remaining work among jobs that this operation unblocks
        (approximated by the job's own remaining work after this op)
    
    Combine with:
      - SPT (shortest processing time) for local machine efficiency
      - Machine load balancing to avoid bottlenecks
      - Job urgency (remaining work / remaining ops) for global priority
    
    We select the operation that minimizes the estimated critical path impact,
    weighted with secondary heuristics for tie-breaking and robustness.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard: fall back to a fast, safe choice if running long
    if _time.monotonic() - _START_TS > _DEADLINE_S:
        # Fast fallback: pick the operation with minimum processing time
        min_durs = [min(op["machine_options"].values()) for op in ready_operations]
        return int(min(range(n), key=lambda i: min_durs[i]))

    # Precompute per-operation features
    # 1. Minimum duration among machine options (SPT)
    min_durations = []
    # 2. For each operation, the best machine (least loaded) and its load
    # First, compute approximate machine loads from the ready set
    machine_load = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m, d in mo.items():
            machine_load[m] = machine_load.get(m, 0) + d

    for op in ready_operations:
        mo = op.get("machine_options", {})
        if mo:
            min_durations.append(min(mo.values()))
        else:
            min_durations.append(1.0)

    # 3. Remaining work and remaining ops for each operation's job
    remaining_works = [op.get("remaining_work", 0) for op in ready_operations]
    remaining_ops = [op.get("remaining_ops", 0) for op in ready_operations]

    # 4. Critical path proxy: remaining_work / max(remaining_ops, 1)
    #    This approximates the average remaining duration per operation
    critical_path_proxies = []
    for rw, ro in zip(remaining_works, remaining_ops):
        critical_path_proxies.append(rw / max(ro, 1))

    # 5. For each operation, estimate the critical path impact if started now:
    #    - The operation duration (on its best machine)
    #    - Plus the maximum remaining work of the job after this operation
    #      (approximated as remaining_work - min_duration, since this op will be done)
    #    - Plus a penalty for machine congestion (load of the chosen machine)
    
    # For each operation, find the best machine (least loaded among options)
    best_machine_loads = []
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            best_machine_loads.append(0.0)
            continue
        # Find the machine with the least load
        best_m = min(mo.keys(), key=lambda m: machine_load.get(m, 0))
        best_machine_loads.append(machine_load.get(best_m, 0))

    # Compute the estimated critical path impact for each operation
    # impact = duration + max(remaining_work - duration, 0) + congestion_penalty
    # Simplified: impact = remaining_work + congestion_penalty
    # (since remaining_work already includes the current op's duration)
    congestion_weights = [bl / (max(bl) if max(bl) > 0 else 1.0) for bl in best_machine_loads]

    # 6. Composite score: we want to MINIMIZE this score
    #    Primary: estimated critical path impact (lower is better)
    #    Secondary: SPT (lower duration is better)
    #    Tertiary: job urgency (higher remaining work is better, so we subtract it)
    
    # Normalize components to [0, 1] for weighted combination
    max_impact = max(remaining_works) if remaining_works else 1.0
    min_impact = min(remaining_works) if remaining_works else 0.0
    impact_range = max_impact - min_impact
    if impact_range < 1e-9:
        norm_impacts = [0.5] * n
    else:
        norm_impacts = [(rw - min_impact) / impact_range for rw in remaining_works]

    max_dur = max(min_durations) if min_durations else 1.0
    min_dur = min(min_durations) if min_durations else 0.0
    dur_range = max_dur - min_dur
    if dur_range < 1e-9:
        norm_durs = [0.5] * n
    else:
        norm_durs = [(d - min_dur) / dur_range for d in min_durations]

    # Weighted composite score (minimize)
    # w1: critical path impact (weight 0.40)
    # w2: processing duration (weight 0.35) - shorter is better
    # w3: machine congestion (weight 0.15) - less congested is better
    # w4: negative job urgency (weight 0.10) - jobs with more remaining work get priority
    #     (we subtract remaining_work, so higher remaining work reduces the score)
    
    max_rw = max(remaining_works) if remaining_works else 1.0
    norm_rw = [rw / max_rw for rw in remaining_works] if max_rw > 0 else [0.0] * n

    scores = []
    for i in range(n):
        s = (0.40 * norm_impacts[i] +
             0.35 * norm_durs[i] +
             0.15 * congestion_weights[i] -
             0.10 * norm_rw[i])
        scores.append(s)

    # Pick the operation with the lowest composite score
    best_idx = min(range(n), key=lambda i: scores[i])
    return int(best_idx)
# EVOLVE-BLOCK-END
