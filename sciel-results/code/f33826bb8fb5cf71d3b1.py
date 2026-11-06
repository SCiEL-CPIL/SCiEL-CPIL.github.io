import numpy as np
import time as time_module
import random as _random
import heapq as _heapq

# ------------------------------------------------------------------
# Open-Shop Dispatch Rule: Slack-Based Greedy with Weighted Composite
#
# Structural Change:
# - Treats the selection as a constraint satisfaction problem focusing on
#   "slack" (earliest possible start minus current time) to avoid blocking
#   critical machine availability.
# - Uses a Hungarian-algorithm-inspired greedy heuristic: prioritize operations
#   that have the most "room" (slack) before they become critical, combined
#   with a weighted composite of remaining work and flexibility.
# - Incorporates a lightweight model-and-optimize tuning mechanism for the
#   weights of the composite score, validated against a correct event-driven
#   simulation of the open-shop problem.
# ------------------------------------------------------------------

_DEADLINE = 150.0
_START_TIME = time_module.time()
_TIME_MARGIN = 2.0

def _elapsed():
    return time_module.time() - _START_TIME

def _remaining_time():
    return _DEADLINE - _TIME_MARGIN - _elapsed()

def _get_min_duration(op):
    """Get the minimum processing time among available machines."""
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0
    return min(mo.values())

def _get_max_duration(op):
    """Get the maximum processing time among available machines."""
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0
    return max(mo.values())

def _get_avg_duration(op):
    """Get the average processing time among available machines."""
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0
    return sum(mo.values()) / len(mo)

def _count_machine_loads(ready_operations):
    """
    Build a dict mapping machine -> count of ready ops that use it.
    Used as a secondary congestion proxy.
    """
    loads = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo.keys():
            loads[m] = loads.get(m, 0) + 1
    return loads

# --- Tunable Weights ---
# Initial defaults: Weighted composite of remaining_work, remaining_ops, and slack-based metrics
_WEIGHTS = (0.6, 0.2, 0.2)  # (w_work, w_ops, w_slack)

def _make_instance(n_jobs=10, n_machines=5, seed=0):
    """Deterministic random open-shop instance: each job has all operations."""
    rng = _random.Random(seed)
    durations = [[rng.randint(1, 20) for _ in range(n_machines)]
                 for _ in range(n_jobs)]
    return durations

def _simulate(durations, weights):
    """Correct event-driven open-shop simulation.
    
    An operation (job j on machine m) may start as soon as BOTH its job and
    its machine are free. We maintain a ready queue of (job, machine) pairs;
    whenever a pair becomes startable we schedule it immediately at
    max(job_free, machine_free) and re-enqueue any newly-startable pairs of
    the same job. This models true concurrency (multiple ops in flight).
    
    Returns the makespan C_max.
    """
    n_jobs, n_machines = len(durations), len(durations[0])
    w_work, w_ops, w_slack = weights

    remaining_ops = [set(range(n_machines)) for _ in range(n_jobs)]
    remaining_work = [sum(durations[j]) for j in range(n_jobs)]
    machine_free = [0.0] * n_machines
    job_free = [0.0] * n_jobs

    # ready[j] = set of machines whose operation can be started now
    ready = [set(range(n_machines)) for _ in range(n_jobs)]
    total_ops = n_jobs * n_machines
    done = 0
    makespan = 0.0

    while done < total_ops:
        # Build a heap of all currently-startable (job, machine) operations.
        heap = []
        for j in range(n_jobs):
            if not ready[j]:
                continue
            rw = remaining_work[j]
            ro = len(remaining_ops[j])
            for m in ready[j]:
                dur = durations[j][m]
                # Slack heuristic: prefer operations with lower duration (more flexible)
                # and higher remaining work (more urgent)
                # Slack is approximated by negative duration (shorter = more slack)
                slack_term = -dur
                score = w_work * rw + w_ops * ro + w_slack * slack_term
                _heapq.heappush(heap, (-score, j, m))

        if not heap:
            break

        neg_score, j, m = _heapq.heappop(heap)
        start_t = max(machine_free[m], job_free[j])
        end_t = start_t + durations[j][m]
        machine_free[m] = end_t
        job_free[j] = end_t
        remaining_ops[j].discard(m)
        remaining_work[j] -= durations[j][m]
        ready[j].discard(m)
        makespan = max(makespan, end_t)
        done += 1

        # Any other operation of job j becomes startable now that it is free.
        for m2 in remaining_ops[j]:
            if machine_free[m2] <= job_free[j]:
                ready[j].add(m2)

    return makespan

def _tune_weights():
    """Fit (w_work, w_ops, w_slack) on representative instances. Returns tuple."""
    global _WEIGHTS
    grid = [
        (0.6, 0.2, 0.2),
        (0.7, 0.1, 0.2),
        (0.5, 0.3, 0.2),
        (0.6, 0.3, 0.1),
        (0.5, 0.2, 0.3),
        (0.8, 0.1, 0.1),
        (0.4, 0.3, 0.3),
        (0.6, 0.2, 0.2),
    ]
    seeds = [1, 2, 3, 4, 5]
    instances = [_make_instance(seed=s) for s in seeds]

    best_pair = _WEIGHTS
    best_obj = float('inf')
    for pair in grid:
        if not _remaining_time() > 1.0:
            break
        total = 0.0
        for inst in instances:
            total += _simulate(inst, pair)
        obj = total / len(instances)
        if obj < best_obj:
            best_obj = obj
            best_pair = pair
    _WEIGHTS = best_pair
    return best_pair

# Run tuning once at import time (fast; guarded by deadline).
_tune_weights()

def _score_op_slack(op, machine_loads, time):
    """
    Score a single ready operation using a slack-based greedy heuristic.
    Higher score = more preferred.
    
    Primary: remaining_work (urgency)
    Secondary: -min_duration (flexibility/slack)
    Tertiary: -machine_load (congestion avoidance)
    """
    min_dur = _get_min_duration(op)
    rw = op.get("remaining_work", 0.0)
    ro = op.get("remaining_ops", 0)
    if isinstance(ro, list):
        ro = len(ro)
    
    # Calculate min load among available machines for this op
    mo = op.get("machine_options", {})
    min_load = 0
    if machine_loads is not None and mo:
        loads = [machine_loads.get(m, 0) for m in mo.keys()]
        min_load = min(loads) if loads else 0

    # Weights from tuning
    w_work, w_ops, w_slack = _WEIGHTS
    
    # Slack term: prefer shorter operations (more flexible)
    # We use negative min_duration as a proxy for slack
    slack_term = -min_dur
    
    score = (w_work * rw 
             + w_ops * ro 
             + w_slack * slack_term 
             - 0.05 * min_load)
             
    return score

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    if _remaining_time() < 0.5:
        # Fallback to simple heuristic
        return _simple_heuristic(ready_operations, time)
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # 1. Compute machine loads
    machine_loads = _count_machine_loads(ready_operations)
    
    # 2. Score all ops with slack-based heuristic
    scores = []
    for i, op in enumerate(ready_operations):
        s = _score_op_slack(op, machine_loads, time)
        scores.append((s, i))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    if n <= 2:
        return scores[0][1]
    
    # 3. Look-ahead refinement for top-K candidates
    top_k = min(3, n)
    
    best_idx = scores[0][1]
    best_mk = float('inf')
    
    for k in range(top_k):
        idx = scores[k][1]
        
        if _remaining_time() < 1.0:
            break
            
        est_mk = _simulate_makespan_slack(ready_operations, idx, time, machine_loads)
        
        # We want to minimize makespan.
        if est_mk < best_mk - 1e-6:
            best_mk = est_mk
            best_idx = idx
            
    return best_idx

def _simulate_makespan_slack(ready_operations, chosen_idx, time, machine_loads):
    """
    Estimate the makespan if we schedule `chosen_idx` now, and then greedily
    schedule the remaining ready operations using a slack-based heuristic.
    """
    n = len(ready_operations)
    if n == 0:
        return 0.0
    
    # 1. Determine machine for the chosen op.
    op_chosen = ready_operations[chosen_idx]
    mo_chosen = op_chosen.get("machine_options", {})
    
    # Pick machine that minimizes finish time
    best_m_chosen = None
    min_finish = float('inf')
    for m, d in mo_chosen.items():
        finish = time + d
        if finish < min_finish:
            min_finish = finish
            best_m_chosen = m
            
    if best_m_chosen is None:
        best_m_chosen = list(mo_chosen.keys())[0] if mo_chosen else 0
        min_finish = time + mo_chosen.get(best_m_chosen, 0.0)
        
    # Initialize virtual availability for all machines
    all_machines = set()
    for op in ready_operations:
        for m in op.get("machine_options", {}).keys():
            all_machines.add(m)
            
    virtual_avail = {m: time for m in all_machines}
    virtual_avail[best_m_chosen] = min_finish
    
    remaining_indices = [i for i in range(n) if i != chosen_idx]
    max_completion = min_finish
    
    # 2. Greedily schedule remaining ops.
    # Strategy: Slack-based heuristic (prefer shorter, more flexible ops)
    while remaining_indices:
        best_i = -1
        best_m = -1
        best_finish = float('inf')
        
        for i in remaining_indices:
            op = ready_operations[i]
            mo = op.get("machine_options", {})
            for m, d in mo.items():
                start_time = virtual_avail[m]
                finish = start_time + d
                # Slack-based: prefer operations that finish earlier
                if finish < best_finish:
                    best_finish = finish
                    best_i = i
                    best_m = m
                    
        if best_i == -1:
            break
            
        # Update state
        virtual_avail[best_m] = best_finish
        if best_finish > max_completion:
            max_completion = best_finish
        remaining_indices.remove(best_i)
        
    return max_completion

def _simple_heuristic(ready_operations, time):
    """
    Fast fallback: Pick op with max remaining_work, tie-break by min_duration.
    """
    best_idx = 0
    best_score = float('-inf')
    
    for i, op in enumerate(ready_operations):
        min_dur = _get_min_duration(op)
        rw = op.get("remaining_work", 0.0)
        
        # Simple heuristic: prioritize remaining work, then flexibility
        s = 1.0 * rw - 0.1 * min_dur
        
        if s > best_score:
            best_score = s
            best_idx = i
            
    return best_idx
