"""Combined dispatch rule for job-shop scheduling with soft-min, WSPT, and machine-load balancing."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
import math
import numpy as np

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _soft_min(vals, beta=8.0):
    """
    Smooth surrogate for min using log-sum-exp.
    For small beta, approximates min. As beta -> inf, becomes exact min.
    """
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    m = min(vals)
    scaled = [v - m for v in vals]
    max_log = max(-beta * s for s in scaled)
    sum_exp = sum(math.exp(-beta * s - max_log) for s in scaled)
    log_sum_exp = max_log + math.log(sum_exp)
    return m - log_sum_exp / beta

def _normalize(arr):
    if not arr:
        return []
    mn = min(arr)
    mx = max(arr)
    if mx - mn < 1e-12:
        return [0.0] * len(arr)
    return [(x - mn) / (mx - mn) for x in arr]

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Args:
        ready_operations: list of dicts, each with keys:
            - job: job identifier
            - op: operation identifier  
            - machine_options: dict {machine: duration} or similar
            - remaining_ops: number of remaining operations in the job
            - remaining_work: total remaining processing time for the job
        time: current simulation time
    
    Returns:
        int: index of the ready operation to start next
    """
    if _deadline_exceeded():
        return 0
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Extract features for each ready operation
    remaining_works = []
    remaining_ops = []
    min_durs = []
    avg_durs = []
    soft_min_durs = []
    all_vals_per_op = []
    
    for op in ready_operations:
        remaining_works.append(op["remaining_work"])
        remaining_ops.append(op["remaining_ops"])
        
        opts = op["machine_options"]
        if isinstance(opts, dict):
            vals = list(opts.values())
        elif isinstance(opts, (list, tuple, np.ndarray)):
            vals = list(opts)
        else:
            vals = [0]
        
        if vals:
            min_durs.append(min(vals))
            avg_durs.append(sum(vals) / len(vals))
            soft_min_durs.append(_soft_min(vals, beta=8.0))
            all_vals_per_op.append(vals)
        else:
            min_durs.append(0)
            avg_durs.append(0)
            soft_min_durs.append(0)
            all_vals_per_op.append([0])
    
    # Normalize features
    nw = _normalize(remaining_works)
    no = _normalize(remaining_ops)
    nd = _normalize(min_durs)
    nad = _normalize(avg_durs)
    nsd = _normalize(soft_min_durs)
    
    # WSPT ratio: remaining_work / soft_min_duration
    wspt = []
    for i in range(n):
        sm = soft_min_durs[i]
        if sm > 1e-9:
            wspt.append(remaining_works[i] / sm)
        else:
            wspt.append(remaining_works[i])
    nwspt = _normalize(wspt)
    
    # Machine load estimation: for each machine, estimate pending work
    machine_load = {}
    for i in range(n):
        opts = ready_operations[i]["machine_options"]
        if isinstance(opts, dict):
            for machine, dur in opts.items():
                if machine not in machine_load:
                    machine_load[machine] = 0.0
                # Weight by remaining work fraction
                rw = remaining_works[i]
                ro = max(remaining_ops[i], 1)
                machine_load[machine] += dur * rw / ro
    
    # For each candidate, compute a machine-load penalty:
    # If we start op i on its best machine, how congested is that machine?
    # We want to avoid starting ops on heavily loaded machines if alternatives exist.
    machine_penalty = []
    for i in range(n):
        opts = ready_operations[i]["machine_options"]
        if isinstance(opts, dict) and opts:
            # Find the machine with the shortest duration (best machine)
            best_machine = min(opts.keys(), key=lambda m: opts[m])
            load = machine_load.get(best_machine, 0.0)
            machine_penalty.append(load)
        else:
            machine_penalty.append(0.0)
    
    nmp = _normalize(machine_penalty)
    
    # Multi-criteria scoring
    # Primary: WSPT ratio (remaining work / processing time) - balances long jobs with short ops
    # Secondary: remaining work
    # Tertiary: remaining ops (urgency)
    # Quaternary: shortest processing time (inverted)
    # Quintic: soft-min duration (inverted) - smooth bottleneck
    # Sextic: machine load penalty (inverted) - avoid congested machines
    
    w1 = 0.35  # WSPT ratio
    w2 = 0.25  # remaining work
    w3 = 0.15  # remaining ops
    w4 = 0.10  # shortest processing time (inverted)
    w5 = 0.10  # soft-min duration (inverted)
    w6 = 0.05  # machine load penalty (inverted)
    
    scores = []
    for i in range(n):
        score = (
            w1 * nwspt[i] +
            w2 * nw[i] +
            w3 * no[i] +
            w4 * (1.0 - nd[i]) +
            w5 * (1.0 - nsd[i]) +
            w6 * (1.0 - nmp[i])
        )
        scores.append(score)
    
    # Add tiny random tiebreaker
    best_score = -1
    best_idx = 0
    for i in range(n):
        s = scores[i] + _random.random() * 1e-8
        if s > best_score:
            best_score = s
            best_idx = i
    
    return best_idx
# EVOLVE-BLOCK-END
