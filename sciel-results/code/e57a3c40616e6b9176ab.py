"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from math import log, exp, sqrt

# We will implement a dispatch rule that tries to minimize makespan in an open-shop setting.
# The current seed uses most-work-remaining (MWR). We can improve by considering machine load.
# Key idea: pick the operation that, when scheduled, leads to the smallest estimated completion
# time, considering both the operation's own duration and the current load on the machine.
#
# We don't have direct access to machine loads, but we can infer from remaining_work and
# machine_options. A common heuristic is to minimize (current_time + duration + machine_load).
# Since we don't have machine_load, we can use a proxy: the minimum duration among options,
# or we can try to balance by preferring machines that are less utilized.
#
# However, we only have: job, op, machine_options {machine: duration}, remaining_ops, remaining_work.
#
# A better approach: For each ready operation, estimate the "impact" of scheduling it now.
# We can compute for each op the minimum possible completion time if we choose the best machine
# (shortest duration), and then prefer the op that has the smallest such completion time,
# but also consider the remaining work to avoid starving jobs.
#
# Let's try a combined score:
#   score = min_duration + alpha * remaining_work
# and pick the op with the smallest score.
#
# But we need to be careful: remaining_work is the total remaining work for the job, not just this op.
# So a job with many ops left might have high remaining_work but a short current op.
#
# Alternative: use the ratio remaining_work / remaining_ops as an estimate of average op duration,
# and then combine with the current op's duration.
#
# Let's try:
#   avg_remaining = remaining_work / max(1, remaining_ops)
#   min_dur = min(machine_options.values())
#   score = min_dur + beta * avg_remaining
#
# We'll tune beta. Start with beta = 0.5.
#
# Also, we should consider that scheduling an op on a specific machine matters. The dispatch function
# can return (index, machine) to specify which machine to use. Let's do that.
#
# For each ready op, we choose the machine that minimizes (duration + machine_load_estimate).
# But we don't have machine_load. So we'll just pick the machine with the shortest duration for that op.
#
# Then, among all ops, we pick the one with the smallest (chosen_duration + beta * avg_remaining).

_dispatch_deadline = None
_dispatch_start = None

def dispatch(ready_operations, time):
    global _dispatch_deadline, _dispatch_start
    
    if _dispatch_deadline is None:
        _dispatch_start = _time.time()
        _dispatch_deadline = _dispatch_start + 120.0  # 120s limit, use 115s to be safe
    
    # If we're running out of time, return a quick fallback
    if _time.time() > _dispatch_deadline - 5.0:
        # Fallback: pick the op with the smallest min duration
        best_idx = 0
        best_dur = float('inf')
        for i, op in enumerate(ready_operations):
            min_dur = min(op["machine_options"].values())
            if min_dur < best_dur:
                best_dur = min_dur
                best_idx = i
        return best_idx
    
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # For each op, compute:
    #   min_dur: shortest duration among machines
    #   best_machine: the machine achieving min_dur
    #   avg_remaining: remaining_work / max(1, remaining_ops)
    #   score: min_dur + beta * avg_remaining
    
    beta = 0.5
    
    best_score = float('inf')
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        min_dur = min(machine_options.values())
        best_machine = min(machine_options, key=machine_options.get)
        
        remaining_work = op["remaining_work"]
        remaining_ops = max(1, op["remaining_ops"])
        avg_remaining = remaining_work / remaining_ops
        
        score = min_dur + beta * avg_remaining
        
        if score < best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine
    
    # Return (index, machine) to specify the machine
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
