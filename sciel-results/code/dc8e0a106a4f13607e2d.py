"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Dynamic machine-load tracking
# We maintain a global estimate of when each machine will be free.
# Since dispatch is called event-driven, we update these estimates as we go.
# However, we don't know the exact future, so we use a heuristic:
# "Projected completion time" = current time + (estimated remaining work on that machine)
# But we don't have direct access to machine state. 
# 
# Alternative structural change: 
# Instead of just looking at remaining_work (job-centric), we look at 
# "critical path" estimation. 
# 
# Let's try a different family: **Min-Regret with Machine Load Approximation**
# Actually, the most robust structural change for FJSP is often **Dynamic Scheduling**.
# But we are restricted to the dispatch interface.
#
# Let's implement a **Load-Balanced Shortest Processing Time (SPT)** variant.
# We track the "virtual load" of each machine based on how many operations 
# we have previously dispatched to it in this simulation run (if we could).
# But dispatch is stateless across calls unless we use globals.
# The prompt allows standard library. We can use a global variable to track
# the "last known time" or simply assume that machines with fewer options 
# are more constrained? No.
#
# Let's try **Critical Ratio (CR)** scheduling.
# CR = (Time due - Current Time) / Remaining Work.
# But we don't have "Time due". 
#
# Let's try **Min-Last-Operation-Completion-Time**.
# For each ready operation, estimate the earliest time it can finish if scheduled now.
# This is simply: time + min(duration for each machine).
# But this ignores machine contention.
#
# A stronger structural change: **Lookahead Depth-1 Greedy with Machine Conflict Avoidance**.
# We can't see the future queue, but we can approximate machine busyness.
# If we assume the simulation is event-driven, the `time` parameter is the current simulation time.
#
# New Strategy: **Weighted Min-Completion-Time with Machine Diversity Bonus**
# 1. Calculate the earliest possible completion time for each op: t_c = time + min_dur.
# 2. Prefer ops with earlier t_c (SPT on completion).
# 3. Tie-breaker: Prefer ops that have fewer machine options (constrained first) 
#    to avoid blocking a unique machine later.
# 4. Tie-breaker: Prefer ops with higher remaining_work (critical path).
#
# This is a structural change from "remaining_work + alpha*min_dur" to a 
# "completion-time first" heuristic with constraint-based tie-breaking.

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        # Pick the fastest machine
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_machine)
    
    # Calculate scores for each operation
    # We want to MINIMIZE the score to pick the best one.
    # Primary metric: Earliest Completion Time (time + min_duration)
    # Secondary metric: Number of machine options (fewer is more urgent/constrained)
    # Tertiary metric: Remaining work (higher is more critical)
    
    candidates = []
    
    for i, op in enumerate(ready_operations):
        durations = op["machine_options"]
        if not durations:
            continue
            
        min_dur = min(durations.values())
        earliest_completion = time + min_dur
        
        # Count options
        num_options = len(durations)
        
        # Remaining work
        rem_work = op.get("remaining_work", 0)
        
        # We want to pick the op with the earliest completion.
        # If tie, pick the one with fewer options (more constrained).
        # If tie, pick the one with more remaining work.
        
        # Create a tuple for sorting:
        # (earliest_completion, num_options, -rem_work)
        # Lower is better for completion and options.
        # Higher is better for rem_work, so we negate it.
        
        candidates.append((earliest_completion, num_options, -float(rem_work), i))
        
    # Sort candidates
    # If candidates is empty, return 0 (should not happen)
    if not candidates:
        return 0
        
    candidates.sort()
    
    best_idx = candidates[0][3]
    best_op = ready_operations[best_idx]
    
    # Choose the machine with the minimum duration for the selected operation
    best_machine = min(best_op["machine_options"].keys(), key=lambda m: best_op["machine_options"][m])
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
