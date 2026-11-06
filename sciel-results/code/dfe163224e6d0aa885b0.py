"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard
_DEADLINE_START = _time.time()
_DEADLINE_LIMIT = 130.0  # safety margin before 150s kill


def _check_deadline():
    return (_time.time() - _DEADLINE_START) > _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # If only one option, return it
    if n == 1:
        return 0

    # Deadline check - if we're close to timeout, fall back to simple rule
    if _check_deadline():
        # Simple MWR fallback
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Fast static fallback if too many candidates (shouldn't happen for 10 jobs)
    if n > 50:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Extract basic info
    jobs = [op["job"] for op in ready_operations]
    remaining_works = [op["remaining_work"] for op in ready_operations]
    machine_options_list = [op["machine_options"] for op in ready_operations]

    # Build a map: job -> (remaining_work, remaining_ops) for ready ops
    # We only see ready ops, but we can use their remaining_work as a proxy for job's total remaining work
    # Actually, remaining_work for an op is likely the total remaining work for that job from this op onwards
    # Let's assume remaining_work[op] is the total remaining processing time for job[op.job]
    
    # For lookahead:
    # For each candidate i:
    #   1. Choose the best machine for op i (shortest duration)
    #   2. Estimate completion time for op i: time + duration
    #   3. Estimate new makespan lower bound:
    #      - For job of op i: new remaining work = remaining_work[i] - duration
    #        new earliest completion = time + duration + (estimated time for rest of job)
    #        But we don't know the rest of the job's work easily.
    #      - Alternative: Makespan lower bound = max over all jobs of (earliest possible completion)
    #        For a job, earliest completion >= time + remaining_work[job]
    #        If we process op i on machine m at time t, it finishes at t + d.
    #        The job's remaining work after this op is remaining_work[i] - d.
    #        So the job's earliest completion is at least t + d + (remaining_work[i] - d) = t + remaining_work[i].
    #        This is the same as before! So the lower bound doesn't change for the job.
    #      
    #      However, machine contention matters. If machine m is busy until time t_m, 
    #      then op i can't start until t_m.
    #      But we don't have machine availability times in the interface.
    #      
    #      Wait, the interface only gives `time` (current global time) and ready operations.
    #      It does NOT give machine availability. 
    #      So we cannot simulate machine contention directly unless we infer it.
    #      
    #      However, we can infer that if multiple ready ops want the same machine, 
    #      only one can start now. The others will wait.
    #      So a better heuristic: 
    #      - For each candidate, check if its preferred machine is "contended" by other ready ops.
    #      - If contended, the effective start time is later.
    #      
    #      Let's implement a simple contention-aware score:
    #      1. For each op, identify the machine with the shortest duration.
    #      2. Count how many other ready ops also prefer that machine (or have it as an option).
    #      3. Penalize ops on heavily contended machines.
    #      4. Also, prioritize ops that are "critical" (high remaining work).
    #      
    #      This is a "1-step" look-ahead in terms of machine contention.

    # Identify preferred machine for each op (shortest duration)
    preferred_machines = []
    min_durations = []
    for i in range(n):
        mo = machine_options_list[i]
        if not mo:
            preferred_machines.append(-1)
            min_durations.append(0.0)
        else:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            preferred_machines.append(best_m)
            min_durations.append(mo[best_m])

    # Count contention: for each machine, how many ops prefer it?
    # Also, how many ops have it as an option?
    machine_preference_count = {}
    machine_option_count = {}
    for i in range(n):
        m = preferred_machines[i]
        if m != -1:
            machine_preference_count[m] = machine_preference_count.get(m, 0) + 1
        for m in machine_options_list[i]:
            machine_option_count[m] = machine_option_count.get(m, 0) + 1

    # Score each op:
    # 1. Urgency: high remaining work
    # 2. Contention penalty: if preferred machine is highly contended, penalize
    # 3. Flexibility bonus: if op has many machine options, it's less critical to start now? 
    #    Actually, if it's flexible, we might want to delay it to free up the machine for a rigid op.
    #    So prioritize ops with FEWER options (rigid) on contended machines.

    max_remaining = max(remaining_works) if remaining_works else 1.0
    max_pref_count = max(machine_preference_count.values()) if machine_preference_count else 1

    scores = []
    for i in range(n):
        urgency = remaining_works[i] / max_remaining if max_remaining > 0 else 0.0
        
        m = preferred_machines[i]
        if m == -1:
            contention_penalty = 0.0
            flexibility_bonus = 0.0
        else:
            pref_count = machine_preference_count.get(m, 0)
            opt_count = machine_option_count.get(m, 0)
            
            # If many ops prefer this machine, starting this one might not be optimal
            # if another op is more urgent or more rigid.
            # But we are scoring THIS op.
            # If this op is the most urgent among those preferring m, it should win.
            # The penalty should be relative.
            
            # Simple penalty: high preference count means more contention.
            contention_penalty = pref_count / max_pref_count
            
            # Flexibility: if this op has many options, it can wait.
            # If it has few options, it must start soon.
            n_options = len(machine_options_list[i])
            flexibility_bonus = 1.0 / (1.0 + n_options)  # Higher for fewer options

        # Composite score:
        # We want high urgency, low contention penalty (relative to urgency), and high flexibility bonus.
        # Let's try: score = urgency * (1 + flexibility_bonus) - 0.5 * contention_penalty * (1 - urgency)
        # This means: if urgency is high, penalty is reduced. If urgency is low, penalty is higher.
        
        score = urgency * (1.0 + 0.5 * flexibility_bonus) - 0.5 * contention_penalty * (1.0 - urgency)
        
        # Tie-breaker: shorter duration is better
        score -= 0.01 * (min_durations[i] / (max(min_durations) if max(min_durations) > 0 else 1.0))
        
        scores.append(score)

    scores = np.array(scores)
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
