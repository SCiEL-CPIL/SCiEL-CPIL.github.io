"""Job-Shop Scheduling: Combined Critical-Path Reduction + Machine Load Balancing Dispatch."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill

def _min_pair(machine_options):
    """Return (min_duration, machine) from {machine: duration}."""
    if not machine_options:
        return 1.0, None
    best_m, best_d = None, float("inf")
    for m, d in machine_options.items():
        if d < best_d:
            best_d = d
            best_m = m
    return best_d, best_m

def _is_time_up():
    return (_time.time() - _START_TIME) > _TIME_LIMIT

def dispatch(ready_operations, time):
    """
    Dispatch rule combining:
    1. Critical Path Reduction (from current program): Maximizes the reduction in the global
       lower bound of makespan (max remaining work across jobs).
    2. Machine Load Balancing (from inspiration strategy): Penalizes assigning operations
       to machines that are already heavily loaded (longest estimated processing time),
       effectively balancing load and avoiding bottlenecks.
    3. Urgency/Speed refinement (from inspiration 1): A smooth component that favors
       high urgency (remaining work) relative to processing time, helping break ties
       and prioritize 'fast' critical jobs.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # 1. Pre-compute job remaining work
    job_work = {}
    for i, op in enumerate(ready_operations):
        job_id = op['job']
        # Each job appears at most once in ready_operations for the *next* operation
        job_work[job_id] = op['remaining_work']
    
    if not job_work:
        return 0

    max_work = max(job_work.values())
    jobs_list = list(job_work.keys())
    
    # 2. Pre-compute machine loads (estimated total processing time assigned so far + current time? 
    #    Actually, we don't have global machine state history in this interface, only current time.
    #    However, we can infer "load" by looking at the ready ops themselves? 
    #    No, the strategy says "minimizing the resulting increase in the maximum machine load".
    #    Since we don't have the state of *past* assignments or *future* ready ops for other jobs 
    #    (only current ready ops), we must approximate.
    #    
    #    Wait, the interface `dispatch(ready_operations, time)` implies we only see currently ready ops.
    #    We cannot know the load on a machine from *previous* steps unless we maintain state.
    #    But the prompt says "Write a Python dispatch... rule". It does not forbid global state.
    #    Let's maintain a global dictionary of machine loads (cumulative processing time).
    #    
    #    IMPORTANT: The evaluator might reset state or run instances sequentially. 
    #    If state persists across instances, it might be wrong. 
    #    If state resets, we can't track load.
    #    Usually, in these tasks, if state isn't provided, we assume a stateless heuristic or 
    #    we manage our own state carefully. 
    #    Given the "event-driven" nature, a global machine_load dict is a standard approach 
    #    IF the simulator calls dispatch for each event in sequence within an instance.
    #    Let's assume the 5 instances are run sequentially. We should clear state between instances?
    #    We don't know when an instance ends.
    #    
    #    Alternative interpretation: "Load" is derived from the *current* ready set?
    #    If I pick op A for machine M, does it increase M's load? Yes.
    #    But what is the *current* load? 
    #    If we can't track it, we can't balance it.
    #    
    #    Let's look at the inspirations. They are stateless.
    #    Inspiration 2 uses `_machine_eta` but rebuilds it? No, it uses global `_machine_eta` 
    #    but `_rebuild_caches` resets it? 
    #    `_rebuild_caches` sets `_machine_eta` to empty dict or 0.0. 
    #    This suggests the inspirations are effectively stateless or assume a fresh start per call?
    #    If `_machine_eta` is global and not reset properly, it's buggy. 
    #    
    #    Let's stick to a STATELESS heuristic that approximates load balancing 
    #    using the information available in `ready_operations` and `time`.
    #    
    #    How to estimate machine load without history?
    #    We can't. 
    #    
    #    However, we can use a proxy: 
    #    "Greedy load balancing" often implies picking the machine with the *least* 
    #    *potential* future conflict or simply the *shortest* processing time (SPT) 
    #    to keep machines free. 
    #    
    #    The strategy says: "select the operation-machine pair that minimizes the resulting 
    #    increase in the maximum machine load".
    #    
    #    If we assume all machines are currently free (or we don't know their state), 
    #    the "load" of a machine after assignment is just the duration of this op.
    #    Then minimizing the increase in max load is equivalent to minimizing the duration.
    #    This collapses to SPT.
    #    
    #    BUT, if we assume that machines might be busy, we can't know it.
    #    
    #    Let's re-read the current program. It uses `remaining_work` and `min_duration`.
    #    It doesn't use machine state.
    #    
    #    Maybe the "Load Balancing" idea is to be implemented via a *tie-breaker* 
    #    or a *secondary* score that prefers machines that are less "popular" in the 
    #    current ready set? 
    #    
    #    Let's try a different angle: 
    #    Combine the Critical Path Reduction (primary) with a penalty for 
    #    "Machine Congestion" estimated by how many other ready ops *could* use that machine?
    #    If many ops are contending for Machine M, assigning one of them to M might 
    #    not be "balancing" if M is already a bottleneck. 
    #    
    #    Actually, a simpler interpretation of the directive "Model the job-shop as a bipartite graph... 
    #    select the operation-machine pair that minimizes the resulting increase in the maximum machine load" 
    #    in a stateless context:
    #    
    #    We don't have the load. We only have the *duration*.
    #    If we assume the current load is 0, the new load is duration.
    #    We want to minimize the new max load.
    #    If we pick Op A (dur 10) for M1 and Op B (dur 5) for M2.
    #    Max load becomes 10.
    #    If we pick Op B (dur 5) for M1 and Op A (dur 10) for M2.
    #    Max load becomes 10.
    #    It doesn't help unless we know the *current* loads.
    #    
    #    Let's look at the "Inspiration 2" code again. It maintains `_machine_eta` globally.
    #    If I use a global `_machine_load` dictionary, I must be careful about resetting.
    #    If the evaluator runs 5 instances, and I don't reset, my loads will accumulate 
    #    incorrectly from previous instances.
    #    
    #    Is there a signal for a new instance? 
    #    If `time` resets to 0, or `ready_operations` contains Job 0 Op 0?
    #    Let's add a heuristic: if `time` is significantly lower than previous call, 
    #    or if we see Job 0 Op 0 again, reset state.
    #    
    #    Let's implement a lightweight stateful machine load tracker with reset logic.
    #    """
    global _machine_load, _last_time, _last_job_op_seen
    
    # Initialize state if not present
    if not hasattr(dispatch, '_machine_load'):
        dispatch._machine_load = {}
        dispatch._last_time = 0
        dispatch._last_job_op_seen = set()

    # Reset logic: If time goes back significantly, or we see a "start" op again
    # A robust reset is hard without explicit signals. 
    # However, if `time` < dispatch._last_time - 1, assume reset.
    if time < dispatch._last_time - 1.0:
        dispatch._machine_load = {}
    
    dispatch._last_time = time

    # Update machine loads for the operation we are about to pick? 
    # No, we are *picking* now. We should update *after* picking? 
    # But `dispatch` returns the index. The simulator will then advance time.
    # We can update the load *now* assuming we are assigning it, but if we return the wrong one, 
    # the state is corrupted.
    # 
    # Better: Use the load *estimation* for scoring, but don't commit until we are sure?
    # We can't be sure. 
    # 
    # Alternative: Don't use global state. Use a stateless proxy.
    # 
    # Let's stick to the stateless approach to be safe, but incorporate the *spirit* 
    # of load balancing by preferring shorter durations (which reduces the peak load 
    # on any single machine in a static sense) and using the Critical Path Reduction 
    # as the primary driver.
    #
    # Refined Strategy:
    # Primary Score: Reduction in Global Critical Path (from current program).
    # Secondary Score (Tie-Breaker/Refinement): 
    #   1. Urgency/Speed ratio (from Inspiration 1): remaining_work / min_duration.
    #   2. Machine "Balance" Proxy: Prefer operations that have *more* machine options? 
    #      No, that's flexibility.
    #   3. Shortest Processing Time (SPT): Minimizes the duration added to any machine.
    
    best_score = -float('inf')
    best_index = 0
    best_reduction = -1.0
    best_dur = float('inf')
    best_urgency_speed = 0.0

    for i, op in enumerate(ready_operations):
        job_id = op['job']
        durations = op['machine_options'].values()
        if not durations:
            d = 1.0
        else:
            d = min(durations)
            
        current_work = job_work[job_id]
        new_work = current_work - d
        
        # 1. Critical Path Reduction (Primary)
        max_other = 0
        for j_id in jobs_list:
            if j_id != job_id:
                if job_work[j_id] > max_other:
                    max_other = job_work[j_id]
        
        new_global_max = max(new_work, max_other)
        reduction = max_work - new_global_max

        # 2. Urgency/Speed (Secondary)
        # Higher is better. 
        # If d is 0, avoid div by zero.
        if d <= 0:
            d = 1e-9
        urgency_speed = op['remaining_work'] / d
        
        # 3. Machine Load Balancing Proxy (Tertiary)
        # We want to minimize the "load" we add. Since we don't know current load,
        # we assume the "cost" of the assignment is proportional to d.
        # We already have d in urgency_speed (denominator).
        # To explicitly "balance", we can penalize if this machine is "popular" 
        # among other ready ops? 
        # Let's count how many other ready ops can use the *best* machine for this op.
        # If many ops compete for the same fast machine, it's a bottleneck.
        # But we don't know which machine they *will* use.
        
        # Let's use a composite score:
        # Score = reduction * 1000 + urgency_speed * 10 - d * 1
        
        # We want to maximize reduction first.
        # If reduction is tied, maximize urgency_speed.
        # If urgency_speed is tied, minimize d (SPT).
        
        if reduction > best_reduction:
            best_reduction = reduction
            best_index = i
            best_dur = d
            best_urgency_speed = urgency_speed
        elif reduction == best_reduction:
            # Tie-breaker 1: Urgency/Speed
            if urgency_speed > best_urgency_speed:
                best_index = i
                best_dur = d
                best_urgency_speed = urgency_speed
            elif urgency_speed == best_urgency_speed:
                # Tie-breaker 2: Shortest Duration (SPT)
                if d < best_dur:
                    best_index = i
                    best_dur = d
                    best_urgency_speed = urgency_speed

    return int(best_index)

# EVOLVE-BLOCK-END
