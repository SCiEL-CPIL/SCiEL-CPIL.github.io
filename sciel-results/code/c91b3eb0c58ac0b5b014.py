"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Stateful tracking to estimate machine loads if possible, 
# but primarily relying on dynamic critical path calculation.
_start_time = None

def dispatch(ready_operations, time):
    global _start_time
    if _start_time is None:
        _start_time = _time.time()
        
    # Deadline guard: if we are running too long, fall back to a fast heuristic
    if _time.time() - _start_time > 100:
         # Fallback: Shortest Processing Time (SPT)
         min_dur = float('inf')
         best_idx = 0
         for i, op in enumerate(ready_operations):
             durs = op["machine_options"].values()
             if durs:
                 min_d = min(durs)
                 if min_d < min_dur:
                     min_dur = min_d
                     best_idx = i
         return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
        
    best_score = -float('inf')
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        # 1. Calculate Critical Path Priority (CPP)
        # The 'remaining_work' is a sum of durations, but 'remaining_ops' count 
        # or a more specific critical path metric is better. 
        # Since we don't have the full graph, we approximate critical path 
        # by assuming the remaining work is distributed along the critical path.
        # A job with high remaining work is likely on the critical path.
        # However, a better proxy for "criticality" in a fixed sequence job shop 
        # is often just the total remaining processing time (MRW) because 
        # the operations are sequential. The last operation's completion time 
        # is the sum of all remaining durations.
        
        # Let's refine: 
        # Primary Metric: Remaining Work (MRW). This directly correlates with 
        # the earliest possible completion time of the job if it were the only 
        # job on its machines. Jobs with more remaining work have a higher 
        # potential to extend the makespan if delayed.
        
        mrw = op["remaining_work"]
        
        # 2. Machine Selection Heuristic
        # We want to pick the machine that minimizes the completion time of this specific operation.
        # Completion Time = current_time + duration.
        # Since current_time is constant, we minimize duration.
        # However, if we can influence machine load, we should prefer machines that are "less busy".
        # Without explicit load state, we can't know "busy". 
        # But we can look at the duration itself. Shorter duration on any machine is generally better 
        # for clearing the bottleneck.
        
        # Let's try a composite score:
        # High MRW is good (priority).
        # Low Duration is good (efficiency).
        
        # If we just pick max MRW, we might pick a job with a very long operation that blocks a machine.
        # If we pick min Duration, we might starve long jobs.
        
        # Strategy: "Critical Path + Shortest Processing Time"
        # Score = MRW - alpha * min_duration
        # This favors jobs that have lots of work left, but breaks ties (or slight preferences) 
        # in favor of shorter operations to reduce idle time on machines.
        
        # Let's estimate the "effective duration" as the minimum duration available on any machine.
        min_dur = min(options.values())
        
        # The weight alpha needs to be tuned. 
        # If MRW is in the same units as duration, a small alpha (e.g., 0.1) makes MRW dominant.
        # A larger alpha (e.g., 1.0) balances them.
        # Given the seed was pure MRW (score 1.0) and current is SPT+0.1MRW (score 1.02),
        # let's try a different balance: Pure MRW is known to be decent. 
        # Let's try "Most Remaining Work" with a tie-breaker of "Shortest Duration".
        # This is effectively MRW, but if two jobs have similar remaining work, pick the one 
        # that can finish its current op faster.
        
        # To make it materially different from the current "SPT + 0.1 MRW", 
        # let's invert the priority: MRW is primary, SPT is secondary.
        
        score = mrw - 0.05 * min_dur
        
        if score > best_score:
            best_score = score
            best_idx = i
            # Select the machine with the shortest duration for this op
            # This assumes the simulator uses the machine we implicitly suggest 
            # or that picking the index implies the best machine choice.
            # The interface allows returning (index, machine).
            # Let's return just the index to be safe, as the simulator likely 
            # picks the optimal machine from options if not specified, 
            # OR we should return the tuple if it's supported.
            # The prompt says: "return index ... (optionally (index, machine))".
            # Returning the tuple is more precise.
            best_machine = min(options, key=options.get)

    # If the interface accepts (index, machine), return that.
    # If it only accepts index, the tuple might cause an error.
    # The prompt says "dispatch(ready_operations, time) -> index ... (optionally (index, machine))".
    # This implies the function should return an index, but *can* return a tuple.
    # To be most robust and potentially better, let's return the tuple if we identified a specific machine.
    # However, standard practice in these "dispatch" interfaces where machine_options is a dict 
    # is that the simulator expects the index, and it might default to the first machine or the best one.
    # If I return a tuple and the simulator expects an int, it breaks.
    # If I return an int and the simulator expects a tuple, it breaks.
    # The prompt says "return index ... (optionally (index, machine))". This usually means 
    # the *evaluator* handles both, or the *simulator* handles both.
    # Given the ambiguity, returning the index is the standard "safe" bet for "index of the ready operation".
    # But wait, if I return the index, how does the simulator know which machine to use?
    # It must have a default rule (e.g., first in dict, or min duration).
    # If I return (index, machine), I control it.
    # Let's assume the evaluator is smart enough to handle the tuple as suggested by "optionally".
    
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return int(best_idx)

# EVOLVE-BLOCK-END
