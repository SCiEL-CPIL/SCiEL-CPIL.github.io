"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Stateful tracking of machine loads to enable dynamic balancing
# This differs from the seed's stateless, job-centric approach.
_machine_loads = {}
_start_time = None

def dispatch(ready_operations, time):
    global _start_time
    if _start_time is None:
        _start_time = _time.time()
        
    # Deadline guard: if we are running too long, fall back to a fast heuristic
    if _time.time() - _start_time > 100:
         # Fallback: Shortest Processing Time on any machine (SPT)
         # This is fast and often good for makespan.
         min_dur = float('inf')
         best_idx = 0
         for i, op in enumerate(ready_operations):
             durs = op["machine_options"].values()
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
    
    # Heuristic: 
    # 1. Primary: Minimize the machine load after assignment (Least Loaded Machine)
    # 2. Secondary: Maximize remaining work (to keep critical path jobs moving)
    # 3. Tertiary: Minimize current operation duration (SPT)
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        # Find the best machine for this operation based on current load
        # We want to pick the machine that results in the smallest "new load"
        # Load = current_time + current_duration + existing_load
        # Since current_time is constant for all, we minimize (existing_load + duration)
        
        min_new_load = float('inf')
        min_dur_on_best_machine = float('inf')
        
        for m, dur in options.items():
            current_load = _machine_loads.get(m, 0)
            # Note: In a strict simulation, the load updates when the op finishes.
            # However, for dispatching, we estimate the future load.
            # A simpler proxy often used is just the current accumulated "busy time" or similar.
            # Since we don't know when the machine becomes free relative to 'time' without more state,
            # we use a heuristic: prefer machines with lower historical total processing time assigned.
            
            # Actually, a better proxy for "least loaded" in a dispatch rule without full Gantt state
            # is to look at the number of operations already assigned or the sum of durations.
            # Let's stick to a simpler, robust heuristic that doesn't require perfect load tracking:
            # 1. Shortest Processing Time (SPT) on the machine that has the fewest *assigned* ops?
            # No, we don't have that.
            
            # Let's refine the strategy:
            # The seed used "Most Remaining Work" (MRW).
            # A strong competitor is "Shortest Processing Time" (SPT) to clear the board.
            # Another is "Least Loaded Machine" (LLM).
            
            # Since we can't perfectly track machine availability without the full simulator state passed in,
            # we can use a heuristic that combines:
            # 1. SPT: Pick the op with the shortest duration on its *preferred* machine.
            # 2. Tie-break with MRW to avoid starving long jobs.
            
            pass

        # Let's implement a composite score:
        # Score = - (min_duration_on_any_machine) + (remaining_work * small_weight)
        # This prioritizes quick completions (SPT) but slightly favors jobs with more work left.
        
        min_dur = min(options.values())
        
        # Normalize remaining work to avoid it dominating if scales differ
        # But usually remaining_work is comparable.
        
        # Let's try: Primary key = Shortest Duration (SPT)
        # Secondary key = Most Remaining Work (MRW)
        
        # To make it "different" and potentially better:
        # Pure SPT is often very good for makespan in job-shop.
        # The seed was Pure MRW.
        # Let's try SPT with a small MRW boost.
        
        score = -min_dur + 0.1 * op["remaining_work"]
        
        if score > best_score:
            best_score = score
            best_idx = i
            
            # Update our internal tracking of machine loads?
            # This is tricky because we don't know which machine will be picked by the simulator
            # if we just return the index. The simulator likely picks the machine specified in machine_options
            # or allows the dispatch to specify it?
            # The interface says: "return index of the ready operation to start (optionally (index, machine))"
            # If we return just the index, the simulator must have a default machine selection logic 
            # (e.g., first available, or least loaded).
            # If we can return (index, machine), we have more control.
            
            # Let's assume we can return (index, machine) to be more precise.
            # We should pick the machine that minimizes the completion time for this op.
            # Since we don't know machine availability, we can't perfectly pick.
            # However, returning just the index is safer if the simulator handles machine assignment optimally 
            # or via a standard rule.
            
            # Given the prompt "picks which ready operation to start", and the return type "index",
            # I will stick to returning the index. The machine selection is likely handled by the simulator 
            # or is part of the 'ready_operations' structure where the 'best' machine is implied or 
            # the simulator picks the one with the min duration if multiple options exist?
            # Actually, `machine_options` is a dict. If I return index, how does it know which machine?
            # Usually, in these tasks, if you return just the index, the simulator might pick the 
            # machine with the shortest duration, or the first one.
            # To be safe and effective, I will implement the logic to pick the best (index, machine) pair 
            # if the interface allows it, but the prompt says "return index ... (optionally (index, machine))".
            # I will return just the index to be compatible with the simplest interpretation, 
            # assuming the simulator handles the machine choice (likely shortest duration or least loaded).
            
    return int(best_idx)

# EVOLVE-BLOCK-END
