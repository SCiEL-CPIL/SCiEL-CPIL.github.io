"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = 120.0  # Match the evaluator limit
_MARGIN = 5.0      # Safety margin

def dispatch(ready_operations, time):
    """
    Dispatch rule for Job-Shop Scheduling.
    
    Strategy:
    1. Deadline Guard: If we are running out of time, fall back to a fast heuristic.
    2. Heuristic: Minimize the "criticality" of the operation.
       We estimate the makespan impact by looking at:
       - The duration of the operation on its best machine.
       - The remaining work of the job.
       - A penalty for choosing a machine that is likely to be busy (approximated by 
         looking at other ready operations that want the same machine).
    
    This is a different family from pure static rules (like MWL or SPT) because it 
    considers the *interaction* between ready operations (machine contention) 
    in a local greedy manner.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - _MARGIN:
        # Fallback: Fastest operation (SPT)
        return int(np.argmin([min(op["machine_options"].values()) for op in ready_operations]))
    
    if not ready_operations:
        return 0
        
    n_ops = len(ready_operations)
    
    # Pre-compute basic metrics
    # min_dur: shortest possible time for this op
    # max_dur: longest possible time (if forced to a bad machine)
    # rem_work: total remaining work for the job
    min_durs = []
    rem_works = []
    
    for op in ready_operations:
        opts = op["machine_options"]
        min_durs.append(min(opts.values()))
        rem_works.append(op["remaining_work"])
        
    # To make this materially different and smarter, we look at machine contention.
    # If multiple ops want the same machine, starting the one with the longest 
    # duration on that specific machine might be bad if it blocks others, 
    # but starting the one with the *shortest* duration on the *most contested* machine 
    # might be good to clear the bottleneck.
    
    # Let's try a "Weighted Shortest Processing Time" with a contention penalty.
    # Score = min_duration + lambda * (contention_penalty)
    # Contention penalty: If this op's preferred machine is also the preferred machine 
    # for other ops, it's a bottleneck.
    
    # 1. Find the "best" machine for each op (min duration)
    best_machines = []
    for op in ready_operations:
        opts = op["machine_options"]
        best_machines.append(min(opts, key=opts.get))
        
    # 2. Count how many ops prefer each machine
    machine_popularity = {}
    for m in best_machines:
        machine_popularity[m] = machine_popularity.get(m, 0) + 1
        
    # 3. Calculate a score for each op.
    # We want to minimize this score.
    # Base cost: min_duration (SPT)
    # Adjustment: If the machine is highly contested, we might prioritize the op that 
    # has the most remaining work on that machine? Or the one that finishes fastest?
    # Actually, in JSS, "Critical Path" heuristics are strong.
    # Let's approximate critical path:
    # Priority = max(remaining_work / min_duration, min_duration) ?
    
    # Let's try a simpler but distinct approach: 
    # "Earliest Completion Time" heuristic.
    # Estimate completion time = time + min_duration.
    # But we don't know when the machine is free.
    
    # Let's use a "Lookahead" score:
    # For each op i, if we start it, it takes min_dur[i].
    # The job has rem_work[i] left.
    # A good heuristic is often: Prioritize jobs with the most remaining work 
    # IF their current operation is short.
    
    # Let's try: Score = min_duration * (1 + rem_work / 100) 
    # No, let's stick to a robust JSS heuristic: 
    # "Most Work Remaining" (MWL) was the baseline.
    # "Shortest Processing Time" (SPT) was inspiration 1 (score 0.94, worse).
    # "Critical Ratio" (CR) = (Due Date - Current Time) / Remaining Work. 
    # We don't have due dates.
    
    # Let's try "Minimum Slack" or "Critical Path".
    # Since we don't have a full Gantt chart, we can approximate.
    
    # New Strategy: **Machine-Load-Aware SPT**
    # If a machine is "hot" (many ops want it), pick the op that takes the *least* time on it.
    # If a machine is "cold", pick the op with the *most* remaining work to balance load.
    
    scores = []
    for i, op in enumerate(ready_operations):
        best_m = best_machines[i]
        pop = machine_popularity.get(best_m, 1)
        min_d = min_durs[i]
        rem_w = rem_works[i]
        
        # If the machine is contested (pop > 1), we strongly prefer SPT (shorter is better)
        # to clear the queue.
        # If the machine is not contested, we prefer MWL (more work is better) to keep 
        # long jobs moving.
        
        # Interpolate between SPT and MWL based on contention.
        # weight_spt = pop / (pop + 1)  -> 0.5 if 1 op, 0.66 if 2, etc.
        weight_spt = pop / (pop + 1.0)
        weight_mwl = 1.0 - weight_spt
        
        # Normalize SPT and MWL to [0,1] roughly to make them comparable?
        # SPT: lower is better.
        # MWL: higher is better.
        
        # Let's just compute a combined score.
        # Score = weight_spt * min_d - weight_mwl * rem_w
        # We want to minimize this.
        
        # To avoid scale issues, we can just use a simple heuristic:
        # If pop > 1: prioritize min_d
        # Else: prioritize rem_w
        
        # Let's use a geometric mean or similar?
        # Let's try: Score = min_d * (rem_w ** 0.1)
        # This favors short durations but doesn't ignore remaining work entirely.
        
        # Actually, a very common effective heuristic is "Critical Path Length".
        # We don't have the graph, but `remaining_work` is a proxy.
        
        # Let's try a different angle: **Randomized Greedy with Tie-Breaking**
        # Sometimes deterministic rules get stuck in local minima.
        # But we need reproducibility? The task doesn't specify, but usually,
        # deterministic is safer for scoring unless we seed RNG.
        
        # Let's go with a **Hybrid Priority**:
        # 1. Identify the operation with the shortest duration on its best machine.
        # 2. Identify the operation with the most remaining work.
        # 3. If the machine of the SPT op is the same as the MWL op, pick SPT (clear bottleneck).
        # 4. Otherwise, pick MWL (keep critical job moving).
        
        # This is a simple "Lookahead" logic.
        
        idx_spt = int(np.argmin(min_durs))
        idx_mwl = int(np.argmax(rem_works))
        
        # If the best SPT op and best MWL op are the same, pick it.
        if idx_spt == idx_mwl:
            return idx_spt
            
        # Check machine contention for the top candidates
        m_spt = best_machines[idx_spt]
        m_mwl = best_machines[idx_mwl]
        
        # If they want different machines, we can start either without immediate conflict 
        # on the same machine. 
        # However, if the MWL job's machine is less contested, it's safer to start it.
        # If the SPT job's machine is more contested, it's urgent to start it.
        
        pop_spt = machine_popularity.get(m_spt, 1)
        pop_mwl = machine_popularity.get(m_mwl, 1)
        
        # Heuristic: If SPT machine is more contested, start SPT.
        # If MWL machine is more contested, start MWL?
        # Actually, if a machine is contested, starting the *shortest* op on it is usually best.
        
        # Let's look at the top 3 SPT ops and top 3 MWL ops and pick the best combo.
        # To keep it fast, we'll just compare the top 1s with a simple rule:
        # If pop_spt > pop_mwl: return idx_spt
        # Else: return idx_mwl
        
        if pop_spt > pop_mwl:
            return idx_spt
        else:
            return idx_mwl

    # Fallback
    return 0
# EVOLVE-BLOCK-END
