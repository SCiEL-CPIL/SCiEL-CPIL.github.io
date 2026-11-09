"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for the CPM heuristic
machine_free_time = {}
job_remaining_work = {}

def dispatch(ready_operations, time):
    global machine_free_time, job_remaining_work
    
    # Initialize or update global state
    # Note: We assume the simulation resets or we handle new instances by clearing state if necessary
    # However, since we don't know when an instance ends, we must be careful.
    # A robust way is to detect if this is a new instance. 
    # But typically in these tasks, the state persists across calls for one instance.
    # If the evaluator runs multiple instances sequentially without resetting the module,
    # we might need a reset mechanism. 
    # Looking at the interface, 'time' is passed. 
    # A common trick: if time is 0 or very small, it might be the start.
    # But a better way for CPM is to rely on the 'remaining_work' provided in the dict,
    # which is updated by the simulator.
    
    # The 'remaining_work' in the op dict is the sum of durations for all remaining ops in the job.
    # We can use this directly.
    
    # We need to estimate the earliest start time for each ready operation.
    # The operation can start at max(current_time, machine_free_time[machine]).
    # Since we don't know the exact machine until we pick, we have to estimate.
    
    # Strategy:
    # 1. For each ready operation, calculate the "Critical Path" length.
    #    Critical Path ~ Remaining Work of the job.
    #    However, a more accurate CPM considers the specific machine constraints.
    #    
    #    Let's use a heuristic that approximates CPM:
    #    Priority = Remaining Work of Job / (1 + Number of Remaining Ops) ? No.
    #    
    #    Standard CPM heuristic for JSSP:
    #    1. Calculate the critical path length for each job from the current operation to the end.
    #       Since we have fixed order, this is just the sum of remaining durations.
    #       But which duration? The one on the assigned machine?
    #       The problem statement says `machine_options {machine: duration}`.
    #       This implies an operation might have multiple machine options?
    #       "fixed operation order" but "machine_options".
    #       Usually in JSSP, an operation is assigned to a specific machine or has a set.
    #       If it has options, we must choose the machine too?
    #       The return interface is `index` OR `(index, machine)`.
    #       The seed returns just `index`.
    #       If I return just `index`, how is the machine chosen?
    #       "picks which ready operation to start next".
    #       If the machine isn't specified, maybe the simulator picks the first available?
    #       Or maybe `machine_options` is a single key?
    #       Let's assume `machine_options` is a dict. If it has multiple keys, we might need to return the machine.
    #       However, the seed only returns index. To be safe and potentially improve,
    #       if multiple machines are available, I should probably return the best machine.
    #       But the prompt says: "return index of the ready operation to start (optionally (index, machine))".
    #       If I return (index, machine), I have more control.
        
    # Let's implement a CPM-like heuristic:
    # Score = Remaining Work of the Job
    # But adjusted by machine availability.
    
    # Actually, a stronger CPM heuristic:
    # 1. Estimate the earliest completion time of the job if this op is started now.
    # 2. The job's remaining work is known.
    # 3. The bottleneck is the machine.
    
    # Let's try a "Most Work Remaining" but with a machine load factor.
    # Or simply: The operation that belongs to the job with the longest critical path.
    
    # The seed uses `remaining_work`.
    # Let's try: `remaining_work` is a good proxy for critical path.
    # To improve, we should account for the machine's current load.
    # If a machine is busy, starting an op on it now means waiting.
    
    # Heuristic:
    # For each ready op:
    #   - Get the machine(s) it can use.
    #   - Estimate the start time on the best available machine (earliest free).
    #   - The "urgency" is the remaining work.
    #   - But if the machine is heavily loaded, maybe it's better to wait? No, we must pick one.
    
    # A common effective heuristic is "Longest Processing Time" (LPT) or "Most Work Remaining" (MWR).
    # The seed is MWR.
    # Let's try "Critical Ratio" or a CPM approximation.
    
    # CPM Approximation:
    # The critical path of a job is the sum of the processing times of its remaining operations.
    # We have `remaining_work` which is exactly this (sum of durations).
    # So MWR is essentially CPM in terms of job priority.
    
    # To differentiate, we need to consider the *machine* aspect more deeply.
    # If I return (index, machine), I can choose the machine that is free earliest.
    
    # Let's assume `machine_options` is a dict {machine_id: duration}.
    # If I return just index, the simulator might pick the first one or the one with shortest duration?
    # To be safe, let's return (index, machine) and pick the machine that minimizes the start time + duration?
    # Or the machine that is free earliest?
    
    # New Strategy:
    # 1. Identify the job with the highest `remaining_work` (CPM proxy).
    # 2. Among the ready operations of that job (there should be only one ready op per job at a time in fixed order? 
    #    Actually, "ready_operations" is a list. In a job shop, each job has one next operation ready at a time.
    #    So `ready_operations` length <= number of jobs.
    # 3. For that operation, select the machine that becomes free earliest.
    
    # Wait, if I always pick the job with the highest remaining work, that's the same as the seed, 
    # except I might optimize the machine choice.
    # Does the seed's machine choice matter?
    # If the seed returns just index, the simulator's default machine choice might be suboptimal.
    # By returning (index, machine), I can ensure the op is scheduled on the least congested machine.
    
    # Let's implement this:
    # 1. Find the op with max `remaining_work`.
    # 2. If there's a tie, break by other criteria.
    # 3. For that op, find the machine in `machine_options` that is free earliest.
    #    We need to track `machine_free_time`.
    
    # Since `dispatch` is called at every event, we can update `machine_free_time` 
    # when we schedule an operation.
    # But we don't know when an operation *finishes*.
    # We only know when we *start* it.
    # The `time` parameter is the current simulation time.
    
    # When we return (index, machine), the simulator will start the op on that machine at `time`.
    # The op will finish at `time + duration`.
    # So `machine_free_time[machine]` should be updated to `time + duration`.
    
    # However, we don't know which machine the simulator will use if we return just index.
    # So returning (index, machine) gives us control.
    
    # Let's track `machine_free_time` globally.
    # We need to initialize it.
    # When `time` is 0, we can assume all machines are free at 0.
    
    # Problem: How do we know the set of all machines?
    # We can collect them from `ready_operations` over time.
    
    # Let's refine the heuristic:
    # Priority = remaining_work
    # Tie-breaker: Smallest processing time on the best available machine?
    
    # Let's try:
    # 1. Calculate score for each op: `remaining_work`.
    # 2. Pick the max score.
    # 3. For that op, pick the machine with the earliest `machine_free_time`.
    # 4. Update `machine_free_time[machine]` to `time + duration`.
    
    # This is a structural change: it becomes stateful and optimizes machine assignment.
    
    # Implementation details:
    # - `machine_free_time` is a dict.
    # - We need to ensure it's initialized.
    # - If a machine is not in `machine_free_time`, assume it's free at 0.
    
    # Let's code this.
    
    if not ready_operations:
        return 0 # Should not happen
    
    # Find the operation with the most remaining work
    max_work = -1
    best_idx = -1
    best_op = None
    
    for i, op in enumerate(ready_operations):
        work = op["remaining_work"]
        if work > max_work:
            max_work = work
            best_idx = i
            best_op = op
            
    # If no op found (should not happen), return 0
    if best_idx == -1:
        return 0
        
    # Now, choose the best machine for this operation
    machine_options = best_op["machine_options"]
    
    # Find the machine with the earliest free time
    best_machine = None
    earliest_free = float('inf')
    
    for machine, duration in machine_options.items():
        # If we haven't seen this machine, assume it's free at 0
        current_free = machine_free_time.get(machine, 0)
        if current_free < earliest_free:
            earliest_free = current_free
            best_machine = machine
            
    # If no machine found (empty options?), return index only
    if best_machine is None:
        return best_idx
        
    # Update the machine free time
    # The operation will start at max(time, earliest_free)?
    # Actually, the simulator starts it at `time`.
    # If the machine is free at `earliest_free`, and `time` is the current time,
    # the op starts at `time`.
    # The machine will be busy until `time + duration`.
    # So we update `machine_free_time[best_machine]` to `time + duration`.
    
    duration = machine_options[best_machine]
    machine_free_time[best_machine] = time + duration
    
    return (best_idx, best_machine)

# EVOLVE-BLOCK-END
