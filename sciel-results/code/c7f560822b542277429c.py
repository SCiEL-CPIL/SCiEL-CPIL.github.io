"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

# We use a global variable to track the start time of the evaluation to enforce the 120s limit.
# Since the problem states the evaluator kills the program after 120s (or 150s wall clock),
# we should be well within that. The lookahead simulation is O(N^2 * M) per dispatch call,
# which is fast for 10 jobs and 5 machines.
_start_time = time.time()
TIME_LIMIT = 110.0  # Safety margin

def _simulate_remaining(ready_operations, time, machine_avail, chosen_op_idx, chosen_machine):
    """
    Simulates the rest of the schedule assuming a simple SPT (Shortest Processing Time) rule
    for all subsequent dispatches, starting with the chosen operation on the chosen machine.
    Returns the estimated makespan.
    """
    # Copy state to avoid modifying the original
    m_avail = dict(machine_avail)
    
    # 1. Execute the chosen operation
    op = ready_operations[chosen_op_idx]
    job_id = op["job"]
    dur = op["machine_options"][chosen_machine]
    
    start_t = max(time, m_avail[chosen_machine])
    end_t = start_t + dur
    m_avail[chosen_machine] = end_t
    
    # Update job status: remove this operation from the job's remaining list
    # We need to track which operations are done for which jobs.
    # ready_operations is a list of dicts. We'll create a set of "done" (job, op) pairs.
    # Or simpler: we can just work with a list of "active" operations.
    # However, the input `ready_operations` changes between calls. 
    # For the simulation, we must assume that the *other* operations in `ready_operations` 
    # are the only ones available immediately. But wait, other jobs might have subsequent 
    # operations that become ready only after the current one finishes.
    
    # The input `ready_operations` contains ALL operations that are ready at `time`.
    # If we pick one, the others are still ready at `time` (or later if the machine is busy, 
    # but SPT usually assumes we can start them as soon as their machine is free).
    # Actually, in a discrete event simulation, if we pick op A on machine M1, 
    # op B (ready at time) on machine M2 can start at max(time, M2_avail).
    
    # To simulate correctly, we need to know the *entire* set of operations that will be ready 
    # in the future. The input `ready_operations` only gives us the current set.
    # We do NOT have access to the full job structure (predecessors/successors) in the input 
    # except via `remaining_ops` and `remaining_work`.
    # This is a critical limitation. We cannot simulate the *true* future readiness of operations 
    # that are NOT in `ready_operations` yet.
    
    # Therefore, a true lookahead simulation of the *entire* job shop is impossible with just 
    # the `ready_operations` list, because we don't know when the next operation of a job 
    # (not currently ready) will become ready.
    
    # HOWEVER, we can approximate.
    # A common approximation in such constrained interfaces is to assume that the 
    # "remaining_work" of a job represents the total processing time left.
    # But we need to simulate the *sequence*.
    
    # Alternative Strategy: Since we can't see future operations, we can only optimize 
    # the current batch of ready operations against the current machine states.
    # But the "Lookahead" inspiration suggests evaluating candidates.
    
    # Let's refine the "Lookahead" to be a "Greedy Simulation of the Current Ready Set".
    # We assume that after dispatching the chosen op, the simulator will dispatch the 
    # remaining ready ops using a default rule (or we can simulate them using SPT).
    # But what about ops that are NOT ready yet? They will become ready later.
    # We can estimate their impact by adding their `remaining_work` to the job's critical path?
    
    # Let's try a simpler, robust heuristic that acts like a lookahead:
    # For each candidate (op, machine):
    # 1. Calculate completion time of this op.
    # 2. Estimate the "tail" of the schedule:
    #    - For the job of the chosen op, the remaining work decreases by `dur`.
    #    - For other ready ops, they will be scheduled next.
    #    
    # Actually, the most effective "lookahead" when future operations are hidden is to 
    # prioritize the operation that leads to the earliest completion of the *most critical* job.
    
    # Let's stick to the "Regret" or "Critical Path" logic but enhanced by simulating the 
    # immediate future of the *current* ready set.
    
    # Simulation of current ready set with SPT:
    # 1. Remove chosen op from the list.
    # 2. While there are ready ops:
    #    - Pick the op with the shortest processing time on its best available machine.
    #    - Update machine avail and time.
    # 3. The makespan is max(machine_avail) + estimated_tail.
    #    estimated_tail: For each job not fully finished, add its remaining_work / max_machines?
    #    This is very rough.
    
    # Let's try a different angle: **Minimize the maximum completion time of the current ready set**,
    # weighted by the job's remaining work.
    
    # Actually, let's look at the "Inspiration 2" again: "Lookahead Simulation".
    # If we can't see future ops, we can still simulate the *current* ready ops.
    
    # Let's implement a simulation of the current ready set using SPT for the rest.
    # This gives a better estimate of the "local" makespan contribution than just looking at one op.
    
    # State:
    # - m_avail: dict of machine -> time
    # - ready_ops: list of other ops (excluding chosen)
    
    # We need to be careful: `ready_operations` might contain ops from the same job?
    # Usually, a job has one ready op at a time. So `ready_operations` has at most 1 op per job.
    
    # Simulation Steps:
    # 1. Start with m_avail updated by the chosen op.
    # 2. Create a list of remaining ready ops.
    # 3. Loop while list is not empty:
    #    - Find the op with the minimum duration on any machine.
    #    - Assign it to the machine that allows it to finish earliest (max(time, m_avail[m]) + dur).
    #    - Update m_avail.
    #    - Remove op from list.
    # 4. Current Makespan Estimate = max(m_avail.values())
    # 5. Add a penalty for jobs that still have work left (not in ready set and not chosen):
    #    We don't know their exact times, but we can add `remaining_work` of those jobs?
    #    No, that double counts or is inaccurate.
    
    # Let's just use the max completion time of the simulated current batch as the score.
    # This favors choices that clear the current bottleneck quickly.
    
    remaining_ops = [op for j, op in enumerate(ready_operations) if j != chosen_op_idx]
    
    # Sort remaining ops by minimum possible duration (SPT approximation)
    # We will simulate them one by one.
    # To be fast, we don't need to be perfect, just consistent.
    
    sim_m_avail = m_avail
    
    # Simple SPT simulation for the rest of the ready ops
    # We pick the op with the smallest min-duration next.
    # This is O(N^2) for N ready ops. N <= 10. Very fast.
    
    # To speed up, we can just sort by min duration once?
    # No, machine availability changes.
    
    # Let's just simulate greedily.
    while remaining_ops:
        # Find best op-machine pair for SPT
        best_score = float('inf')
        best_op = None
        best_m = None
        
        for op in remaining_ops:
            for m, d in op["machine_options"].items():
                start = max(time, sim_m_avail.get(m, 0.0))
                end = start + d
                # We want to minimize the end time? Or just pick the shortest duration?
                # Standard SPT picks shortest duration. But if machine is busy, it might not finish earliest.
                # Let's pick the pair that minimizes the completion time.
                if end < best_score:
                    best_score = end
                    best_op = op
                    best_m = m
        
        if best_op is None:
            break
            
        # Update sim state
        sim_m_avail[best_m] = best_score
        remaining_ops.remove(best_op)
        
    # Estimated Makespan from current batch
    local_makespan = max(sim_m_avail.values()) if sim_m_avail else time
    
    # Add a penalty for jobs that are NOT in the ready set (and not the chosen one)
    # These jobs have work remaining but are not ready. They will delay the final makespan.
    # We can estimate their contribution by summing their remaining_work?
    # No, that's too pessimistic.
    # Let's just return local_makespan. The global dispatch loop will handle the rest.
    
    return local_makespan

def dispatch(ready_operations, time):
    global _start_time
    
    if not ready_operations:
        return 0

    # Check time limit
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to simple rule
        min_dur_op = 0
        min_dur_val = float('inf')
        for i, op in enumerate(ready_operations):
            d = min(op["machine_options"].values())
            if d < min_dur_val:
                min_dur_val = d
                min_dur_op = i
        return min_dur_op

    # Initialize machine avail if not present?
    # We don't have a global state for machine avail in the seed, but the seed used one.
    # The prompt says "optionally, which eligible machine runs it".
    # If we return (index, machine), we control the machine.
    # But we need to know the machine's current availability.
    # The seed program maintained a global `_machine_avail`.
    # We MUST maintain this global state to know when machines are free.
    
    # Since we are replacing the seed, we need to define this global state here.
    # But wait, the seed's global state might be lost if the module is reloaded or if 
    # the evaluator runs each instance in a fresh process?
    # "five generated instances" - if they are run in the same process, global state persists.
    # If not, we are in trouble.
    # However, the seed program used a global `_machine_avail` and scored 0.8699.
    # This implies the global state approach is viable (or the evaluator resets it and the seed 
    # re-initialized it on first call per instance? No, `_init_state` only adds new machines, 
    # it doesn't reset times to 0 if they exist. If the process is fresh, `_machine_avail` is empty, 
    # so it initializes to 0. This works for fresh processes.
    # If the process persists across instances, the machine avail times from instance 1 will 
    # corrupt instance 2.
    # The seed scored 0.8699, which is close to 1.0. This suggests it might be working 
    # or the instances are independent processes.
    # To be safe, we should try to detect if we are in a new instance.
    # How? If `time` is 0 or close to 0, it's likely the start.
    # Or if all machines in `ready_operations` are in `_machine_avail` with non-zero values 
    # but `time` is 0?
    
    # Let's add a reset mechanism: If `time` is 0, assume it's the start of a new instance 
    # and reset machine avail to 0.
    
    global _machine_avail
    if '_machine_avail' not in globals():
        _machine_avail = {}
        
    if time == 0:
        _machine_avail = {}
        
    # Ensure all machines in current ops are in the dict
    for op in ready_operations:
        for m in op["machine_options"].keys():
            if m not in _machine_avail:
                _machine_avail[m] = 0.0

    best_score = float('inf')
    best_index = 0
    best_machine = None
    
    # Evaluate each ready operation
    for i, op in enumerate(ready_operations):
        # For each possible machine for this operation
        for m, dur in op["machine_options"].items():
            # Calculate completion time for this specific assignment
            start_t = max(time, _machine_avail.get(m, 0.0))
            end_t = start_t + dur
            
            # Lookahead: Simulate the rest of the ready ops with SPT
            # We pass a copy of machine_avail with this machine updated
            temp_m_avail = dict(_machine_avail)
            temp_m_avail[m] = end_t
            
            # Simulate
            est_makespan = _simulate_remaining(ready_operations, time, temp_m_avail, i, m)
            
            # We want to minimize the estimated makespan
            # Add a small tie-breaker: prefer shorter duration if makespans are close?
            # Or prefer earlier start?
            
            if est_makespan < best_score:
                best_score = est_makespan
                best_index = i
                best_machine = m

    # Update global state with the chosen assignment
    if best_machine is not None:
        op = ready_operations[best_index]
        dur = op["machine_options"][best_machine]
        start_t = max(time, _machine_avail.get(best_machine, 0.0))
        _machine_avail[best_machine] = start_t + dur

    return (best_index, best_machine)

# EVOLVE-BLOCK-END
