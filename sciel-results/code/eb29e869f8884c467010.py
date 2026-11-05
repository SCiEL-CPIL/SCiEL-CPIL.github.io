"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

# Deadline guard: 145s is a safe margin before the 150s kill limit.
_START_TIME = time.time()
_DEADLINE = 145.0

# Beam search parameters
BEAM_WIDTH = 5
LOOKAHEAD_DEPTH = 3

def _simulate_and_score(ready_ops, current_time, depth):
    """
    Simulate the next 'depth' dispatch decisions using a greedy SPT (Shortest Processing Time) 
    rule to estimate the resulting schedule state.
    Returns a score (lower is better) representing the estimated makespan contribution 
    or remaining work.
    """
    # Copy state to avoid modifying original
    # ready_ops is a list of dicts. We need to track which are done and machine availability.
    # Since we don't have global machine state, we assume 'time' is the current global time.
    # In a job shop, if multiple ops are ready, they can run in parallel on different machines.
    # However, the dispatch function is called when a machine becomes free or a new job becomes ready.
    # The prompt says "picks which ready operation to start next". This implies a single machine 
    # is available or we are prioritizing for a specific machine? 
    # "dispatch(ready_operations, time) -> index of the ready operation to start"
    # Usually in event-driven simulations, this is called when an event occurs (machine free).
    # If multiple machines are free, the simulator might call dispatch for each? 
    # Or does dispatch return the op to start on the *currently available* machine?
    # The interface returns an index. It doesn't specify which machine.
    # "machine_options {machine: duration}" suggests the op can go to multiple machines.
    # If the simulator provides `ready_operations`, it likely contains ops that are ready to start.
    # If I pick index i, the simulator will start op[i] on its best available machine?
    # Or does the simulator handle machine assignment?
    # "dispatch ... picks which ready operation to start next"
    # Let's assume the simulator handles the actual machine assignment based on the chosen op's preferences 
    # or that the 'time' parameter helps us infer machine load.
    
    # For the lookahead, we need to simulate the effect of picking one op.
    # If we pick op i, it occupies a machine for duration d.
    # Other ready ops might wait or run in parallel if on different machines.
    
    # Since we don't know which machine is free, we must assume the worst-case or best-case scenario, 
    # or that the "time" passed is the time when a specific machine became free.
    # Let's assume the standard case: one machine just became free, and we have a set of ready ops.
    # We pick one. It runs. The machine is busy for 'duration'.
    # Other machines might be free too? If so, the simulator would have triggered dispatch for them too?
    # Given the single return value, it's likely a single-machine dispatch decision context.
    
    # Let's estimate the "cost" of the schedule by looking at the sum of completion times 
    # or the maximum completion time of the batch of ready ops if they were processed sequentially 
    # on the free machine, or in parallel if possible.
    
    # A simpler proxy for makespan reduction in JSS:
    # 1. Minimize the time the critical path is blocked.
    # 2. Minimize the total flow time.
    
    # Let's use a weighted sum:
    # Score = min_completion_time + alpha * (total_remaining_work / num_ready)
    
    # Actually, let's just simulate the next 'depth' steps assuming we pick the current candidate,
    # and for subsequent steps, we pick the best available op using a simple greedy rule (SPT).
    
    # State: 
    # - remaining_ready_ops: list of ops
    # - current_time
    # - machine_available_time: when the current machine will be free (starts at current_time)
    
    # Wait, if we pick op i, it runs on some machine. 
    # If we assume the machine is free at 'time', it finishes at time + duration.
    # During this time, other ops are waiting.
    
    # Let's define the score as the estimated makespan of the *current set of ready ops* 
    # assuming they are processed on the available machine(s).
    # Since we don't know the number of free machines, let's assume 1 free machine for the dispatch decision.
    
    # Simulation:
    # 1. Pick candidate op.
    # 2. Determine its duration on the "best" machine (min duration).
    # 3. Update time to time + duration.
    # 4. Remove op from ready list.
    # 5. For the next 'depth-1' steps, pick the op with min duration (SPT) among remaining.
    # 6. Accumulate the completion times.
    # 7. Return the sum of completion times or the final time.
    
    ops = list(ready_ops)
    current_t = current_time
    total_completion_time = 0
    
    for step in range(depth):
        if not ops:
            break
        
        if step == 0:
            # Use the candidate passed in? 
            # Actually, this function is called for EACH candidate in the beam.
            # So we need to pass the specific candidate index to simulate.
            pass
        
        # This function needs to know which op to pick first.
        # Let's refactor: _evaluate_candidate(candidate_idx, ready_ops, time, depth)
    
    return 0 # Placeholder

def _evaluate_candidate(candidate_idx, ready_ops, current_time, depth):
    """
    Evaluate the candidate by simulating the next 'depth' steps.
    Step 0: Pick candidate_idx.
    Steps 1..depth-1: Pick the op with min duration (SPT) from remaining.
    Score: Sum of completion times of these operations.
    """
    remaining_indices = list(range(len(ready_ops)))
    if candidate_idx not in remaining_indices:
        return float('inf')
    
    # Remove candidate from remaining for subsequent steps
    remaining_indices.remove(candidate_idx)
    
    current_t = current_time
    score = 0
    
    # Step 0: Process candidate
    op = ready_ops[candidate_idx]
    # Choose the machine with minimum duration for this op
    min_dur = min(op["machine_options"].values())
    current_t += min_dur
    score += current_t # Completion time of this op
    
    # Subsequent steps: Greedy SPT
    for _ in range(depth - 1):
        if not remaining_indices:
            break
        
        # Find op with min duration among remaining
        best_idx = -1
        best_dur = float('inf')
        for idx in remaining_indices:
            r_op = ready_ops[idx]
            dur = min(r_op["machine_options"].values())
            if dur < best_dur:
                best_dur = dur
                best_idx = idx
        
        if best_idx == -1:
            break
            
        remaining_indices.remove(best_idx)
        current_t += best_dur
        score += current_t
        
    return score

def dispatch(ready_operations, time):
    """
    Dispatch rule: Beam Search with Lookahead.
    
    1. For each ready operation, estimate the cost of picking it now by simulating 
       the next few dispatch decisions (using SPT for subsequent picks).
    2. Select the operation with the minimum estimated cost.
    """
    if not ready_operations:
        return 0
    
    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback to simple SPT to avoid timeout
        min_dur = float('inf')
        best_idx = 0
        for i, op in enumerate(ready_operations):
            dur = min(op["machine_options"].values())
            if dur < min_dur:
                min_dur = dur
                best_idx = i
        return int(best_idx)
        
    n = len(ready_operations)
    
    # If few ops, evaluate all. If many, sample or just top K?
    # With 10 jobs, ready_ops is likely small (<10). Evaluating all is fast.
    best_score = float('inf')
    best_idx = 0
    
    for i in range(n):
        # Check deadline frequently if n is large, but n is small here
        if time.time() > _DEADLINE:
            break
            
        score = _evaluate_candidate(i, ready_operations, time, LOOKAHEAD_DEPTH)
        if score < best_score:
            best_score = score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
