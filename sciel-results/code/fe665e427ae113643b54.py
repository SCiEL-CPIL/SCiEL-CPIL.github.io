"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

# Global state to track machine availability if needed, but we rely on local simulation
# We need to be careful not to mutate the passed-in state.

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145  # 5s safety margin before 150s limit

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just pick the first one or random
        return random.randint(0, len(ready_operations) - 1)
    
    n_ops = len(ready_operations)
    
    # Strategy: Machine-centric bottleneck heuristic
    # For each ready operation, estimate the "effective completion time" 
    # by considering the minimum duration it could achieve on any eligible machine.
    # Then, to balance load, we penalize operations that would go to already-busy machines.
    # Since we don't have explicit machine load in the input, we approximate 
    # "load" by looking at the distribution of durations across ready operations.
    # 
    # A different structural approach: 
    # Instead of just picking the job with max remaining work, we score each operation
    # by a composite metric that includes:
    # 1. The operation's own processing time (SPT favors short)
    # 2. The job's remaining work (LWR favors long - known to help from island 5 finding)
    # 3. A "congestion" proxy: if many ready ops share the same best machine, 
    #    that machine is likely congested. We can estimate this by counting 
    #    how many other ready ops also list that machine as their best option.
    
    # Step 1: For each operation, find its best machine (shortest duration)
    op_info = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        best_machine = min(machine_options, key=lambda m: machine_options[m])
        best_duration = machine_options[best_machine]
        rem_work = op["remaining_work"]
        op_info.append({
            'idx': i,
            'best_machine': best_machine,
            'best_duration': best_duration,
            'rem_work': rem_work,
            'num_options': len(machine_options)
        })
    
    # Step 2: Count how many ops prefer each machine (congestion proxy)
    machine_congestion = {}
    for info in op_info:
        m = info['best_machine']
        machine_congestion[m] = machine_congestion.get(m, 0) + 1
    
    # Step 3: Score each operation
    # We want to minimize makespan. Key insights:
    # - Long remaining work jobs should be prioritized (LWR) to avoid delaying critical paths
    # - But we should avoid sending many ops to the same congested machine
    # - Shorter processing times free up machines faster
    
    # Score = rem_work * alpha - congestion_penalty * beta + duration * gamma
    # We want to maximize the score (pick highest priority)
    # rem_work: higher is better (prioritize long jobs)
    # congestion: lower is better (avoid crowded machines)
    # duration: lower is better (SPT)
    
    # Normalize or use raw values
    scores = []
    for info in op_info:
        rem_work = info['rem_work']
        congestion = machine_congestion[info['best_machine']]
        duration = info['best_duration']
        
        # Composite score: prioritize high remaining work, low congestion, short duration
        # Weights are chosen to balance these factors
        score = rem_work * 10.0 - congestion * 5.0 - duration * 1.0
        scores.append((score, info['idx']))
    
    # Pick the operation with the highest score
    scores.sort(reverse=True)
    best_idx = scores[0][1]
    
    return int(best_idx)

# EVOLVE-BLOCK-END
