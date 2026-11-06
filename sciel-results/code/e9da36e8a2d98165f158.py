"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Track start time for deadline guard
_START_TIME = _time.time()
_DEADLINE = 120.0  # seconds, with safety margin below

def _quick_eval(ready_operations, time):
    """
    A fast proxy for makespan: sum of (time + min_duration) for all ready ops.
    This captures the immediate delay impact.
    """
    total = 0.0
    for op in ready_operations:
        min_dur = min(op["machine_options"].values())
        total += (time + min_dur)
    return total

def dispatch(ready_operations, time):
    # Deadline guard: if we're close to the limit, fall back to a simple rule
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback: most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    best_score = -1e18
    best_index = 0
    
    # 2-step look-ahead
    # For each candidate op i, we simulate scheduling it.
    # Then we evaluate the "quality" of the resulting state.
    # The quality is estimated by the sum of estimated completion times 
    # of the remaining operations in the next step.
    
    # Pre-compute min durations for speed
    min_durs = np.array([min(op["machine_options"].values()) for op in ready_operations])
    
    for i in range(n_ops):
        # Simulate scheduling op i
        # It will finish at time + min_dur[i] (assuming we pick the best machine)
        finish_time_i = time + min_durs[i]
        
        # The remaining operations are all except i
        # We need to estimate the "cost" of the remaining state.
        # A good proxy for the makespan impact of the remaining ops is the sum of their 
        # earliest possible completion times, assuming they can start immediately 
        # (which is an optimistic bound, but relative comparisons hold).
        
        # However, a better proxy for "congestion" is considering that 
        # if we pick op i, we are delaying the others by potentially competing for machines.
        # But since we don't know the machine choices of others yet, we use a simpler metric:
        # Total remaining work + penalty for the time we just added.
        
        # Let's use a metric that favors picking the operation that minimizes the 
        # "average delay" of the remaining set.
        # If we pick i, the "load" added to the system is finish_time_i.
        # The remaining ops are j != i.
        
        # Metric: Sum of (finish_time_i + min_dur[j]) for all j != i?
        # This is equivalent to (n-1)*finish_time_i + sum(min_dur[j] for j != i).
        # Minimizing this means minimizing finish_time_i (shortest duration first) 
        # and maximizing the sum of remaining durations (which is constant).
        # So this reduces to Shortest Processing Time (SPT).
        
        # SPT is a strong baseline. Can we do better?
        # Let's try a "Regret" on the 2nd step.
        # If we pick i, what is the "best" choice for the next op?
        # And what is the "second best"?
        
        # This is getting expensive. Let's stick to a robust 1-step heuristic 
        # but change the family to "Machine Load Balancing" (LPT on machines).
        
        # Actually, the current program uses Regret.
        # Let's try "Most Critical Operation" based on slack.
        # Slack = remaining_time_to_deadline - remaining_work.
        # But we don't have a deadline, we have makespan.
        
        # Let's try a different structural approach: 
        # **Machine-Centric Dispatching**
        # Instead of picking the op, we could pick the machine that is most "free" 
        # relative to the work it has, and then pick the best op for it?
        # No, the interface forces us to return an op index.
        
        # Let's try **Weighted Shortest Processing Time (WSPT)**
        # Weight = remaining_ops (number of ops left in job)
        # This prioritizes jobs that are "long" in terms of steps, 
        # preventing them from getting stuck.
        
        # Or **Most Remaining Operations (MRO)**
        # Prioritize jobs with the most steps remaining.
        
        # Let's combine: 
        # Score = -min_duration + 0.1 * remaining_ops
        # This is SPT with a tie-breaker for job length.
        
        # Let's try a pure **SPT** first, as it's a fundamentally different 
        # family from Regret/Load-Balancing. SPT is often optimal for 1||Cmax.
        
        # To make it "flexible" aware, we use the min duration.
        
        # Let's implement SPT with a small penalty for jobs that have 
        # only 1 machine option (less flexible).
        
        pass # We will implement the actual logic below

    # Implementation:
    # Score = -min_duration (we want to minimize, so we maximize -min_duration)
    # Tie-breaker: Jobs with fewer machine options are more constrained, 
    # so they should be scheduled earlier (like Regret).
    # Flexibility = len(machine_options)
    # Constraint = 1.0 / Flexibility
    
    scores = np.zeros(n_ops)
    for i in range(n_ops):
        min_dur = min_durs[i]
        flexibility = len(ready_operations[i]["machine_options"])
        constraint = 1.0 / flexibility if flexibility > 0 else 0.0
        
        # Primary: Shortest duration
        # Secondary: Most constrained (highest constraint value)
        # Tertiary: Most remaining work (urgency)
        
        scores[i] = -min_dur + 0.1 * constraint + 0.01 * ready_operations[i]["remaining_work"]
        
        # Update best
        if scores[i] > best_score:
            best_score = scores[i]
            best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
