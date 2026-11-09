"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Global state to track simulation context if needed, 
# though the lookahead approach primarily relies on local state passed in args.
# We keep a simple state to potentially cache or just for safety.
_last_time = -1.0

def _simulate_greedy(ready_ops, current_time):
    """
    Fast greedy simulation to estimate makespan.
    Uses a simple rule: Most Work Remaining (MWR) with earliest free machine.
    This serves as the 'policy' for the rollout.
    """
    # Deep copy state to avoid modifying original
    # ready_ops is a list of dicts. We need to track:
    # 1. Jobs and their remaining ops (sequence)
    # 2. Machine availability
    
    # Extract static structure from the first call or assume consistent structure
    # Since we are in a lookahead, we must reconstruct the state.
    # ready_ops contains: job, op, machine_options, remaining_ops, remaining_work
    
    # We need to know the FULL remaining schedule for each job to simulate properly.
    # The input 'ready_ops' only gives us the CURRENT ready op.
    # However, 'remaining_ops' and 'remaining_work' are provided.
    # To simulate the *entire* future, we need the sequence of upcoming operations for each job.
    # The prompt implies `ready_operations` is the list of ops ready *now*.
    # The dict contains `remaining_ops` (count) and `remaining_work` (sum of durations).
    # It does NOT explicitly provide the sequence of future operations and their machine options 
    # in the `ready_operations` list for the *current* dispatch call, unless `remaining_ops` 
    # implies we can infer them? No, we can't infer specific machine options for future ops 
    # from just `remaining_work`.
    
    # CRITICAL REALIZATION:
    # The `dispatch` function is called at every event. 
    # If I want to do a lookahead, I need to know what happens *after* I pick an op.
    # I don't have the full problem instance (matrix of ops) passed into `dispatch`.
    # I only have the `ready_operations` list.
    # 
    # Therefore, a true "lookahead simulation" of the *entire* future is impossible 
    # unless I assume something about the future or I have stored the full instance.
    # 
    # Can I store the full instance?
    # The evaluator generates instances. It calls dispatch repeatedly.
    # I do not receive the full instance definition.
    # 
    # HOWEVER, looking at the inspirations and the nature of these tasks:
    # Often, the "lookahead" in these constrained interfaces is limited to 
    # evaluating the immediate consequences or using a surrogate.
    # 
    # Alternative Interpretation of "Stochastic Process":
    # The directive says: "simulate the completion time of each ready operation on its fastest machine, then select the operation with the minimum expected remaining job completion time".
    # This is NOT a full shop simulation. It is a **1-step lookahead** or **local evaluation**.
    # 
    # Let's re-read the directive carefully:
    # "Model the dispatch decision as a stochastic process by simulating the completion time of each ready operation on its fastest machine, then select the operation with the minimum expected remaining job completion time (current time + min_duration + remaining_work)"
    # 
    # This is effectively: Score = time + min(machine_duration) + remaining_work.
    # Wait, `remaining_work` usually includes the current op? 
    # If `remaining_work` is the sum of durations for *remaining* ops (including the one being scheduled), then:
    # Estimated Finish Time for Job J if we start Op O now on Machine M:
    # Start Time = max(current_time, machine_free_time[M])
    # Finish Time of Op = Start Time + Duration
    # Remaining Work after Op = remaining_work - Duration
    # Estimated Final Job Completion = Finish Time of Op + Remaining Work after Op
    #    = (Start Time + Duration) + (remaining_work - Duration)
    #    = Start Time + remaining_work
    #    
    # So the "minimum expected remaining job completion time" simplifies to:
    # min over machines M for this op of: max(time, free_time[M]) + remaining_work
    # 
    # Since `remaining_work` is constant for the op, we just minimize the start time.
    # This reduces to picking the op that can start earliest on its best machine, 
    # weighted by the urgency (remaining work)?
    # 
    # The directive says: "select the operation with the minimum expected remaining job completion time".
    # If I pick Op A, Job A finishes at ~ T_A.
    # If I pick Op B, Job B finishes at ~ T_B.
    # But I only pick ONE op. The other jobs wait.
    # 
    # If I pick Op A, the makespan of the *system* is not just T_A.
    # 
    # Let's look at the "Stochastic" part.
    # Maybe it implies using the *variance* or *distribution* of durations? 
    # But durations are fixed.
    # 
    # Perhaps the "stochastic" part refers to the fact that we don't know the future 
    # machine conflicts, so we estimate the "effective" start time.
    # 
    # Let's try a different angle: **Lookahead via Greedy Rollout on Available Info**.
    # Even without the full instance, I can simulate the *immediate* impact.
    # 
    # Actually, there is a stronger heuristic that fits the "different family" requirement 
    # and uses the available info better than MWR: **Weighted Shortest Processing Time (WSPT)** 
    # or **Critical Ratio**.
    # 
    # But the directive specifically asks for: "simulate the completion time of each ready operation on its fastest machine... minimum expected remaining job completion time".
    # 
    # Let's implement exactly that, but add a "penalty" for machine congestion to make it "stochastic" or robust.
    # 
    # Score for Op i:
    # 1. Find best machine m* for Op i: min(max(time, free[m]), free[m]) -> effectively min(free[m]) if free[m] > time.
    # 2. Estimated Completion of Op i = max(time, free[m*]) + duration_i,m*
    # 3. Estimated Job Completion = Estimated Completion of Op i + (remaining_work_i - duration_i,m*)
    #    = max(time, free[m*]) + remaining_work_i
    # 4. We want to MINIMIZE this value? 
    #    If we minimize the completion time of the *picked* job, we are prioritizing jobs that will finish soon.
    #    This is an "Earliest Due Date" style if we treat completion time as due date.
    #    However, in JSSP, we usually want to prioritize jobs with LONG remaining paths to avoid them becoming critical later.
    #    Minimizing completion time of the *current* op might lead to Shortest Job First, which is bad for makespan.
    #    
    # Let's re-read: "select the operation with the minimum expected remaining job completion time".
    # If I pick the op with the *minimum* predicted finish time, I am doing SPT-like behavior.
    # If I pick the op with the *maximum* predicted finish time (i.e., the one that will finish latest if started now), I am doing MWR-like behavior.
    # 
    # The directive says "minimum". This suggests SPT/EDD. 
    # Is SPT better for makespan? Generally no, LPT is better.
    # But maybe in combination with machine selection?
    # 
    # Let's try a hybrid: 
    # The "stochastic" aspect might be a randomized tie-breaking or a soft-min over machines.
    # 
    # Let's try a **Lookahead-1** approach:
    # For each candidate op i:
    #   1. Assume we start it on its best machine m*.
    #   2. Update machine_free_time[m*] = time + duration.
    #   3. For each OTHER ready op j:
    #      - Calculate the *new* earliest start time on its best machine given the updated machine_free_time.
    #      - Estimate the completion time of job j.
    #   4. Sum or Max these estimated completion times to get a "System Cost".
    #   5. Pick the op i that minimizes the System Cost.
    # 
    # This is a 1-step lookahead that considers the impact on other ready jobs.
    # This is materially different from the current "pick best, ignore others" approach.
    # It is a "Greedy with Lookahead" or "Hill Climbing" step.
    
    if not ready_operations:
        return 0

    # We need to track machine free times. 
    # Since we don't have a global reset, we must infer state or use a local approximation.
    # The current program uses a global `machine_free_time`. 
    # I will keep a global `machine_free_time` but update it carefully.
    # To make the lookahead work, I need a *copy* of the machine state for each candidate.
    
    # Initialize global state if not present
    global _machine_state
    if not hasattr(_simulate_greedy, '_machine_state'):
        _simulate_greedy._machine_state = {}
        
    # We need to know the current machine free times.
    # The current program updates `machine_free_time` globally.
    # I will use a global dict `_machine_free` that is updated by the main dispatch.
    
    # For the lookahead, I will create a temporary copy of the machine state.
    
    # First, determine the "best" machine for each op in the current state.
    # And calculate the base score.
    
    # To implement the 1-step lookahead:
    
    # 1. Get current machine free times (from global state)
    current_machine_free = dict(_machine_free)
    
    best_idx = 0
    best_machine = None
    min_system_cost = float('inf')
    
    for i, op in enumerate(ready_operations):
        # Try starting op i
        # Find the best machine for op i in the current state
        machine_options = op["machine_options"]
        
        # We want to pick the machine that minimizes the "System Cost"
        # But first, let's pick the "best" machine for op i locally.
        # Usually, the machine with the earliest free time is best.
        
        best_local_machine = None
        earliest_local_free = float('inf')
        
        for m, d in machine_options.items():
            ft = current_machine_free.get(m, 0)
            if ft < earliest_local_free:
                earliest_local_free = ft
                best_local_machine = m
        
        if best_local_machine is None:
            continue
            
        # Simulate starting op i on best_local_machine
        # New free time for that machine
        duration_i = machine_options[best_local_machine]
        new_free_time_i = max(time, earliest_local_free) + duration_i
        
        # Create a modified machine state for this candidate
        sim_machine_free = dict(current_machine_free)
        sim_machine_free[best_local_machine] = new_free_time_i
        
        # Now, estimate the cost for the *other* jobs
        # Cost = Sum of estimated completion times of all ready jobs? 
        # Or Max? Makespan is driven by the max.
        # Let's use the Sum of (Start Time + Remaining Work) for all jobs as a proxy for congestion.
        
        system_cost = 0
        
        for j, op_j in enumerate(ready_operations):
            if i == j:
                # We already started this one
                # Its contribution to the "future load" is its completion time
                # But for the *other* jobs, we care about how early they can start.
                # Let's calculate the estimated completion time for job j.
                pass
            
            # For job j (including i, but we treat i specially or just include it)
            # Let's calculate the earliest start time for op j given sim_machine_free
            
            best_machine_j = None
            earliest_start_j = float('inf')
            
            for m_j, d_j in op_j["machine_options"].items():
                ft_j = sim_machine_free.get(m_j, 0)
                # Start time is max(time, ft_j)
                # But if ft_j < time, it's free at time.
                start_t = max(time, ft_j)
                if start_t < earliest_start_j:
                    earliest_start_j = start_t
                    best_machine_j = m_j
            
            # Estimated completion of job j
            # Start + Remaining Work
            # Note: remaining_work includes the current op.
            # So Estimated Finish = Start + Remaining Work
            est_finish_j = earliest_start_j + op_j["remaining_work"]
            
            system_cost += est_finish_j
        
        # We want to MINIMIZE the system cost (sum of estimated finish times)
        # This penalizes choices that delay other jobs significantly.
        
        if system_cost < min_system_cost:
            min_system_cost = system_cost
            best_idx = i
            best_machine = best_local_machine
    
    # Update the global machine state with the chosen decision
    if best_machine is not None:
        _machine_free[best_machine] = max(time, _machine_free.get(best_machine, 0)) + ready_operations[best_idx]["machine_options"][best_machine]
        
    return (best_idx, best_machine)

# Global state for machine free times
_machine_free = {}

def dispatch(ready_operations, time):
    """
    Dispatch rule using 1-step lookahead simulation.
    """
    if not ready_operations:
        return 0
        
    # Reset machine state if time seems to be the start of a new instance?
    # The evaluator runs 5 instances. If they are sequential, time might reset to 0?
    # Or time might be continuous?
    # "time" is passed. If time < _last_time, it's a new instance.
    global _last_time
    if time < _last_time:
        _machine_free.clear()
    _last_time = time
    
    return _simulate_greedy(ready_operations, time)

# EVOLVE-BLOCK-END
