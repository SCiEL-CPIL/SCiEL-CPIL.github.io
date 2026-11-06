"""Strategy: Dynamic Priority via Internal SGS Simulation and Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _simulate_sgs(instance, order):
    """
    Simulates the Serial Schedule Generation Scheme (SGS) for a given activity order.
    Returns the makespan (total duration).
    """
    n = len(instance["durations"])
    durations = instance["durations"]
    preds = instance["predecessors"]
    resources = instance["resources"]
    capacities = instance["capacities"]
    
    # Preprocess predecessors for quick lookup
    # preds is usually a list of lists.
    # We need to know when an activity is eligible.
    # An activity is eligible if all its predecessors are scheduled.
    
    # We maintain:
    # 1. scheduled: set of completed activities
    # 2. ready: set of activities whose preds are in scheduled
    # 3. current_time: time of the last scheduled activity
    # 4. resource_usage: array of current resource usage
    
    scheduled = [False] * n
    ready = set()
    
    # Initialize ready set
    for i in range(n):
        if len(preds[i]) == 0:
            ready.add(i)
            
    # Sort the order list. The SGS picks the highest priority (earliest in list) from ready.
    # To make this efficient, we can just iterate through the `order` list.
    # Wait, standard SGS:
    # While not all scheduled:
    #   Find the activity in `order` that is in `ready`.
    #   Schedule it at the earliest possible time (which is current_time in Serial SGS, 
    #   because Serial SGS schedules activities one by one at the earliest time compatible with 
    #   precedence and resources, but since we process one by one, the time is determined by 
    #   max(pred_end_times) and resource availability).
    
    # Actually, Serial SGS definition:
    # At each step, select the highest priority activity from the set of eligible (ready) activities.
    # Schedule it at the earliest time slot where resources are available.
    # The time is max(earliest_start_due_to_precedence, earliest_time_resources_available).
    
    # Since we are doing Serial SGS, we don't have parallel execution in the "selection" phase,
    # but the scheduled activities DO run in parallel in the timeline.
    # So we need to track resource usage over time.
    
    # Efficient simulation for 30 activities:
    # We can just iterate through the sorted `order`.
    # For each activity in `order`:
    #   If it is in `ready`:
    #     1. Determine earliest start time (EST) due to precedence.
    #        EST_prec = max(end_time[p] for p in preds[i])
    #     2. Determine earliest start time due to resources.
    #        We need to check resource availability from EST_prec onwards.
    #        Since n is small (30), we can check time slots.
    #        However, time slots can be continuous? No, usually integer or discrete.
    #        Let's assume integer time slots for simplicity, or just check intervals.
    #        Given 30 activities, we can maintain a list of "busy intervals" for each resource.
    #        Or simpler: since n is small, we can just scan time.
    #        But time can be large.
    #        Better: maintain a list of (end_time, resource_id, amount) for currently active resources?
    #        No, resources are consumed during [start, end).
        
    # Let's use a simpler approach for n=30:
    # Keep track of resource usage at specific time points.
    # Since we only add 30 activities, there are at most 30 "change points" in resource usage.
    # We can maintain a list of events: (time, resource_id, delta_usage).
    # Or just a list of (start, end, resource_id, amount).
    
    # Let's stick to a simple list of active resource intervals.
    # active_resources: list of [end_time, resource_id, amount]
    
    end_times = [0] * n
    active_resources = [] # List of [end_time, res_id, amount]
    
    for i in order:
        if not ready.__contains__(i):
            continue
            
        # Calculate EST due to precedence
        est_prec = 0
        for p in preds[i]:
            if end_times[p] > est_prec:
                est_prec = end_times[p]
                
        # Calculate EST due to resources
        # We need to find the earliest time t >= est_prec where for all resources r,
        # demand[i][r] + current_usage(r, t) <= capacity[r]
        
        # To find current_usage(r, t), we sum amounts of intervals in active_resources
        # that cover time t.
        # Since we only have at most 30 intervals total in active_resources,
        # and we might need to check a few time points, this is fast enough.
        
        # Candidate start times are:
        # 1. est_prec
        # 2. end times of any active resource interval that starts before or at est_prec + something?
        # Actually, the resource usage is constant between end times of active intervals.
        # So the only times where resource availability changes are the end_times of active intervals.
        # So we check:
        # t = est_prec
        # Then for each end_time in active_resources:
        #    if end_time > est_prec:
        #        t = end_time
        #        check if resources are available at t.
        #        if yes, break.
        
        demands = resources[i]
        
        def is_available(t):
            for r in range(len(capacities)):
                if demands[r] == 0:
                    continue
                usage = 0
                for interval in active_resources:
                    if interval[0] > t: # interval ends after t, so it is active at t
                        usage += interval[2]
                if usage + demands[r] > capacities[r]:
                    return False
            return True
        
        start_time = est_prec
        if not is_available(start_time):
            # Check end times of active resources
            # We want the smallest end_time > start_time
            # Sort active_resources by end_time? Or just find min.
            min_end = float('inf')
            for interval in active_resources:
                if interval[0] > start_time and interval[0] < min_end:
                    min_end = interval[0]
            
            if min_end != float('inf'):
                # Try min_end
                # It might not be enough if multiple resources conflict at different times?
                # Actually, if we jump to the next end time, usage decreases.
                # We might need to iterate if usage still too high?
                # No, usage only decreases at end times.
                # So if it's not available at t, and we jump to the next end time t',
                # usage at t' is <= usage at t.
                # So it should be available?
                # Wait, what if a new activity started at t' ? No, we are scheduling sequentially.
                # Only existing activities end.
                # So yes, jumping to the next end time should resolve resource conflicts
                # caused by the activities currently running.
                
                start_time = min_end
                # Just in case, verify (though logic suggests it should work)
                # If multiple resources have end times, we jump to the earliest one.
                # If that's not enough, we might need to jump again?
                # No, because at that time, at least one resource frees up.
                # But what if another resource is still tight?
                # Example: Res 1 tight until 10. Res 2 tight until 20.
                # We are at 0. Need both.
                # Jump to 10. Res 1 free. Res 2 still tight.
                # We need to jump to 20.
                # So we should loop until available.
                
                while not is_available(start_time):
                    min_end = float('inf')
                    for interval in active_resources:
                        if interval[0] > start_time and interval[0] < min_end:
                            min_end = interval[0]
                    if min_end == float('inf'):
                        break # Should not happen if capacities are valid
                    start_time = min_end

        # Schedule activity i at start_time
        end_time = start_time + durations[i]
        end_times[i] = end_time
        
        # Add to active resources
        for r in range(len(capacities)):
            if demands[r] > 0:
                active_resources.append([end_time, r, demands[r]])
                
        # Update ready set
        # Remove i from ready
        ready.remove(i)
        
        # Add successors of i to ready if all their preds are scheduled
        # We need a successor list or just check all unscheduled?
        # n=30, so checking all is O(n^2) total, which is 900 ops. Fine.
        for j in range(n):
            if not scheduled[j] and j not in ready:
                # Check if all preds of j are in scheduled or {i}
                all_pred_done = True
                for p in preds[j]:
                    if p != i and not scheduled[p]:
                        all_pred_done = False
                        break
                if all_pred_done:
                    ready.add(j)
                    
        scheduled[i] = True
        
    # Makespan is max end_time
    return max(end_times)

def priority(instance):
    start_time = time.time()
    deadline = start_time + 100.0 # Keep 20s margin for 120s limit
    
    n = len(instance["durations"])
    durations = instance["durations"]
    preds = instance["predecessors"]
    
    # Initial order: Longest Processing Time (LPT) is a good heuristic baseline
    # Or Critical Path?
    # Let's start with LPT (sort by duration descending)
    initial_order = list(range(n))
    initial_order.sort(key=lambda i: -durations[i])
    
    best_order = initial_order
    best_makespan = _simulate_sgs(instance, best_order)
    
    # Local Search: Swap adjacent elements or random swaps
    # Given n=30, 100s is plenty for many iterations.
    
    # To speed up, we can cache or optimize _simulate_sgs?
    # n=30 is small. _simulate_sgs is O(n^2 * R) roughly.
    # 30^2 * 4 = 3600 ops per sim.
    # 100s / 3600 ops... Python is slow. Maybe 10k-100k sims per second?
    # Let's just run as many as we can.
    
    random.seed(42) # Deterministic for reproducibility if needed, though task doesn't specify
    
    # Greedy Hill Climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try swapping adjacent pairs
        for i in range(n - 1):
            if time.time() > deadline:
                break
            # Swap i and i+1
            new_order = best_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            # Check validity? SGS handles precedence, so any permutation is a valid "priority list"
            # for the SGS algorithm. The SGS will just skip ineligible ones.
            # Wait, does the order matter if an activity is not eligible?
            # Yes, SGS picks the highest priority ELIGIBLE activity.
            # So the relative order of ineligible activities doesn't matter until they become eligible.
            
            new_ms = _simulate_sgs(instance, new_order)
            if new_ms < best_makespan:
                best_makespan = new_ms
                best_order = new_order
                improved = True
                
    # If time remains, try random restarts or more aggressive swaps
    if time.time() < deadline - 10:
        # Random local search
        for _ in range(10000): # Cap iterations to avoid infinite loop if sim is fast
            if time.time() > deadline:
                break
            # Random swap
            i, j = random.sample(range(n), 2)
            new_order = best_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            new_ms = _simulate_sgs(instance, new_order)
            if new_ms < best_makespan:
                best_makespan = new_ms
                best_order = new_order

    # Return priorities.
    # The priority function must return a score for each activity.
    # Higher score = higher priority.
    # In our best_order, index 0 is highest priority.
    # So priority[i] = n - 1 - position_of_i_in_best_order
    
    pos = [0] * n
    for idx, act in enumerate(best_order):
        pos[act] = idx
        
    priorities = [float(n - 1 - pos[i]) for i in range(n)]
    
    return np.array(priorities)
# EVOLVE-BLOCK-END
