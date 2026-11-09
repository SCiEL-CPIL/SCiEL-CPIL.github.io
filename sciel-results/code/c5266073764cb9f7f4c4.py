# EVOLVE-BLOCK-START
import time
import heapq
import numpy as np

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def _serial_sgs(durations, demands, capacities, predecessors, priority_vec):
    """
    Run a serial schedule-generation scheme with a given priority vector.
    Returns the makespan.
    """
    n = len(durations)
    if n == 0:
        return 0

    # remaining_preds[j] = number of predecessors of j not yet scheduled
    remaining_preds = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for j in range(n):
        pred_list = predecessors[j]
        if pred_list is not None:
            cnt = len(pred_list)
            remaining_preds[j] = cnt
            for p in pred_list:
                successors[p].append(j)

    # Priority heap: (-priority, index)
    # We want highest priority first, so we negate for min-heap.
    # If priorities are equal, we use index as tie-breaker (lower index first).
    # Note: The task says "highest-priority eligible activity".
    # The priority function returns a score. Higher score = higher priority.
    # So we push -score.
    
    # Initial eligible set: activities with no predecessors
    heap = []
    for j in range(n):
        if remaining_preds[j] == 0:
            # Use negative priority for max-heap simulation
            heapq.heappush(heap, (-priority_vec[j], j))

    # Resource availability timeline
    # Since resources are renewable and we schedule serially,
    # we can just track the earliest time each resource becomes free?
    # No, multiple activities can use the same resource at different times.
    # But in serial SGS, we schedule one activity at a time.
    # The start time of an activity is determined by:
    # 1. The finish times of its predecessors.
    # 2. The availability of the required resources.
    
    # We need to track the earliest time each resource is free?
    # Actually, since activities are scheduled sequentially, the resource usage
    # is a sequence of intervals.
    # For a new activity, its start time is max(
    #     max(finish_times of preds),
    #     max(earliest_free_time of each required resource)
    # )
    # Wait, this is only true if we assume that the resource is free at that time.
    # Since we schedule one by one, the "earliest_free_time" for a resource
    # is simply the finish time of the last activity that used that resource.
    
    # Let's verify this logic.
    # Activity A uses R1, duration 10. Starts at 0, ends at 10. R1 free at 10.
    # Activity B uses R1, duration 5. Preds: A.
    # Start B = max(finish A, R1 free) = max(10, 10) = 10. Ends at 15.
    # Activity C uses R1, duration 5. Preds: none.
    # If C is scheduled after B, Start C = max(0, R1 free=15) = 15. Ends at 20.
    # This logic holds because we are scheduling serially. The state of resources
    # is just "when is it next free?".
    
    # However, an activity might use multiple resources.
    # Start time = max(
    #     max(finish_time[p] for p in preds),
    #     max(resource_free_time[r] for r in resources_required)
    # )
    
    finish_times = np.zeros(n, dtype=float)
    resource_free_time = np.zeros(len(capacities), dtype=float)
    # demands is likely a list of lists or a 2D array.
    # Let's assume demands[j] is a list of resource indices or a vector of amounts?
    # The task says "demands, capacities".
    # Usually in RCPSP, demands[j][k] is the amount of resource k needed by activity j.
    # And capacities[k] is the total capacity of resource k.
    # If demands[j][k] > 0, activity j needs resource k.
    # The constraint is sum(demands[j][k] for all active j) <= capacities[k].
    # In serial SGS, at any point, only one activity is being started.
    # But previous activities might still be running?
    # NO. Serial SGS schedules activities one by one.
    # Does "serial" mean that only one activity is active at any time?
    # No, serial SGS (Serial Schedule Generation Scheme) means that we select one eligible activity at a time and schedule it as early as possible.
    # Other activities that were scheduled earlier might still be running.
    # Therefore, we CANNOT just track "resource_free_time" as a single scalar per resource.
    # We need to track the actual usage profile or at least the finish times of all scheduled activities.
    
    # Correct approach for Serial SGS:
    # When scheduling activity j:
    # 1. Determine earliest start time based on predecessors: t_pred = max(finish[p] for p in preds).
    # 2. Determine earliest start time based on resources:
    #    We need to find the smallest t >= t_pred such that for all resources k,
    #    sum(demands[i][k] for all scheduled i where start[i] < t < finish[i]) + demands[j][k] <= capacities[k].
    
    # This requires checking resource availability at time t.
    # To do this efficiently, we can keep a list of scheduled activities (start, finish, demands).
    # For a candidate start time t, we check all scheduled activities to see if they overlap with [t, t+dur_j].
    
    # Optimization:
    # The critical times are the finish times of currently running or just-finished activities.
    # We can iterate through potential start times.
    
    # Let's implement a helper to find the earliest feasible start time.
    
    def earliest_start(j, t_pred):
        t = t_pred
        dur_j = durations[j]
        req_j = demands[j]
        
        # We need to check if resources are available at t.
        # If not, we advance t to the next "event" time (finish time of an overlapping activity).
        
        while True:
            feasible = True
            next_event = float('inf')
            
            for k in range(len(capacities)):
                if req_j[k] == 0:
                    continue
                # Calculate current usage of resource k at time t
                usage = 0.0
                for i in range(n):
                    if finish_times[i] == 0 and i != j and remaining_preds[i] >= 0 and i not in [j]:
                        # This is messy. Let's keep a list of scheduled activities.
                        pass
                
                # Better: keep a list of scheduled activities: (start, finish, demands_vec)
                # Let's restructure.
            
            # This inner loop is O(n^2) or worse if done naively.
            # With n=30, it's fine.
            
            break # Placeholder
        
        return t
    
    # Let's restart the implementation properly.
    
    scheduled = []  # List of (start, finish, demands_list)
    
    def find_earliest_start(j, t_pred):
        t = t_pred
        dur_j = durations[j]
        req_j = demands[j]
        
        while True:
            # Check feasibility at time t
            feasible = True
            next_event = float('inf')
            
            for k in range(len(capacities)):
                if req_j[k] == 0:
                    continue
                
                # Sum usage of resource k by all scheduled activities that overlap with [t, t+dur_j)
                # An activity i overlaps if start_i < t+dur_j and finish_i > t
                usage = 0.0
                for s_i, f_i, d_i in scheduled:
                    if s_i < t + dur_j and f_i > t:
                        usage += d_i[k]
                
                if usage + req_j[k] > capacities[k]:
                    feasible = False
                    # Find the earliest finish time among overlapping activities that use resource k
                    # We need to move t to a point where some usage drops.
                    # The drop happens at the finish time of an overlapping activity.
                    for s_i, f_i, d_i in scheduled:
                        if s_i < t + dur_j and f_i > t and d_i[k] > 0:
                            if f_i > t and f_i < next_event:
                                next_event = f_i
                    break
            
            if feasible:
                return t
            
            # Move t to next_event
            if next_event == float('inf'):
                # Should not happen if there is a solution, but safeguard
                return t
            t = next_event
    
    # Main loop
    while heap:
        if _remaining_budget() < 1.0:
            # Timeout safety: return current best estimate?
            # Since we are inside the SGS, we can't easily return a partial makespan.
            # But the evaluator kills us anyway.
            # Let's just proceed, the outer loop will handle timeout.
            pass
            
        neg_pri, j = heapq.heappop(heap)
        
        # Calculate earliest start based on predecessors
        t_pred = 0.0
        for p in predecessors[j]:
            if finish_times[p] > t_pred:
                t_pred = finish_times[p]
        
        # Calculate earliest start based on resources
        t_start = find_earliest_start(j, t_pred)
        
        finish_times[j] = t_start + durations[j]
        scheduled.append((t_start, finish_times[j], demands[j]))
        
        # Update successors
        for s in successors[j]:
            remaining_preds[s] -= 1
            if remaining_preds[s] == 0:
                heapq.heappush(heap, (-priority_vec[s], s))
    
    # Makespan is the max finish time
    if len(finish_times) == 0:
        return 0
    return float(np.max(finish_times))


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    
    # Strategy:
    # 1. Calculate "criticality" based on longest path to end (successors).
    # 2. Combine with resource pressure (demand/capacity ratio).
    # 3. Combine with duration (longer tasks are harder to schedule).
    
    # Build successors
    successors = [[] for _ in range(n)]
    for j in range(n):
        if predecessors[j]:
            for p in predecessors[j]:
                successors[p].append(j)
    
    # Calculate longest path from each node to the end (reverse topological order)
    # in_degree for topological sort (original)
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        if predecessors[j]:
            in_degree[j] = len(predecessors[j])
    
    queue = []
    for j in range(n):
        if in_degree[j] == 0:
            queue.append(j)
    
    top_order = []
    while queue:
        u = queue.pop(0)
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Reverse topological order
    rev_top_order = list(reversed(top_order))
    
    # longest_path_to_end[i] = max(durations[i] + longest_path_to_end[j] for j in successors[i])
    longest_path_to_end = np.zeros(n, dtype=float)
    for i in rev_top_order:
        max_succ = 0.0
        for j in successors[i]:
            if durations[j] + longest_path_to_end[j] > max_succ:
                max_succ = durations[j] + longest_path_to_end[j]
        longest_path_to_end[i] = durations[i] + max_succ
    
    # Resource pressure:
    # For each activity, how much of each resource does it use relative to capacity?
    # High demand means it's more constrained.
    # We can sum the ratios for resources it uses.
    # Or take the max ratio.
    # Let's use the max ratio of (demand/capacity) for each activity.
    # If capacity is 0, handle division by zero.
    
    resource_pressure = np.zeros(n, dtype=float)
    for j in range(n):
        max_ratio = 0.0
        for k in range(len(capacities)):
            if capacities[k] > 0:
                ratio = demands[j][k] / capacities[k]
                if ratio > max_ratio:
                    max_ratio = ratio
        resource_pressure[j] = max_ratio
    
    # Combine scores:
    # We want to schedule activities that are on the critical path first.
    # We also want to schedule resource-heavy activities first to avoid bottlenecks?
    # Or light activities to fill gaps?
    # Usually, "Longest Processing Time" (LPT) or "Critical Path" (CP) are good heuristics.
    # A combination: Score = alpha * normalized_critical_path + beta * normalized_duration + gamma * normalized_pressure
    
    # Normalize
    max_cp = np.max(longest_path_to_end) if n > 0 else 1.0
    if max_cp == 0: max_cp = 1.0
    norm_cp = longest_path_to_end / max_cp
    
    max_dur = np.max(durations) if n > 0 else 1.0
    if max_dur == 0: max_dur = 1.0
    norm_dur = durations / max_dur
    
    max_press = np.max(resource_pressure) if n > 0 else 1.0
    if max_press == 0: max_press = 1.0
    norm_press = resource_pressure / max_press
    
    # Weights
    # Critical path is usually the most important.
    score = 0.6 * norm_cp + 0.3 * norm_dur + 0.1 * norm_press
    
    return score
# EVOLVE-BLOCK-END
