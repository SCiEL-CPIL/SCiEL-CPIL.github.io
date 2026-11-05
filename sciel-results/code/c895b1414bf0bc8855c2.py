"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_bottom_levels(instance):
    """Compute bottom levels (longest path to end) for each activity."""
    n = len(instance["durations"])
    # Build successor map
    successors = [[] for _ in range(n)]
    preds = instance["predecessors"]
    for i, plist in enumerate(preds):
        for p in plist:
            successors[p].append(i)
    
    # Compute bottom levels via reverse topological order
    # First, get topological order
    in_degree = [0] * n
    for i, plist in enumerate(preds):
        in_degree[i] = len(plist)
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Reverse topological order
    bottom = [0.0] * n
    for i in reversed(topo_order):
        max_succ = 0.0
        for s in successors[i]:
            max_succ = max(max_succ, bottom[s])
        bottom[i] = instance["durations"][i] + max_succ
    
    return np.array(bottom, dtype=float)


def _compute_resource_pressure(instance):
    """Compute resource pressure: demand/capacity ratio for each activity, weighted by resource scarcity."""
    n = len(instance["durations"])
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, num_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (num_resources,)
    
    # For each resource, compute how constrained it is
    # Pressure for activity i on resource r = demand[i][r] / capacity[r]
    # Weight by how many activities also demand that resource (contention)
    
    num_resources = demands.shape[1]
    pressure = np.zeros(n, dtype=float)
    
    for r in range(num_resources):
        cap = capacities[r]
        if cap <= 0:
            cap = 1.0
        # Activities that use this resource
        using = demands[:, r] > 0
        count_using = np.sum(using)
        # Higher contention (more activities using same resource) -> higher pressure
        contention_factor = count_using / n if n > 0 else 1.0
        # Demand-to-capacity ratio
        ratio = demands[:, r] / cap
        # Combine: ratio * contention_factor
        pressure += ratio * contention_factor
    
    return pressure


def _simulate_greedy_schedule(instance, priority_scores):
    """Simulate a serial schedule generation scheme to identify bottleneck activities."""
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    
    num_resources = demands.shape[1]
    
    # State
    remaining_preds = [list(p) for p in preds]
    completed = [False] * n
    start_times = [0.0] * n
    end_times = [0.0] * n
    # Track resource usage over time: list of (start, end, resource, amount)
    resource_events = [[] for _ in range(num_resources)]
    
    time_now = 0.0
    scheduled_count = 0
    
    while scheduled_count < n:
        # Find eligible activities
        eligible = [i for i in range(n) if not completed[i] and len(remaining_preds[i]) == 0]
        
        if not eligible:
            # Should not happen with valid precedence, but safety
            break
        
        # Find the earliest time we can start any eligible activity
        # For each eligible activity, compute the earliest feasible start time
        best_activity = -1
        best_start = float('inf')
        best_priority = float('-inf')
        
        for i in eligible:
            # Earliest start based on predecessors
            pred_end_max = 0.0
            for p in preds[i]:
                pred_end_max = max(pred_end_max, end_times[p])
            
            # Check resource feasibility
            feasible_start = pred_end_max
            # We need to find the earliest time >= feasible_start where resources are available
            # For simplicity, we check at feasible_start and increment if needed
            t = feasible_start
            max_checks = 1000
            for _ in range(max_checks):
                # Check if resources are available at time t
                feasible = True
                for r in range(num_resources):
                    if demands[i, r] > 0:
                        # Sum resource usage at time t
                        usage = 0.0
                        for (s, e, amt) in resource_events[r]:
                            if s <= t < e:
                                usage += amt
                        if usage + demands[i, r] > capacities[r] + 1e-9:
                            feasible = False
                            break
                if feasible:
                    break
                # Advance t to next event time
                next_times = []
                for r in range(num_resources):
                    for (s, e, amt) in resource_events[r]:
                        if s <= t < e:
                            next_times.append(e)
                if next_times:
                    t = min(next_times)
                else:
                    t += 1.0
            
            if t < best_start or (t == best_start and priority_scores[i] > best_priority):
                best_start = t
                best_activity = i
                best_priority = priority_scores[i]
        
        if best_activity == -1:
            break
        
        i = best_activity
        start_times[i] = best_start
        end_times[i] = best_start + durations[i]
        completed[i] = True
        scheduled_count += 1
        
        # Add resource events
        for r in range(num_resources):
            if demands[i, r] > 0:
                resource_events[r].append((best_start, end_times[i], demands[i, r]))
        
        # Update remaining predecessors for successors
        for j in range(n):
            if i in remaining_preds[j]:
                remaining_preds[j].remove(i)
    
    makespan = max(end_times) if end_times else 0.0
    return makespan, start_times, end_times


def priority(instance):
    """
    Compute priority scores for activities using resource-weighted criticality.
    
    Combines:
    1. Bottom level (critical path length from activity to end)
    2. Resource pressure (demand-to-capacity ratio weighted by contention)
    3. A greedy simulation to identify which activities actually cause delays
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    
    # Step 1: Compute bottom levels (static criticality)
    bottom = _compute_bottom_levels(instance)
    
    # Step 2: Compute resource pressure
    pressure = _compute_resource_pressure(instance)
    
    # Normalize both to [0, 1] range for combination
    bottom_norm = (bottom - np.min(bottom)) / (np.max(bottom) - np.min(bottom) + 1e-9)
    pressure_norm = (pressure - np.min(pressure)) / (np.max(pressure) - np.min(pressure) + 1e-9)
    
    # Initial combined score: weighted combination
    # Higher bottom level = more critical, higher pressure = more resource-intensive
    combined = 0.6 * bottom_norm + 0.4 * pressure_norm
    
    # Step 3: Run a greedy simulation to identify bottleneck activities
    # Use the initial combined score to simulate, then refine
    if time.time() < deadline - 30.0:
        makespan, start_times, end_times = _simulate_greedy_schedule(instance, combined)
        
        # Identify activities that are on the critical path of the greedy schedule
        # and also have high resource usage
        # An activity is a bottleneck if its start time was delayed due to resource contention
        # We can approximate this by checking if the activity's start time > max pred end time
        
        bottleneck_score = np.zeros(n, dtype=float)
        for i in range(n):
            pred_end_max = 0.0
            for p in preds[i]:
                pred_end_max = max(pred_end_max, end_times[p])
            delay = start_times[i] - pred_end_max
            # If delayed, it's a bottleneck
            if delay > 1e-9:
                bottleneck_score[i] = delay
        
        # Normalize bottleneck score
        if np.max(bottleneck_score) > 1e-9:
            bottleneck_norm = bottleneck_score / np.max(bottleneck_score)
        else:
            bottleneck_norm = np.zeros(n, dtype=float)
        
        # Refine the combined score
        # Activities that are both critical AND bottlenecks get highest priority
        refined = 0.4 * bottom_norm + 0.3 * pressure_norm + 0.3 * bottleneck_norm
        
        # If we have time, do another iteration
        if time.time() < deadline - 15.0:
            makespan2, start_times2, end_times2 = _simulate_greedy_schedule(instance, refined)
            
            bottleneck_score2 = np.zeros(n, dtype=float)
            for i in range(n):
                pred_end_max = 0.0
                for p in preds[i]:
                    pred_end_max = max(pred_end_max, end_times2[p])
                delay = start_times2[i] - pred_end_max
                if delay > 1e-9:
                    bottleneck_score2[i] = delay
            
            if np.max(bottleneck_score2) > 1e-9:
                bottleneck_norm2 = bottleneck_score2 / np.max(bottleneck_score2)
            else:
                bottleneck_norm2 = np.zeros(n, dtype=float)
            
            # Final refined score
            final = 0.35 * bottom_norm + 0.25 * pressure_norm + 0.40 * bottleneck_norm2
            return final
    
    return combined
# EVOLVE-BLOCK-END
