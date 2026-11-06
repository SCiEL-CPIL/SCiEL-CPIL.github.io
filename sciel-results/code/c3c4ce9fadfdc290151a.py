<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, num_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (num_resources,)
    predecessors = instance["predecessors"]  # list of lists, predecessors[i] = list of pred indices
    
    n = len(durations)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Compute transitive successor count (how many activities depend on this one)
    # Use DFS/BFS from each node - n=30 so this is fine
    def count_successors(start):
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for s in successors[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        return len(visited)
    
    succ_counts = np.array([count_successors(i) for i in range(n)], dtype=float)
    
    # Resource pressure: sum over resources of demand/capacity
    # For each activity i, pressure_i = sum_r (demand[i,r] / capacity[r])
    with np.errstate(divide='ignore', invalid='ignore'):
        resource_pressure = np.sum(demands / capacities[np.newaxis, :], axis=1)
    resource_pressure = np.nan_to_num(resource_pressure, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize each component to [0,1] range for combination
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-9:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    norm_succ = normalize(succ_counts)
    norm_pressure = normalize(resource_pressure)
    norm_duration = normalize(durations)
    
    # Combine: weight successor count heavily, then resource pressure, then duration
    # Activities with many successors and high resource pressure should go first
    score = 0.5 * norm_succ + 0.3 * norm_pressure + 0.2 * norm_duration
    
    return score
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, num_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (num_resources,)
    predecessors = instance["predecessors"]  # list of lists, predecessors[i] = list of pred indices
    
    n = len(durations)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Compute transitive successor count
    def count_successors(start):
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for s in successors[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        return len(visited)
    
    succ_counts = np.array([count_successors(i) for i in range(n)], dtype=float)
    
    # Resource pressure: sum over resources of demand/capacity
    with np.errstate(divide='ignore', invalid='ignore'):
        resource_pressure = np.sum(demands / capacities[np.newaxis, :], axis=1)
    resource_pressure = np.nan_to_num(resource_pressure, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute total float (backward pass)
    # First, compute earliest start times via forward pass
    # Topological order
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
    
    top_order = []
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    while queue:
        node = queue.popleft()
        top_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If topological sort didn't visit all nodes (cycle), use range(n) as fallback
    if len(top_order) < n:
        top_order = list(range(n))
    
    # Forward pass: earliest start/finish
    ES = np.zeros(n)
    EF = np.zeros(n)
    for node in top_order:
        for p in predecessors[node]:
            ES[node] = max(ES[node], EF[p])
        EF[node] = ES[node] + durations[node]
    
    # Project duration
    project_duration = max(EF) if n > 0 else 0
    
    # Backward pass: latest start/finish
    LF = np.full(n, project_duration)
    LS = np.full(n, project_duration)
    # Process in reverse topological order
    for node in reversed(top_order):
        for s in successors[node]:
            LF[node] = min(LF[node], LS[s])
        LS[node] = LF[node] - durations[node]
    
    # Total float
    total_float = LS - ES
    total_float = np.maximum(total_float, 0)
    
    # Resource scarcity per activity: for each resource r, 
    # scarcity contribution = demand[i,r] / capacity[r] * (sum of demands on r / capacity[r])
    # This captures how much an activity contributes to overall resource contention
    with np.errstate(divide='ignore', invalid='ignore'):
        # For each resource, total demand sum
        total_demand_per_resource = np.sum(demands, axis=0)  # shape (num_resources,)
        # Scarcity multiplier: how loaded each resource is
        load_factor = total_demand_per_resource / capacities  # shape (num_resources,)
        load_factor = np.nan_to_num(load_factor, nan=0.0, posinf=0.0, neginf=0.0)
        # Activity's scarcity score = sum_r demand[i,r]/capacity[r] * load_factor[r]
        scarcity = np.sum(demands / capacities[np.newaxis, :] * load_factor[np.newaxis, :], axis=1)
    scarcity = np.nan_to_num(scarcity, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize each component to [0,1] range
    def normalize(x):
        x_min, x_max = x.min(), x_max
        if x_max - x_min < 1e-9:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    norm_succ = normalize(succ_counts)
    norm_pressure = normalize(resource_pressure)
    norm_duration = normalize(durations)
    norm_float = normalize(total_float)  # lower float = more critical = higher priority
    
    # For criticality, we want LOW float to have HIGH priority, so invert
    crit = 1.0 - norm_float
    
    # Component matrix: each row is an activity, columns are [succ, pressure, duration, scarcity, crit]
    components = np.column_stack([norm_succ, norm_pressure, norm_duration, normalize(scarcity), crit])
    
    def simulate_sgs(weights):
        """Serial SGS simulation with given weights. Returns makespan."""
        scores = components @ weights
        # Handle ties deterministically
        order = np.argsort(-scores, kind='stable')
        
        scheduled = [False] * n
        available = set()
        for i in range(n):
            if len(predecessors[i]) == 0:
                available.add(i)
        
        # Track resource usage over time
        # We'll simulate event by event
        # Resource availability: track when each resource unit is free
        # Simpler: at each step, check if resources are available at the earliest start time
        
        current_time = 0.0
        scheduled_count = 0
        remaining = set(range(n))
        
        # Pre-compute for efficiency
        while scheduled_count < n:
            # Find eligible activities (all predecessors scheduled)
            eligible = []
            for i in remaining:
                all_pred_done = all(scheduled[p] for p in predecessors[i])
                if all_pred_done:
                    eligible.append(i)
            
            if not eligible:
                break
            
            # Among eligible, find those that can start at current_time given resource constraints
            # We need to check if resources are available
            # Track resource usage: for simplicity, track end times per resource
            # Use a simpler approach: binary search for earliest start time
            
            # Find the best eligible activity by score
            # But we also need to respect resources
            # Let's find the earliest time each eligible activity can start
            
            best_activity = -1
            best_time = float('inf')
            best_score = -float('inf')
            
            for i in eligible:
                # Find earliest start time for activity i
                # Must be >= current_time and >= all predecessors' finish times
                est = current_time
                for p in predecessors[i]:
                    if scheduled[p]:
                        # We need to track finish times
                        pass
                # Simplified: just use current_time as earliest start
                # Check resource availability at current_time
                # This is a simplification; a proper SGS would check resource profiles
                
                # For now, just pick highest score eligible
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_activity = i
                    best_time = current_time
            
            if best_activity == -1:
                break
            
            # Schedule best_activity
            scheduled[best_activity] = True
            remaining.discard(best_activity)
            scheduled_count += 1
            
            # Update current_time to finish time of this activity
            current_time = best_time + durations[best_activity]
        
        # This simplified simulation is not accurate; let's do a proper one
        return project_duration  # placeholder
    
    # Proper serial SGS simulation
    def simulate_sgs_proper(weights):
        """Proper serial SGS: at each time step, select highest-priority eligible activity."""
        scores = components @ weights
        # Use stable sort for tie-breaking
        priority_order = list(np.argsort(-scores, kind='stable'))
        
        scheduled = [False] * n
        finish_times = [0.0] * n
        remaining = set(range(n))
        makespan = 0.0
        
        # We'll use a time-stepping approach
        # At each step, determine which activities are eligible (all preds done)
        # Among eligible, pick the one with highest score
        # Schedule it at the earliest feasible time considering resources
        
        # Track resource usage as a list of (end_time, amount) per resource
        # For n=30 and 4 resources, we can use a simple approach
        
        # Simpler approach: iterate through activities in priority order,
        # scheduling each at the earliest feasible time
        
        for i in priority_order:
            # Find earliest start time
            est = 0.0
            for p in predecessors[i]:
                est = max(est, finish_times[p])
            
            # Check resource constraints
            # We need to find the earliest time >= est where resources are available
            # Track resource usage intervals
            # For simplicity with small n, use a sweep approach
            
            # Get all relevant time points
            time_points = set([est])
            for t in finish_times:
                if t > est:
                    time_points.add(t)
            
            sorted_times = sorted(time_points)
            
            # Try each time point
            feasible_time = None
            for t in sorted_times:
                if t < est:
                    continue
                # Check if resources are available at time t for duration[i]
                # We need to check that for each resource r,
                # the total demand of activities overlapping [t, t+dur[i]] on resource r
                # plus demand[i][r] <= capacity[r]
                
                ok = True
                for r in range(len(capacities)):
                    total_demand = demands[i][r]
                    for j in range(n):
                        if j == i or not scheduled[j]:
                            continue
                        # Check if activity j overlaps with [t, t+durations[i]]
                        # Activity j runs from start_j to finish_times[j]
                        # We need start_j, which we haven't tracked...
                        
                        # This approach is getting complex. Let me use a different method.
                        pass
                    # Simplified check: assume resources are available
                    # This is not correct but lets us get a baseline
                    if total_demand > capacities[r]:
                        ok = False
                        break
                
                if ok:
                    feasible_time = t
                    break
            
            if feasible_time is None:
                feasible_time = est
            
            finish_times[i] = feasible_time + durations[i]
            scheduled[i] = True
            makespan = max(makespan, finish_times[i])
        
        return makespan
    
    # Even simpler and correct SGS for small n:
    def sgs_simulation(weights):
        """Correct serial SGS for small problem sizes."""
        scores = components @ weights
        priority_order = list(np.argsort(-scores, kind='stable'))
        
        # Track start and finish times
        start_times = [0.0] * n
        finish_times = [0.0] * n
        scheduled = [False] * n
        
        # We'll simulate step by step
        # At each step, find the next activity to schedule
        
        # Use a resource profile approach
        # For each resource, maintain a list of (start, end, amount) intervals
        
        # Simpler: since n=30, we can check feasibility by brute force
        # For each candidate start time, check all active activities
        
        # Actually, let's use the standard serial SGS algorithm:
        # 1. Maintain a set of eligible activities (all predecessors completed)
        # 2. At each time point, from eligible activities, pick the one with highest priority
        # 3. Schedule it at the earliest feasible time
        # 4. Update eligible set and continue
        
        remaining = set(range(n))
        makespan = 0.0
        
        # We need to track when resources become free
        # For each resource r, maintain a list of busy intervals
        
        # Alternative simple approach for small n:
        # Process activities in priority order, scheduling each at earliest feasible time
        
        for idx in priority_order:
            # Earliest start due to precedence
            est = 0.0
            for p in predecessors[idx]:
                est = max(est, finish_times[p])
            
            # Now find earliest time >= est where resources are available
            # We'll check candidate times: est, and finish times of already-scheduled activities
            
            candidate_times = [est]
            for j in range(n):
                if scheduled[j] and finish_times[j] > est:
                    candidate_times.append(finish_times[j])
            
            candidate_times = sorted(set(candidate_times))
            
            assigned_time = None
            for t in candidate_times:
                if t < est:
                    continue
                
                # Check resource feasibility at time t for duration durations[idx]
                feasible = True
                for r in range(len(capacities)):
                    if demands[idx][r] == 0:
                        continue
                    # Sum demands of all scheduled activities that overlap [t, t+durations[idx]]
                    overlap_demand = 0.0
                    for j in range(n):
                        if j == idx or not scheduled[j]:
                            continue
                        # Activity j is scheduled from start_times[j] to finish_times[j]
                        # Check overlap with [t, t + durations[idx]]
                        if start_times[j] < t + durations[idx] and finish_times[j] > t:
                            overlap_demand += demands[j][r]
                    
                    if overlap_demand + demands[idx][r] > capacities[r] + 1e-9:
                        feasible = False
                        break
                
                if feasible:
                    assigned_time = t
                    break
            
            if assigned_time is None:
                # Should not happen if est is valid, but fallback
                assigned_time = est
            
            start_times[idx] = assigned_time
            finish_times[idx] = assigned_time + durations[idx]
            scheduled[idx] = True
            makespan = max(makespan, finish_times[idx])
        
        return makespan
    
    # Now do local search over weights
    # Initial weight: the original heuristic
    best_weights = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    best_makespan = sgs_simulation(best_weights)
    
    # Local search: perturb weights and evaluate
    random.seed(42)
    num_iterations = 0
    
    while time.time() < deadline - 10.0:
        num_iterations += 1
        
        # Perturb weights
        new_weights = best_weights.copy()
        # Add small random perturbation
        for i in range(len(new_weights)):
            new_weights[i] += random.gauss(0, 0.1)
        
        # Ensure non-negative and normalize
        new_weights = np.maximum(new_weights, 0)
        w_sum = new_weights.sum()
        if w_sum < 1e-9:
            new_weights = best_weights.copy()
        else:
            new_weights /= w_sum
        
        # Evaluate
        makespan = sgs_simulation(new_weights)
        
        if makespan < best_makespan - 1e-9:
            best_weights = new_weights
            best_makespan = makespan
    
    # Final score using best weights
    score = components @ best_weights
    return score
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
