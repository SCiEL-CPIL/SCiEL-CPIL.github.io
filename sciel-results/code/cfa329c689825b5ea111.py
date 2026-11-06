"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque


def priority(instance):
    """
    Score each activity using a time-expanded min-cut bottleneck heuristic.
    
    This approach models resource contention in a time-expanded network where
    each activity's priority reflects the minimum cut capacity impact of starting
    that activity at its earliest feasible time. We approximate this by:
    1. Computing earliest start times (ES) via forward pass.
    2. Building a time-expanded view of resource usage over the project horizon.
    3. For each activity, computing the 'bottleneck impact' as the product of
       its resource demand weighted by the peak resource saturation at its
       earliest start window, multiplied by its critical path length.
    
    This prioritizes activities that create the tightest resource cuts when
    scheduled at their earliest opportunity.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse demands and capacities
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    # Handle case where demands might be 1D (single resource)
    if demands.ndim == 1:
        demands = demands.reshape(-1, 1)
        capacities = capacities.reshape(-1)
    
    num_resources = capacities.shape[0]
    predecessors = instance["predecessors"]
    
    # Build successor graph
    succ = defaultdict(list)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            succ[p].append(i)
    
    # Topological order
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # Forward pass: compute earliest start times (ES)
    es = np.zeros(n)
    for i in topo_order:
        preds = predecessors[i] if i < len(predecessors) else []
        if preds:
            es[i] = max(es[p] + durations[p] for p in preds)
    
    # Backward pass: compute latest finish times and critical path lengths
    # First, compute the project makespan (critical path length)
    project_end = max(es[i] + durations[i] for i in range(n)) if n > 0 else 0
    
    # Longest path from each node to end (critical path contribution)
    longest_to_end = np.zeros(n)
    for i in reversed(topo_order):
        max_succ = 0
        for s in succ[i]:
            if longest_to_end[s] > max_succ:
                max_succ = longest_to_end[s]
        longest_to_end[i] = durations[i] + max_succ
    
    # Build time-expanded resource profile
    # Use discrete time steps (integer times) for efficiency
    max_time = int(np.ceil(project_end)) if project_end > 0 else 1
    
    # resource_usage[t, r] = total demand of activities active at time t for resource r
    # We'll use a difference array for efficiency
    # For each activity i, it occupies [es[i], es[i] + durations[i])
    # We track resource usage over time
    
    # For each resource, compute the peak utilization ratio over time
    # and the average utilization during each activity's window
    
    # Efficient approach: for each activity, compute the average resource
    # saturation across its duration window
    
    # First, build the resource usage profile using difference arrays
    # resource_diff[t, r] stores the change in usage at time t
    resource_diff = np.zeros((max_time + 2, num_resources), dtype=float)
    
    for i in range(n):
        start = int(np.floor(es[i]))
        end = int(np.ceil(es[i] + durations[i]))
        if end > start and start < max_time:
            for r in range(num_resources):
                if demands[i, r] > 0:
                    resource_diff[start, r] += demands[i, r]
                    if end < max_time + 1:
                        resource_diff[end, r] -= demands[i, r]
    
    # Compute cumulative resource usage over time
    resource_usage = np.zeros((max_time + 1, num_resources), dtype=float)
    for t in range(1, max_time + 1):
        resource_usage[t, :] = resource_usage[t - 1, :] + resource_diff[t, :]
    
    # Compute resource saturation ratio at each time step
    safe_capacities = np.where(capacities > 0, capacities, 1.0)
    resource_saturation = resource_usage / safe_capacities  # shape: (max_time+1, num_resources)
    
    # For each activity, compute the bottleneck impact:
    # - Average resource saturation across its duration window
    # - Weighted by the activity's own demand profile
    # - Multiplied by its critical path length
    
    priority_scores = np.zeros(n)
    
    for i in range(n):
        start = int(np.floor(es[i]))
        end = int(np.ceil(es[i] + durations[i]))
        if end <= start:
            end = start + 1
        if start < 0:
            start = 0
        if end > max_time:
            end = max_time
        
        if end <= start:
            # Activity has zero duration or is outside range
            avg_sat = 0.0
        else:
            # Average saturation across the activity's window
            window_sat = resource_usage[start:end, :]  # shape: (window_len, num_resources)
            # Weighted average by the activity's demand
            if np.sum(demands[i]) > 0:
                # For each resource, compute average saturation in window
                avg_sat_per_r = np.mean(window_sat, axis=0)
                # Weight by the activity's demand for each resource
                avg_sat = np.sum(demands[i] * avg_sat_per_r)
            else:
                avg_sat = 0.0
        
        # Bottleneck impact = critical path length * resource bottleneck intensity
        # Use a combination: the activity's own demand creates a "cut" proportional
        # to how saturated the resources are during its execution window
        bottleneck = longest_to_end[i] * (1.0 + avg_sat)
        
        priority_scores[i] = bottleneck
    
    # Ensure non-negative and add small epsilon for tie-breaking
    priority_scores = np.maximum(priority_scores, 0.0) + 1e-6
    
    return priority_scores
# EVOLVE-BLOCK-END
