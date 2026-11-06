"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    Uses a combination of longest processing time (LPT), critical path length,
    and resource pressure.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin below 120s limit
    
    # Extract instance data
    N = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # Shape (N, R)
    capacities = np.asarray(instance["capacities"], dtype=float)  # Shape (R,)
    
    # Handle predecessors: assume list of lists where preds[i] is list of indices
    # Or a matrix? Let's assume standard format.
    if "predecessors" in instance:
        preds_data = instance["predecessors"]
        # If it's a list of lists
        if isinstance(preds_data, list) and (not preds_data or isinstance(preds_data[0], (list, tuple, np.ndarray))):
            preds = [list(p) if p else [] for p in preds_data]
        else:
            # Maybe it's a matrix or something else, try to handle gracefully
            try:
                preds_matrix = np.asarray(preds_data)
                if preds_matrix.ndim == 2:
                    preds = [list(np.where(preds_matrix[i] != 0)[0]) for i in range(N)]
                else:
                    preds = [[] for _ in range(N)]
            except:
                preds = [[] for _ in range(N)]
    else:
        preds = [[] for _ in range(N)]
    
    # 1. Compute Critical Path Length using both forward and backward passes
    successors = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            successors[p].append(i)
    
    # Topological order for computation
    in_degree = [0] * N
    for i in range(N):
        for p in preds[i]:
            in_degree[i] += 1
            
    from collections import deque
    queue = deque([i for i in range(N) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If topo_order is incomplete, there's a cycle, handle gracefully
    if len(topo_order) < N:
        # Fallback to simple LPT
        return durations.copy()
    
    # Forward pass: earliest start times
    est = np.zeros(N)
    for i in topo_order:
        if preds[i]:
            est[i] = max(est[p] + durations[p] for p in preds[i])
        # else est[i] remains 0
    
    # Backward pass: latest finish times
    # First compute project duration (makespan without resource constraints)
    project_duration = max(est[i] + durations[i] for i in range(N))
    
    # Backward pass: latest start times
    lst = np.full(N, project_duration)
    for i in reversed(topo_order):
        if successors[i]:
            lst[i] = min(lst[s] - durations[i] for s in successors[i])
        # else lst[i] remains project_duration
    
    # Total float: lst[i] - est[i]
    # Critical path length for activity i: est[i] + durations[i] + (project_duration - lst[i] - durations[i])
    # Actually, the "criticality" of an activity is better measured by its position on the critical path
    # Use: criticality[i] = est[i] + durations[i] + max(0, project_duration - lst[i])
    # But a simpler and effective metric: the length of the longest path through activity i
    # longest_path_through_i = est[i] + durations[i] + (project_duration - lst[i])
    
    # A better metric: combine forward and backward path lengths
    # Forward path to i: est[i]
    # Duration of i: durations[i]  
    # Backward path from i: project_duration - lst[i]
    # Total path length through i: est[i] + durations[i] + (project_duration - lst[i])
    # But this is constant for all activities on the critical path
    # Instead, use the "critical path length" as the sum of durations on the longest path from source to i and from i to sink
    # This is: est[i] + durations[i] + (project_duration - lst[i])
    # For activities not on the critical path, this will be less than project_duration
    
    # Actually, let's use a simpler and well-known heuristic: 
    # The "critical path length" for activity i is the length of the longest path from the start to i plus from i to the end
    # This is exactly: est[i] + durations[i] + (project_duration - lst[i])
    # But we want activities with HIGHER priority to be scheduled first
    # Activities on the critical path should have higher priority
    # The criticality can be measured by how close the activity is to being on the critical path
    # Use: cpl[i] = est[i] + durations[i] + (project_duration - lst[i])
    # This gives the total path length through activity i
    # Activities on the critical path have cpl[i] = project_duration
    # Activities with more slack have smaller cpl[i]
    
    cpl = est + durations + (project_duration - lst)
    
    # Normalize: activities on critical path get highest score
    # This is a better measure of criticality than just backward pass
    
    # 2. Compute Resource Pressure
    # For each activity, compute the ratio of its demand to capacity for each resource
    # and take the maximum (bottleneck resource)
    # Also consider the "resource scarcity" - how many other activities compete for the same resources
    if demands.ndim == 2:
        # Demand ratio: how much of each resource this activity uses relative to capacity
        demand_ratio = demands / capacities
        resource_pressure = np.max(demand_ratio, axis=1)
        
        # Additionally, compute a "resource contention" metric:
        # For each activity, sum the demand ratios across all resources, weighted by how scarce each resource is
        # This captures activities that use multiple scarce resources
        # Weight by inverse capacity (scarcer resources get higher weight)
        scarcity_weights = 1.0 / capacities
        resource_contention = np.sum(demand_ratio * scarcity_weights, axis=1)
        # Combine: max ratio captures the bottleneck, sum captures overall resource usage
        resource_pressure_combined = 0.6 * resource_pressure + 0.4 * (resource_contention / np.max(resource_contention))
    else:
        # If demands is 1D, assume single resource
        resource_pressure = demands / capacities
        resource_pressure_combined = resource_pressure.copy()
    
    # 3. Combine scores
    # Normalize each component to [0, 1]
    if np.max(cpl) > 0:
        cpl_norm = cpl / np.max(cpl)
    else:
        cpl_norm = np.zeros(N)
    
    if np.max(durations) > 0:
        dur_norm = durations / np.max(durations)
    else:
        dur_norm = np.zeros(N)
    
    if np.max(resource_pressure_combined) > 0:
        rp_norm = resource_pressure_combined / np.max(resource_pressure_combined)
    else:
        rp_norm = np.zeros(N)
    
    # Weighted combination
    # Critical path length should dominate as it directly affects makespan
    # Duration is correlated with CPL but adds information for activities with same CPL
    # Resource pressure helps break ties and accounts for resource contention
    alpha = 0.6  # Weight for CPL (increased - most important for makespan)
    beta = 0.25  # Weight for Duration (reduced - partially redundant with CPL)
    gamma = 0.15  # Weight for Resource Pressure (reduced - secondary effect)
    
    scores = alpha * cpl_norm + beta * dur_norm + gamma * rp_norm
    
    # Check time
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
