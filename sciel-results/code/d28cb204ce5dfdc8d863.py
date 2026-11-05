import numpy as np
from collections import deque
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(durations)
    
    # Handle case where n is 0
    if n == 0:
        return np.array([])
    
    # Extract resource info
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    n_resources = len(capacities) if capacities is not None else 0
    
    # Build successors list and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological Sort (Kahn's Algorithm)
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If there's a cycle, topo_order will be incomplete. 
    # RCPSP instances are DAGs, so this shouldn't happen.
    # But just in case, if len(topo_order) < n, we might have issues.
    # For safety, if incomplete, we fall back to LDF?
    if len(topo_order) < n:
         return durations

    # Calculate Longest Path to End (LD)
    LD = np.zeros(n)
    # Process in reverse topological order
    for u in reversed(topo_order):
        max_succ_LD = 0.0
        for v in successors[u]:
            if LD[v] > max_succ_LD:
                max_succ_LD = LD[v]
        LD[u] = durations[u] + max_succ_LD
    
    # Calculate demand-to-capacity ratio (resource pressure)
    # Higher ratio means more likely to cause resource contention
    resource_pressure = np.zeros(n)
    if demands is not None and capacities is not None and n_resources > 0:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        # For each activity, compute max demand/capacity ratio across resources
        # This identifies activities that are most likely to bottleneck resources
        for i in range(n):
            if i < len(demands_arr):
                for r in range(n_resources):
                    if capacities_arr[r] > 0:
                        ratio = demands_arr[i][r] / capacities_arr[r]
                        if ratio > resource_pressure[i]:
                            resource_pressure[i] = ratio
    
    # Dynamic resource-contention priority:
    # For each activity, compute the maximum resource utilization ratio among all
    # activities that share at least one resource with it. This captures the
    # competitive pressure each activity faces.
    
    # Build a resource-sharing graph: for each activity, find all other activities
    # that share at least one resource with non-zero demand
    resource_sharing_pressure = np.zeros(n)
    
    if demands is not None and capacities is not None and n_resources > 0:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        # For each resource, find which activities use it (non-zero demand)
        # Then for each activity, the sharing pressure is the max demand/capacity
        # ratio among all activities that share any resource with it
        
        # Precompute for each activity: which resources it uses
        # And for each resource: which activities use it
        
        # To keep it efficient, we'll compute a "resource contention score"
        # that reflects how much competition an activity faces
        
        # For each activity i, contention = max over all j that share a resource with i
        # of (demand[j][r] / capacity[r]) for the shared resource r
        
        # This is O(n^2 * R) in worst case, but n=30, R=4, so it's fine
        
        for i in range(n):
            if i >= len(demands_arr):
                continue
            max_contention = 0.0
            for j in range(n):
                if i == j or j >= len(demands_arr):
                    continue
                # Check if i and j share any resource
                for r in range(n_resources):
                    if demands_arr[i][r] > 0 and demands_arr[j][r] > 0:
                        if capacities_arr[r] > 0:
                            ratio = demands_arr[j][r] / capacities_arr[r]
                            if ratio > max_contention:
                                max_contention = ratio
            resource_sharing_pressure[i] = max_contention
    
    # Normalize all components
    LD_min, LD_max = LD.min(), LD.max()
    LD_range = LD_max - LD_min if LD_max > LD_min else 1.0
    LD_norm = (LD - LD_min) / LD_range
    
    rp_min, rp_max = resource_pressure.min(), resource_pressure.max()
    rp_range = rp_max - rp_min if rp_max > rp_min else 1.0
    rp_norm = (resource_pressure - rp_min) / rp_range
    
    rsp_min, rsp_max = resource_sharing_pressure.min(), resource_sharing_pressure.max()
    rsp_range = rsp_max - rsp_min if rsp_max > rsp_min else 1.0
    rsp_norm = (resource_sharing_pressure - rsp_min) / rsp_range
    
    # Weighted combination: 
    # - Critical path (LD): 50%
    # - Own resource pressure: 20%
    # - Resource sharing/contention pressure: 30%
    # This emphasizes both criticality and the competitive environment
    combined_score = 0.5 * LD_norm + 0.2 * rp_norm + 0.3 * rsp_norm
    
    # Add tiny tie-breaker for numerical stability
    return combined_score + durations * 1e-9
