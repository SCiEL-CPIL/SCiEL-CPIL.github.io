"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Compute priority scores for activities using critical path (longest path to end).
    Higher priority means the activity should be scheduled earlier.
    """
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors
    preds = instance.get("predecessors", None)
    if preds is None:
        # Assume no precedence
        return durations.copy()
    
    # preds[i] is a list of predecessor indices for activity i
    # Build successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
    
    # Longest path from each activity to the end (backward pass)
    # longest_path[i] = duration[i] + max(longest_path[j] for j in successors[i])
    # Use topological order (reverse of topological sort)
    # Since we have predecessors, we can do a reverse topological sort
    
    # Compute in-degrees for topological sort (forward)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If not all nodes are in topo_order, there's a cycle; fallback to duration
    if len(topo_order) != n:
        return durations.copy()
    
    # Reverse topological order for backward pass
    rev_topo = list(reversed(topo_order))
    
    longest_path = np.zeros(n, dtype=float)
    for i in rev_topo:
        max_succ = 0.0
        for j in successors[i]:
            if longest_path[j] > max_succ:
                max_succ = longest_path[j]
        longest_path[i] = durations[i] + max_succ
    
    # Use longest_path as priority (higher = more critical)
    # Add a small resource pressure component: activities that demand more of scarce resources
    # get a slight boost to ensure they're considered early
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        # Resource scarcity ratio: demand / capacity for each resource
        # Sum over resources, weighted by duration
        if demands.ndim == 2 and capacities.ndim == 1:
            # demands[i, r] is demand of activity i for resource r
            # scarcity[r] = 1 / capacity[r]
            with np.errstate(divide='ignore', invalid='ignore'):
                scarcity = 1.0 / np.where(capacities > 0, capacities, 1.0)
            # resource_pressure[i] = sum over r of demands[i, r] * scarcity[r]
            resource_pressure = demands @ scarcity
            # Normalize resource pressure to [0, 1]
            rp_max = resource_pressure.max()
            if rp_max > 0:
                resource_pressure = resource_pressure / rp_max
            # Combine: critical path + small resource pressure component
            priority_scores = longest_path + 0.1 * resource_pressure
        else:
            priority_scores = longest_path
    else:
        priority_scores = longest_path
    
    # Safety: check time
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Fallback to simple duration if something went wrong
        return durations.copy()
    
    return priority_scores
# EVOLVE-BLOCK-END
