"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque


def priority(instance):
    """
    Score each activity based on a resource-bottleneck heuristic derived from
    a max-flow/min-cut analysis on a simplified time-expanded network.
    
    The idea: activities that consume scarce resources relative to capacity
    and have many successors (critical path length) should be prioritized.
    We approximate the 'min cut capacity' impact by combining:
      1. Resource scarcity: sum over resources of (demand / capacity) for each activity
      2. Critical path length: longest path from this activity to the end (successor count weighted)
    
    Higher score = higher priority. The serial SGS will pick the highest-priority
    eligible activity first.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse demands and capacities
    # demands: shape (n, num_resources) or similar
    # capacities: shape (num_resources,)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    # Handle case where demands might be 1D (single resource)
    if demands.ndim == 1:
        demands = demands.reshape(-1, 1)
        capacities = capacities.reshape(-1)
    
    num_resources = capacities.shape[0]
    
    # Compute resource scarcity score: for each activity, sum over resources
    # of (demand / capacity). This quantifies how much of the resource pool
    # the activity consumes.
    # Avoid division by zero
    safe_capacities = np.where(capacities > 0, capacities, 1.0)
    scarcity = np.sum(demands / safe_capacities, axis=1)
    
    # Compute critical path length: longest path from each activity to the sink
    # Build successor graph from predecessors
    predecessors = instance["predecessors"]
    # predecessors[i] is a list of activities that must precede i
    
    # Build adjacency: succ[j] = list of activities that depend on j
    succ = defaultdict(list)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            succ[p].append(i)
    
    # Topological order (Kahn's algorithm)
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
    
    # If topological sort didn't include all nodes (cycle), just use range
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # Compute longest path from each node to the end (reverse topological)
    # longest[i] = durations[i] + max(longest[s] for s in succ[i])
    longest = np.zeros(n)
    # Process in reverse topological order
    for i in reversed(topo_order):
        max_succ = 0
        for s in succ[i]:
            if longest[s] > max_succ:
                max_succ = longest[s]
        longest[i] = durations[i] + max_succ
    
    # Combine: priority = scarcity * (1 + longest)
    # This weights activities that are both resource-heavy and on long paths
    priority_scores = scarcity * (1.0 + longest)
    
    # Normalize to ensure positive scores and reasonable range
    # Add a small epsilon to avoid all zeros
    priority_scores = priority_scores + 1e-6
    
    return priority_scores
# EVOLVE-BLOCK-END
