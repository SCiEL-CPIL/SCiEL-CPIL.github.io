"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Compute priority scores for assembly line balancing tasks.
    
    Uses a combination of positional weight (critical path) and 
    successor count (buckling index) to prioritize tasks.
    
    Returns: array of priority scores (higher = more urgent to schedule)
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Parse predecessors
    # predecessors[i] = list of task indices that must come before task i
    predecessors = instance["predecessors"]
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
    
    # Compute positional weight: time[i] + sum of all successors' positional weights
    # Use topological order (reverse topological for successors)
    # Since we have predecessors, we can compute in forward topological order
    # for successors, or just use iterative relaxation
    
    # Compute out-degree (number of immediate successors)
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Compute positional weight via iterative method (handles any DAG)
    # pos_weight[i] = times[i] + sum(pos_weight[j] for j in successors[i])
    # We need reverse topological order. Let's compute in-degree first.
    
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If not all nodes are in topo_order, there's a cycle - fall back to times
    if len(topo_order) < n:
        return times.copy()
    
    # Compute positional weights in reverse topological order
    pos_weight = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        pos_weight[node] = times[node]
        for succ in successors[node]:
            pos_weight[node] += pos_weight[succ]
    
    # Buckling index: positional_weight / (1 + out_degree)
    # This is a well-known heuristic for assembly line balancing
    buckling = pos_weight / (1.0 + out_degree)
    
    # Also consider a composite: emphasize tasks that are both critical 
    # and have few successors (to avoid blocking)
    # Normalize components
    max_pos = np.max(pos_weight) if n > 0 else 1.0
    max_out = np.max(out_degree) if n > 0 else 1.0
    
    # Primary: buckling index (proven effective for ALB)
    # Secondary: add a small bonus for tasks with high positional weight 
    # but low out-degree (critical path tasks that don't branch)
    
    # Use buckling index as the primary priority
    priority_scores = buckling.copy()
    
    # Add a tie-breaking component: slight preference for longer tasks 
    # among similar buckling indices
    if max_pos > 0:
        priority_scores += 0.01 * (times / max_pos)
    
    return priority_scores
# EVOLVE-BLOCK-END
