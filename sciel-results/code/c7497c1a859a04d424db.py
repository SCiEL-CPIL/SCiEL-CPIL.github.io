"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task using a combination of:
    1. Longest downstream path (critical path length from task to end)
    2. Task duration
    3. Number of immediate successors (fan-out)
    
    This 'Critical Path Density' approach prioritizes tasks that are
    bottlenecks on the critical path and also high-fanout branching points,
    which should reduce downstream fragmentation and station count.
    """
    start_time = time.time()
    DEADLINE = 120.0  # Safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)
    predecessors = instance.get("predecessors", [])
    
    # Build adjacency: successors[i] = list of tasks that depend on i
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    
    for i, preds in enumerate(predecessors):
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Compute longest path from each task to any sink (downstream critical path length)
    # Use topological order via Kahn's algorithm
    from collections import deque
    
    # Topological sort
    topo_order = []
    in_deg = in_degree.copy()
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)
    
    # If not all nodes are in topo_order, there's a cycle; fall back to simple ordering
    if len(topo_order) < n:
        # Fallback: just use task duration as priority
        return times.copy()
    
    # Compute longest downstream path for each node (reverse topological order)
    longest_downstream = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in successors[node]:
            if longest_downstream[succ] > max_succ:
                max_succ = longest_downstream[succ]
        longest_downstream[node] = times[node] + max_succ
    
    # Compute number of immediate successors (fan-out)
    fan_out = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Compute number of immediate predecessors (fan-in)
    fan_in = np.array([len(predecessors[i]) if i < len(predecessors) else 0 for i in range(n)], dtype=float)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        # Return simple priority if we're running out of time
        return longest_downstream.copy()
    
    # Priority score: Critical Path Density + fan-out
    # Critical Path Density = task duration / longest downstream path
    # This identifies bottleneck tasks (high duration relative to their downstream path)
    # Combined with fan-out to prioritize branching points that cause fragmentation
    
    # Compute critical path density: ratio of task time to total downstream path
    # Avoid division by zero by adding epsilon
    eps = 1e-8
    cpd = times / (longest_downstream + eps)
    
    # Also compute a "position" metric: how much of the total path this task represents
    # This captures tasks that are a large fraction of their downstream path
    total_path = np.max(longest_downstream) if n > 0 else 1.0
    if total_path == 0:
        total_path = 1.0
    
    # Normalize components
    max_cpd = np.max(cpd) if n > 0 else 1.0
    max_fanout = np.max(fan_out) if n > 0 else 1.0
    
    if max_cpd == 0: max_cpd = 1.0
    if max_fanout == 0: max_fanout = 1.0
    
    norm_cpd = cpd / max_cpd
    norm_fanout = fan_out / max_fanout
    
    # Also use the raw longest downstream path (normalized) as a tiebreaker
    # Tasks with longer downstream paths should be scheduled earlier
    norm_ld = longest_downstream / total_path
    
    # Combined score: 
    # - Critical path density (bottleneck identification): weight 1.0
    # - Fan-out (branching points): weight 0.4
    # - Downstream path length (tiebreaker): weight 0.3
    score = 1.0 * norm_cpd + 0.4 * norm_fanout + 0.3 * norm_ld
    
    return score
# EVOLVE-BLOCK-END
