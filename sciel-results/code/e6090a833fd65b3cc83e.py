"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_longest_path_downstream(times, successors, n):
    """
    For each task, compute the longest path from that task to any sink (downstream).
    This is the critical path length if we start at that task and go forward.
    tasks with longer downstream paths are more 'critical' in terms of tail work.
    """
    # longest_path[i] = times[i] + max(longest_path[j] for j in successors[i])
    # We compute this in topological order from sinks backwards.
    # But we have predecessors given, so let's build successors.
    
    # Actually, let's think about what helps packing.
    # We want to select tasks that:
    # 1. Have long processing times (LPT)
    # 2. Have many successors (unlock more work)
    # 3. Are on critical paths
    
    # A good heuristic is: priority = time[i] + sum of times of immediate successors? 
    # Or: priority = time[i] * (1 + len(successors[i]))?
    # Or: longest path from i to sink (including i's time)
    
    # Let's compute: for each node, the longest path starting from it going downstream.
    # This requires topological sort.
    
    # Build successors from predecessors
    pred = instance_preds  # passed in
    succ = [list() for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Topological sort (Kahn's algorithm)
    in_degree = [len(pred[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If not all nodes processed, there's a cycle - fallback
    if len(topo_order) != n:
        return np.asarray(times, dtype=float)
    
    # Compute longest path from each node to sink (downstream)
    # Process in reverse topological order
    longest_down = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        best = 0.0
        for v in succ[u]:
            if longest_down[v] > best:
                best = longest_down[v]
        longest_down[u] = times[u] + best
    
    return longest_down


def _compute_rank_by_successors(times, successors, n):
    """Count of immediate successors."""
    ranks = np.zeros(n, dtype=float)
    for i in range(n):
        ranks[i] = len(successors[i])
    return ranks


def priority(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Build successors
    succ = [list() for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
    
    # Compute various priority signals
    # 1. Longest downstream path
    longest_down = _compute_longest_path_downstream(times, succ, n)
    
    # 2. Number of immediate successors
    succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        succ_count[i] = len(succ[i])
    
    # 3. Normalized time
    max_time = np.max(times) if n > 0 else 1.0
    if max_time == 0:
        max_time = 1.0
    norm_time = times / max_time
    
    # 4. Normalized longest downstream
    max_ld = np.max(longest_down) if n > 0 else 1.0
    if max_ld == 0:
        max_ld = 1.0
    norm_ld = longest_down / max_ld
    
    # 5. Normalized successor count
    max_sc = np.max(succ_count) if n > 0 else 1.0
    if max_sc == 0:
        max_sc = 1.0
    norm_sc = succ_count / max_sc
    
    # Combine: weight the critical path (longest downstream) heavily, 
    # add a bonus for unlocking more successors
    # Priority = norm_ld * 0.6 + norm_time * 0.3 + norm_sc * 0.1
    priority_scores = norm_ld * 0.6 + norm_time * 0.3 + norm_sc * 0.1
    
    # Check deadline
    if time.time() > deadline:
        return priority_scores
    
    return priority_scores
# EVOLVE-BLOCK-END
