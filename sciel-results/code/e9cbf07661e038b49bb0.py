"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import heapq


def priority(instance):
    """
    Compute priority scores for assembly tasks using a combination of:
    1. Task duration (longer tasks first)
    2. Number of immediate successors (tasks that unlock more work first)
    3. Position in the precedence graph (tasks closer to the end first)
    
    Returns an array of priority scores (higher = more priority).
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Parse predecessors
    pred = instance["predecessors"]
    # pred[i] is a list of task indices that must be completed before task i
    
    # Build successor lists
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Compute rank (longest path from start to each task) and reverse rank
    # rank[i] = longest path length from any source to task i
    # Use topological order
    in_degree = [0] * n
    for i in range(n):
        for p in pred[i]:
            in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo_order = []
    in_degree_copy = in_degree[:]
    while q:
        node = q.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree_copy[s] -= 1
            if in_degree_copy[s] == 0:
                q.append(s)
    
    # If not all nodes are in topo_order, there's a cycle; handle gracefully
    if len(topo_order) != n:
        # Fallback: just use task times
        return times
    
    # Compute forward rank: longest path from start to each node
    forward_rank = [0.0] * n
    for node in topo_order:
        for s in succ[node]:
            if forward_rank[node] + times[node] > forward_rank[s]:
                forward_rank[s] = forward_rank[node] + times[node]
    
    # Compute backward rank: longest path from each node to end
    reverse_topo = list(reversed(topo_order))
    backward_rank = [0.0] * n
    for node in reverse_topo:
        for p in pred[node]:
            if backward_rank[node] + times[node] > backward_rank[p]:
                backward_rank[p] = backward_rank[node] + times[node]
    
    # Total path length through each task
    total_path = [forward_rank[i] + times[i] + backward_rank[i] for i in range(n)]
    
    # Number of immediate successors
    num_succ = [len(succ[i]) for i in range(n)]
    
    # Normalize each component
    max_time = max(times) if max(times) > 0 else 1
    max_total_path = max(total_path) if max(total_path) > 0 else 1
    max_num_succ = max(num_succ) if max(num_succ) > 0 else 1
    
    # Combine scores:
    # - Weight by task time (longer tasks are more important)
    # - Weight by total path (tasks on critical paths first)
    # - Weight by number of successors (unlocking more work first)
    
    time_score = times / max_time
    path_score = np.array(total_path) / max_total_path
    succ_score = np.array(num_succ, dtype=float) / max_num_succ
    
    # Priority: higher is better
    # Use a weighted combination
    # Emphasize path length and time, with a smaller boost for successors
    priority_scores = 0.5 * path_score + 0.4 * time_score + 0.1 * succ_score
    
    return priority_scores
# EVOLVE-BLOCK-END
