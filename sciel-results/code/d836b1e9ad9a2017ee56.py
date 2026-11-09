"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores for each task using a 'bottleneck density' heuristic.
    
    The idea:
    - For each task, compute the longest path (in time) from that task to any
      successor (i.e., the critical path length downstream).
    - Also compute the minimum number of remaining stations needed to schedule
      the downstream subtree, approximated by ceil(downstream_time / cycle_time),
      where downstream_time is the total processing time in the downstream subtree
      (or the longest path, whichever is more conservative).
    - The priority is the ratio of the task's own processing time to this
      minimum number of remaining stations, which penalizes long tasks that
      sit on critical paths where station capacity is scarce.
    
    We also add a small tie-breaker based on the number of immediate successors
    (tasks with more successors are more "critical" and should be scheduled earlier).
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Build adjacency list: predecessors[i] is a list of task indices that must come before i
    preds = instance.get("predecessors", [None] * n)
    
    # Build successor list: succ[i] = list of tasks that depend on i
    succ = [[] for _ in range(n)]
    for i in range(n):
        if preds[i] is not None:
            for p in preds[i]:
                succ[p].append(i)
    
    # Compute longest path from each node to any descendant (critical path downstream)
    # Use topological order
    in_degree = [0] * n
    for i in range(n):
        if preds[i] is not None:
            for p in preds[i]:
                in_degree[i] += 1
    
    # Topological sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succ[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)
    
    # Compute longest path from each node to a leaf (downstream critical path)
    longest_down = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for node in reversed(topo_order):
        if succ[node]:
            longest_down[node] = times[node] + max(longest_down[s] - times[s] if s in [x for x in succ[node]] else 0 for s in succ[node]) if False else times[node] + max(longest_down[s] for s in succ[node])
        else:
            longest_down[node] = times[node]
    
    # Recompute more carefully: longest_down[node] = times[node] + max(longest_down[s] for s in succ[node]) if succ[node] else times[node]
    # But we want the path from node to the end, so:
    longest_down = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if succ[node]:
            longest_down[node] = times[node] + max(longest_down[s] for s in succ[node])
        else:
            longest_down[node] = times[node]
    
    # Compute total downstream time (sum of all tasks in the subtree rooted at node)
    # This is more expensive but gives a better estimate of required stations
    subtree_time = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        subtree_time[node] = times[node] + sum(subtree_time[s] for s in succ[node])
    
    # Minimum number of remaining stations needed for the downstream subtree
    # Approximate as ceil(subtree_time[node] / cycle_time), but at least 1
    min_remaining_stations = np.maximum(1.0, np.ceil(subtree_time / cycle_time))
    
    # Bottleneck density: ratio of task time to min remaining stations
    # Higher values indicate tasks that are long relative to the available station capacity
    bottleneck_density = times / min_remaining_stations
    
    # Add a small bonus for tasks with many successors (they are more critical)
    # Normalize successor count
    succ_counts = np.array([len(s) for s in succ], dtype=float)
    max_succ = succ_counts.max() if n > 0 else 1.0
    if max_succ > 0:
        succ_bonus = succ_counts / max_succ
    else:
        succ_bonus = np.zeros(n)
    
    # Final priority: bottleneck density + small weight on successor count
    # The successor bonus helps break ties in favor of more critical tasks
    priority_scores = bottleneck_density + 0.1 * succ_bonus
    
    return priority_scores
# EVOLVE-BLOCK-END
