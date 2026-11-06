"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Compute priority scores for assembly line balancing tasks.
    
    Uses a combination of:
    1. Position weight (PW): sum of cycle_time / (pos_weight) where pos_weight
       accounts for the task's position in the precedence graph.
       Specifically, we use the "rank" based approach:
       - For each task, compute its "downstream" complexity (how many tasks it
         must precede, weighted by their durations) to favor tasks that unlock
         many downstream tasks early.
       
    2. Task duration: longer tasks get higher priority to fill stations efficiently.
    
    3. Predecessor count: tasks with fewer predecessors are more likely to be
       eligible early, so they get a slight boost.
    
    The combined score is:
        score = duration + alpha * downstream_weight + beta * (1 / (1 + num_predecessors))
    
    Where downstream_weight is computed as the sum of durations of all tasks
    that this task directly or indirectly precedes (i.e., its "successor tree"
    total duration), normalized appropriately.
    
    This encourages selecting tasks that are both long (to pack stations well)
    and that unlock many subsequent tasks (to keep the pipeline flowing).
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]  # list of lists: predecessors[i] = list of task indices that must come before i
    n = len(times)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute downstream weight: total duration of all tasks that i precedes (transitively)
    # Use topological order (reverse of predecessors)
    # Compute in-degree
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(predecessors[i])
    
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
    
    # If topological sort didn't include all nodes (cycle), fallback
    if len(topo_order) < n:
        # Fallback: just use duration-based priority
        return times.copy()
    
    # Process in reverse topological order to compute downstream weights
    downstream_weight = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        # downstream_weight[node] = times[node] + sum of downstream_weight[succ] for succ in successors[node]
        # But we want the total duration of tasks that node precedes (not including itself)
        # Let's compute: total_downstream[node] = sum of times[succ] + total_downstream[succ] for all succ in successors[node]
        total = 0.0
        for succ in successors[node]:
            total += times[succ] + downstream_weight[succ]
        downstream_weight[node] = total
    
    # Normalize downstream weight to be comparable to times
    max_dw = np.max(downstream_weight) if n > 0 else 1.0
    if max_dw > 0:
        normalized_dw = downstream_weight / max_dw
    else:
        normalized_dw = np.zeros(n, dtype=float)
    
    # Predecessor count
    pred_count = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Combine scores
    alpha = 1.0  # weight for downstream
    beta = 0.5   # weight for predecessor count boost
    
    # Score: duration + alpha * normalized_downstream + beta * (1 / (1 + pred_count))
    scores = times + alpha * normalized_dw + beta / (1.0 + pred_count)
    
    return scores
# EVOLVE-BLOCK-END
