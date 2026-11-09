"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Score each assembly task for the greedy line balancing heuristic.
    
    The evaluator fills stations by repeatedly picking the highest-priority
    eligible task that fits the remaining cycle time. We want to minimize
    the number of stations.
    
    Strategy based on the research directive:
    - We cannot dynamically set priority per-station because the function
      is called once per instance and returns a static score vector.
    - Instead, we approximate the 'tightest fit first' intuition with a
      static heuristic that favors tasks that are 'critical' in the sense
      of having high downstream workload (rank) and high processing time.
    
    A strong static heuristic for line balancing is to use a combination of:
    1. Rank (longest path from task to end / or to start) - prioritizes critical path tasks.
    2. Processing time - larger tasks are harder to place, so place them first.
    3. Number of successors - tasks with fewer successors are more constrained in placement later? 
       Actually, tasks with more successors (high 'rank' if defined from end) should go first.
    
    Let's compute the 'rank' as the length of the longest path from the task to the sink
    (i.e., the minimum time remaining after this task if we follow the critical path).
    Alternatively, the standard heuristic is 'Ranked Positional Weight' (RPW) or 'Longest Processing Time' (LPT)
    combined with precedence.
    
    Common effective static heuristics:
    - RPW (Ranked Positional Weight): For each task, sum of its own time + times of all its successors.
      Higher RPW means the task is on the critical path and has more work following it.
      Prioritizing high RPW ensures critical path tasks are placed early.
    
    Let's implement RPW.
    
    Steps:
    1. Build adjacency list from predecessors.
    2. Compute RPW for each task using topological sort (reverse topological order).
    3. Return RPW as the priority.
    
    Wait, the directive says: "priority score for each task is dynamically set to the negative of its 'slack'... forcing the greedy selection to pack tightest tasks first".
    Since we can't be dynamic, we approximate 'tightest' by favoring tasks that have little slack in the critical path.
    RPW is a good proxy for criticality.
    
    Another option: Priority = RPW + time. Or just RPW.
    Usually, RPW alone is very strong.
    
    Let's try:
    priority[i] = rpW[i]
    
    If two tasks have the same RPW, maybe break ties by time? But the evaluator picks highest priority first.
    If we return just RPW, ties are broken arbitrarily (or by index). 
    To make it more robust, we can add a small fraction of the time to break ties, favoring larger tasks.
    priority[i] = rpW[i] + 0.001 * time[i]
    
    Let's implement RPW calculation.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    
    # Build successor list
    # preds[i] is a list of predecessors of task i
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    
    # Compute RPW (Ranked Positional Weight)
    # rpW[i] = time[i] + sum(rpW[j] for j in succs[i])
    # This requires processing in reverse topological order.
    
    # Topological sort
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1
            
    from collections import deque
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    topo_order = []
    in_degree_temp = in_degree[:]
    while q:
        node = q.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_degree_temp[s] -= 1
            if in_degree_temp[s] == 0:
                q.append(s)
    
    # If graph has cycles (shouldn't), handle gracefully
    if len(topo_order) != n:
        # Fallback to simple time-based priority
        return times
    
    # Compute RPW in reverse topological order
    rpW = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        rpW[i] = times[i]
        for s in succs[i]:
            rpW[i] += rpW[s]
    
    # Add small time component to break ties, favoring larger tasks
    # This approximates the 'tightest fit' by ensuring large tasks (which are harder to pack)
    # are considered with high priority, especially if they have similar RPW.
    priority_scores = rpW + 0.001 * times
    
    return priority_scores
# EVOLVE-BLOCK-END
