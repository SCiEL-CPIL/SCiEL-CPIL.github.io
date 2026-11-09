"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Score each assembly task using a slack-minimization approach inspired by
    bipartite matching between tasks and station slots.
    
    We simulate a greedy LPT assignment to estimate the remaining capacity
    when each task would typically be placed. Tasks that would leave little
    slack (tight fit) get higher priority, forcing the actual greedy solver
    to pack tightly and minimize wasted capacity.
    
    The priority combines:
    1. Estimated negative slack from simulated placement (primary)
    2. Processing time (LPT) as tiebreaker
    3. Critical path position (RPW) to respect precedence
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    
    if n == 0:
        return np.array([])
    
    # Build successor list
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            if p < n:
                succs[p].append(i)
    
    # Compute in-degree and topological order
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            if p < n:
                in_degree[i] += 1
    
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
    
    # Fallback if cycle detected
    if len(topo_order) != n:
        return times.copy()
    
    # Compute RPW (Ranked Positional Weight)
    rpW = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        rpW[i] = times[i]
        for s in succs[i]:
            rpW[i] += rpW[s]
    
    # Simulate greedy LPT assignment to estimate remaining capacity per task
    # Sort tasks by time descending (LPT)
    lpt_order = np.argsort(-times)
    
    # Track which station each task would be placed in
    task_station = [-1] * n
    station_loads = []  # list of (load, set_of_tasks)
    
    # Greedy: for each task in LPT order, place in first feasible station
    # that has all predecessors already placed
    placed = [False] * n
    # We need to respect precedence: a task can only be placed if all its
    # predecessors are placed. So we iterate in topological order, but among
    # eligible tasks, pick the one with highest time (LPT-like).
    
    # Better approach: simulate the actual greedy process
    # Eligible tasks are those whose predecessors are all placed
    # Among eligible, pick highest priority (we'll use LPT for simulation)
    
    eligible = []
    for i in range(n):
        if in_degree[i] == 0:
            eligible.append(i)
    
    # Use a max-heap by time for LPT selection among eligible
    import heapq
    # Python heapq is min-heap, so negate times
    heap = [(-times[i], i) for i in eligible]
    heapq.heapify(heap)
    
    placed = [False] * n
    remaining_preds = in_degree[:]
    station_loads = []  # list of current loads
    task_remaining = [0.0] * n  # estimated remaining capacity when task is placed
    
    while heap:
        neg_time, task = heapq.heappop(heap)
        time_val = -neg_time
        
        # Find best station (tightest fit) or create new
        best_station = -1
        best_remaining = float('inf')
        
        for si, load in enumerate(station_loads):
            rem = cycle_time - load
            if rem >= time_val:
                if rem < best_remaining:
                    best_remaining = rem
                    best_station = si
        
        if best_station == -1:
            # Create new station
            best_station = len(station_loads)
            station_loads.append(0.0)
            best_remaining = cycle_time
        
        # Place task
        station_loads[best_station] += time_val
        task_remaining[task] = best_remaining - time_val  # slack after placement
        placed[task] = True
        
        # Update eligible successors
        for s in succs[task]:
            remaining_preds[s] -= 1
            if remaining_preds[s] == 0:
                heapq.heappush(heap, (-times[s], s))
    
    # task_remaining[task] = slack (remaining capacity after placing task)
    # We want to prioritize tasks that would have LOW slack (tight fit)
    # So negative slack = -task_remaining means tight fit gets higher priority
    
    # Normalize slack to [0, 1] where 0 = very tight, 1 = very loose
    max_slack = max(task_remaining) if max(task_remaining) > 0 else 1.0
    slack_norm = np.array([s / max_slack for s in task_remaining])
    
    # Negative slack: tight fit (low slack) -> high priority
    neg_slack = 1.0 - slack_norm
    
    # Normalize other components
    rpW_norm = (rpW - rpW.min()) / (rpW.max() - rpW.min() + 1e-10)
    time_norm = (times - times.min()) / (times.max() - times.min() + 1e-10)
    
    # Weighted combination:
    # Primary: negative slack (tight fit) - this is the bipartite matching insight
    # Secondary: processing time (LPT)
    # Tertiary: critical path importance
    priority_scores = 0.6 * neg_slack + 0.3 * time_norm + 0.1 * rpW_norm
    
    # Scale to meaningful range
    priority_scores = priority_scores * (times.max() + 1)
    
    return priority_scores
# EVOLVE-BLOCK-END
