"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import heapq
import time


def _build_graph(instance):
    """Build adjacency and in-degree arrays from the instance."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
    
    return times, adj, in_deg


def _topological_sort(adj, in_deg, n):
    """Return a topological ordering of tasks."""
    queue = deque()
    for i in range(n):
        if in_deg[i] == 0:
            queue.append(i)
    
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)
    
    return order


def _simulate(instance, priority_scores):
    """
    Simulate the greedy list scheduling with given priority scores.
    Returns number of stations.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
    
    completed = [False] * n
    station_load = 0.0
    num_stations = 1
    remaining_in_deg = in_deg.copy()
    
    while True:
        eligible = []
        for i in range(n):
            if not completed[i] and remaining_in_deg[i] == 0:
                eligible.append(i)
        
        if not eligible:
            break
        
        eligible.sort(key=lambda i: priority_scores[i], reverse=True)
        
        placed_any = False
        for i in eligible:
            if station_load + times[i] <= cycle_time:
                completed[i] = True
                station_load += times[i]
                placed_any = True
                for j in adj[i]:
                    remaining_in_deg[j] -= 1
        
        if not placed_any:
            num_stations += 1
            station_load = 0.0
            continue
        
        if all(completed):
            break
    
    return num_stations


def priority(instance):
    """
    Score each assembly task using a refined bottleneck density.
    
    priority[i] = rank_down[i] * (1 + times[i] / soft_min_stations[i])
    
    where rank_down is the longest path from task i to any sink (including
    task i's own time), and soft_min_stations is a log-sum-exp (soft-max)
    aggregate of the station pressures of all immediate successors plus the
    task's own time, normalized by cycle_time.
    
    The soft-max captures parallel downstream workload: a task with several
    heavy successors faces more station pressure than one with a single
    successor of the same critical-path length. This explicitly penalizes
    long tasks that sit on critical paths where station capacity is scarce,
    forcing the greedy filler to reserve tight cycle-time slots for
    high-impact tasks.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    if cycle_time <= 0:
        return times.copy()
    
    preds = instance["predecessors"]
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
    
    order = _topological_sort(adj, in_deg.copy(), n)
    
    # Downstream rank: longest path from task to any sink (including task's own time)
    rank_down = np.zeros(n, dtype=float)
    for i in reversed(order):
        max_succ = 0.0
        for j in adj[i]:
            if rank_down[j] > max_succ:
                max_succ = rank_down[j]
        rank_down[i] = times[i] + max_succ
    
    # Soft-min station pressure: use log-sum-exp (soft-max) over the
    # station units of each immediate successor plus the task's own time.
    # This is a smooth surrogate for the max, but also accumulates the
    # magnitude of parallel downstream work (unlike a hard max).
    #
    # pressure[i] = logsumexp( times[i]/cycle_time, rank_down[j]/cycle_time for j in adj[i] )
    # soft_min_stations[i] = max(1.0, pressure[i])
    #
    # For a single successor, this reduces to ~ max(task, succ) in station units.
    # For multiple successors, it grows with the number and size of successors,
    # reflecting that more parallel work requires more stations.
    
    # We compute pressure in station units (time / cycle_time).
    own_units = times / cycle_time
    
    pressure = np.zeros(n, dtype=float)
    for i in reversed(order):
        vals = [own_units[i]]
        for j in adj[i]:
            vals.append(rank_down[j] / cycle_time)
        # log-sum-exp for numerical stability
        m = max(vals)
        if m == 0.0:
            pressure[i] = 0.0
        else:
            s = sum(np.exp(v - m) for v in vals)
            pressure[i] = m + np.log(s)
    
    soft_min_stations = np.maximum(1.0, pressure)
    
    # Bottleneck density: ratio of task time to estimated stations for its subtree
    bottleneck_density = times / soft_min_stations
    
    # Primary priority is downstream rank (critical path length),
    # weighted by a factor that increases with bottleneck density.
    priority_scores = rank_down * (1.0 + bottleneck_density)
    
    return priority_scores

# EVOLVE-BLOCK-END
