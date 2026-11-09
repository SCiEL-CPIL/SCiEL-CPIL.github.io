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
    Score each assembly task using bottleneck density:
    priority[i] = rank_down[i] * (1 + times[i] / min_stations_downstream[i])
    
    where rank_down is the longest path from task to any sink,
    and min_stations_downstream is the minimum number of stations needed
    to complete the downstream subtree of task i.
    
    This penalizes long tasks that sit on critical paths where station
    capacity is scarce.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    preds = instance["predecessors"]
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
    
    order = _topological_sort(adj, in_deg.copy(), n)
    
    # Downstream rank: longest path from task to any sink (including task's own time)
    # Use a smooth soft-max (log-sum-exp style) over successors with a small
    # temperature so the priority is differentiable-ish and captures the
    # "second-best" critical path as well, not just the single longest.
    # This is a smooth surrogate for the hard max in the rank recurrence.
    rank_down = np.zeros(n, dtype=float)
    # Temperature for soft-max: smaller = closer to hard max.
    # Use a temperature relative to the mean task time so it scales with instance.
    mean_t = float(np.mean(times)) if n > 0 else 1.0
    if mean_t <= 0:
        mean_t = 1.0
    tau = 0.25 * mean_t  # softness parameter
    
    for i in reversed(order):
        succs = adj[i]
        if not succs:
            rank_down[i] = times[i]
        else:
            vals = np.array([rank_down[j] for j in succs], dtype=float)
            # Soft-max: tau * log(sum(exp(v/tau)))
            # Numerically stable: subtract max first.
            v_max = float(np.max(vals))
            if tau > 1e-12:
                soft = tau * float(np.log(np.sum(np.exp((vals - v_max) / tau)))) + v_max
            else:
                soft = v_max
            rank_down[i] = times[i] + soft
    
    # Count of immediate successors (branching pressure).
    succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        succ_count[i] = float(len(adj[i]))
    
    # Minimum stations estimate for the downstream subtree of task i.
    # Lower bound: ceil(rank_down[i] / cycle_time).
    min_stations_est = np.maximum(1.0, np.ceil(rank_down / cycle_time))
    
    # Bottleneck density: ratio of task time to estimated stations for its subtree.
    # Higher density => task is a bigger bottleneck relative to available capacity.
    bottleneck_density = times / min_stations_est
    
    # Branching pressure: tasks with many successors create more downstream
    # "open" tasks that will compete for stations; placing them early frees
    # capacity sooner for the fan-out.
    # Normalize succ_count to [0,1] via a smooth saturation.
    max_succ = float(np.max(succ_count)) if n > 0 else 0.0
    if max_succ > 0:
        branch_pressure = succ_count / max_succ
    else:
        branch_pressure = np.zeros(n, dtype=float)
    
    # Combine within the same family:
    #   priority = rank_down * (1 + bottleneck_density) * (1 + alpha * branch_pressure)
    # The multiplicative form keeps the critical-path ordering dominant while
    # letting branching and density modulate it smoothly.
    alpha = 0.5  # weight for branching pressure
    priority_scores = rank_down * (1.0 + bottleneck_density) * (1.0 + alpha * branch_pressure)
    
    return priority_scores

# EVOLVE-BLOCK-END
