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
    succ_count = np.zeros(n, dtype=int)

    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
            succ_count[p] += 1

    return times, adj, in_deg, succ_count


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


def _compute_priorities(instance, alpha_cpd=2.0, alpha_rank=1.0, alpha_unlock=0.2, 
                         soft_temp=1.0, damp=0.85):
    """
    Compute priority scores with tunable parameters.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]

    preds = instance["predecessors"]
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    succ_count = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
            succ_count[p] += 1

    order = _topological_sort(adj, in_deg.copy(), n)

    # Downstream rank: longest path from task to any sink (including task's own time)
    rank_down = np.zeros(n, dtype=float)
    # Longest downstream path excluding the task's own time
    longest_downstream = np.zeros(n, dtype=float)
    
    for i in reversed(order):
        succs = adj[i]
        if succs:
            vals = rank_down[succs]
            m = float(vals.max())
            # soft-max with temperature
            soft = m + float(np.log(np.sum(np.exp((vals - m) / soft_temp)))) * soft_temp
            # damp toward hard max to keep critical-path emphasis
            soft = damp * m + (1 - damp) * soft
            longest_downstream[i] = soft
        else:
            soft = 0.0
            longest_downstream[i] = 0.0
        rank_down[i] = times[i] + soft

    # Critical Path Density: times[i] / slack[i]
    slack = cycle_time - times - longest_downstream
    eps = 1e-6
    slack_safe = np.maximum(slack, eps)
    cpd = times / slack_safe

    # Refined downstream-station estimate
    total_work_down = np.zeros(n, dtype=float)
    for i in reversed(order):
        s = 0.0
        for j in adj[i]:
            s += total_work_down[j]
        total_work_down[i] = times[i] + s

    path_lb = rank_down / cycle_time
    work_lb = total_work_down / cycle_time
    min_stations_est = np.maximum(1.0, np.maximum(path_lb, work_lb))

    # Bottleneck density
    bottleneck_density = times / min_stations_est

    # Unlocking power
    unlock = np.log1p(succ_count.astype(float))
    max_unlock = float(unlock.max()) if n else 1.0
    if max_unlock <= 0:
        max_unlock = 1.0
    unlock_norm = unlock / max_unlock

    # Normalize
    max_cpd = float(cpd.max()) if n > 0 else 1.0
    if max_cpd <= 0:
        max_cpd = 1.0
    cpd_norm = cpd / max_cpd
    
    max_rank = float(rank_down.max()) if n > 0 else 1.0
    if max_rank <= 0:
        max_rank = 1.0
    rank_norm = rank_down / max_rank

    priority_scores = (
        alpha_cpd * cpd_norm + 
        alpha_rank * rank_norm * (1.0 + bottleneck_density) + 
        alpha_unlock * unlock_norm * rank_norm
    )

    return priority_scores


def priority(instance):
    """
    Score each assembly task using Critical Path Density with local search polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Compute initial priorities
    best_scores = _compute_priorities(instance)
    best_stations = _simulate(instance, best_scores)
    
    # Local search: try small perturbations to improve
    # Strategy: swap priorities of pairs of tasks and see if it helps
    rng = np.random.RandomState(42)
    
    max_iterations = 50
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Randomly pick two tasks and swap their priority scores
        i, j = rng.choice(n, 2, replace=False)
        
        # Create perturbed scores
        perturbed = best_scores.copy()
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        
        # Evaluate
        stations = _simulate(instance, perturbed)
        
        if stations < best_stations:
            best_scores = perturbed
            best_stations = stations
        elif stations == best_stations:
            # Keep perturbation if same (explore)
            if rng.random() < 0.5:
                best_scores = perturbed
    
    # If time remains, try parameter tuning
    if time.time() < deadline - 10.0:
        param_configs = [
            (2.5, 1.2, 0.3, 0.8, 0.80),
            (1.8, 1.5, 0.1, 1.2, 0.90),
            (2.2, 0.8, 0.4, 1.0, 0.85),
            (3.0, 1.0, 0.1, 0.6, 0.75),
            (1.5, 2.0, 0.2, 1.5, 0.95),
        ]
        
        for alpha_cpd, alpha_rank, alpha_unlock, soft_temp, damp in param_configs:
            if time.time() > deadline:
                break
            
            scores = _compute_priorities(instance, alpha_cpd, alpha_rank, alpha_unlock, 
                                         soft_temp, damp)
            stations = _simulate(instance, scores)
            
            if stations < best_stations:
                best_scores = scores
                best_stations = stations
    
    return best_scores

# EVOLVE-BLOCK-END
