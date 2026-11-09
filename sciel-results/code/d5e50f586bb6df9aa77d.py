# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def _simulate_station_count(times, predecessors, cycle_time, priorities):
    """Simulate greedy assignment and return number of stations."""
    n = len(times)
    successors = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i]
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    remaining_preds = pred_count[:]
    available = set(i for i in range(n) if remaining_preds[i] == 0)
    
    station_count = 0
    remaining_time = 0
    assigned = set()
    
    while len(assigned) < n:
        if remaining_time == 0:
            station_count += 1
            remaining_time = cycle_time
        
        best_task = -1
        best_priority = -np.inf
        for t in available:
            if t in assigned:
                continue
            if times[t] <= remaining_time:
                if priorities[t] > best_priority:
                    best_priority = priorities[t]
                    best_task = t
        
        if best_task == -1:
            station_count += 1
            remaining_time = cycle_time
            best_task = -1
            best_priority = -np.inf
            for t in available:
                if t in assigned:
                    continue
                if times[t] <= remaining_time:
                    if priorities[t] > best_priority:
                        best_priority = priorities[t]
                        best_task = t
            if best_task == -1:
                min_t = min(available - assigned, key=lambda x: times[x])
                best_task = min_t
        
        assigned.add(best_task)
        remaining_time -= times[best_task]
        available.discard(best_task)
        
        for s in successors[best_task]:
            remaining_preds[s] -= 1
            if remaining_preds[s] == 0:
                available.add(s)
    
    return station_count


def _compute_ranks(times, predecessors):
    """Compute rank for each task: longest path from task to end."""
    n = len(times)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    in_deg = [len(predecessors[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for s in successors[u]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    
    longest = [0.0] * n
    for u in reversed(topo_order):
        if successors[u]:
            longest[u] = times[u] + max(longest[s] for s in successors[u])
        else:
            longest[u] = times[u]
    
    return longest


def _compute_downstream_work(times, predecessors):
    """Compute total downstream work (sum of times of all successors, recursively)."""
    n = len(times)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    in_deg = [len(predecessors[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for s in successors[u]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    
    downstream = [0.0] * n
    for u in reversed(topo_order):
        downstream[u] = times[u]
        for s in successors[u]:
            downstream[u] += downstream[s]
    
    return downstream


def _compute_downstream_criticality(times, predecessors):
    """Compute criticality based on downstream work and number of parallel paths."""
    n = len(times)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    in_deg = [len(predecessors[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for s in successors[u]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    
    # Longest path from node to end
    longest = [0.0] * n
    for u in reversed(topo_order):
        if successors[u]:
            longest[u] = times[u] + max(longest[s] for s in successors[u])
        else:
            longest[u] = times[u]
    
    # Total downstream work
    downstream = [0.0] * n
    for u in reversed(topo_order):
        downstream[u] = times[u]
        for s in successors[u]:
            downstream[u] += downstream[s]
    
    # Number of successors (out-degree)
    out_deg = [len(successors[i]) for i in range(n)]
    
    # Criticality: prioritize tasks with high longest path AND high downstream work
    # Also penalize tasks that have many successors (they are less critical)
    criticality = np.zeros(n)
    for i in range(n):
        # Normalize longest path and downstream
        criticality[i] = longest[i] + 0.5 * downstream[i] - 0.1 * out_deg[i]
    
    return criticality


def priority(instance):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Compute various priority features
    ranks = _compute_ranks(times, predecessors)
    downstream_work = _compute_downstream_work(times, predecessors)
    criticality = _compute_downstream_criticality(times, predecessors)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Generate candidate priority schemes
    candidates = []
    
    # Scheme 1: rank only
    candidates.append(ranks.copy())
    
    # Scheme 2: time only (LPT)
    candidates.append(times.copy())
    
    # Scheme 3: rank + alpha * time
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        candidates.append(ranks + alpha * times)
    
    # Scheme 4: rank + beta * downstream_work
    for beta in [0.5, 1.0, 1.5]:
        candidates.append(ranks + beta * downstream_work)
    
    # Scheme 5: criticality
    candidates.append(criticality.copy())
    
    # Scheme 6: rank * (1 + out_degree)
    candidates.append(ranks * (1.0 + out_degree))
    
    # Scheme 7: rank + out_degree
    candidates.append(ranks + out_degree)
    
    # Scheme 8: criticality + rank
    candidates.append(criticality + ranks)
    
    # Evaluate all candidates
    best_score = float('inf')
    best_priorities = None
    
    for cand in candidates:
        score = _simulate_station_count(times, predecessors, cycle_time, cand)
        if score < best_score:
            best_score = score
            best_priorities = cand.copy()
    
    # Local search: perturb best_priorities
    rng = np.random.default_rng(42)
    
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            i = rng.integers(0, n)
            perturbation = rng.normal(0, 1.0)
            new_priorities = best_priorities.copy()
            new_priorities[i] += perturbation
            
            score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
            if score < best_score:
                best_score = score
                best_priorities = new_priorities
                improved = True
    
    # Try pair swaps
    if time.time() < deadline - 5:
        improved = True
        while improved and (time.time() < deadline - 5):
            improved = False
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                i, j = rng.integers(0, n, size=2)
                if i == j:
                    continue
                new_priorities = best_priorities.copy()
                new_priorities[i], new_priorities[j] = new_priorities[j], new_priorities[i]
                score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
                if score < best_score:
                    best_score = score
                    best_priorities = new_priorities
                    improved = True
    
    return best_priorities
# EVOLVE-BLOCK-END
