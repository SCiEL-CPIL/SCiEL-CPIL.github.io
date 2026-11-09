"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import heapq

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _build_graph(instance):
    """Build adjacency, in-degree, and successor map."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    adj = defaultdict(list)       # task -> successors
    in_deg = np.zeros(n, dtype=int)
    
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                adj[i].append(j)
                in_deg[j] += 1
    
    return adj, in_deg


def _topo_sort(n, in_deg, adj):
    """Kahn's algorithm for topological sort."""
    in_deg_copy = in_deg.copy()
    queue = [i for i in range(n) if in_deg_copy[i] == 0]
    heapq.heapify(queue)
    topo = []
    while queue:
        u = heapq.heappop(queue)
        topo.append(u)
        for v in adj[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                heapq.heappush(queue, v)
    return topo


def _longest_path_forward(times, topo, adj):
    """Longest path from start to each node."""
    n = len(times)
    fwd = np.zeros(n, dtype=float)
    for u in topo:
        fwd[u] = times[u]
        for v in adj[u]:
            if fwd[u] + times[v] > fwd[v]:
                fwd[v] = fwd[u] + times[v]
    return fwd


def _longest_path_backward(times, topo, adj):
    """Longest path from each node to end."""
    n = len(times)
    bwd = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in adj[u]:
            if bwd[v] > best:
                best = bwd[v]
        bwd[u] = times[u] + best
    return bwd


def _count_successors(n, adj):
    """Count direct and indirect successors for each node."""
    succ_count = np.zeros(n, dtype=int)
    for i in range(n):
        visited = set()
        stack = list(adj[i])
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(adj[v])
        succ_count[i] = len(visited)
    return succ_count


def _downstream_time(times, adj):
    """Total time of node and all its successors."""
    n = len(times)
    downstream = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set([i])
        stack = list(adj[i])
        total = times[i]
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                total += times[v]
                stack.extend(adj[v])
        downstream[i] = total
    return downstream


def _count_predecessors(instance):
    """Count direct predecessors for each task."""
    preds = instance["predecessors"]
    n = len(instance["times"])
    pred_counts = np.zeros(n, dtype=int)
    for i in range(n):
        if i in preds:
            pred_counts[i] = len(preds[i])
    return pred_counts


def _normalize(arr):
    """Normalize array to [0, 1]."""
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _assign_stations(instance, priority_scores, cycle_time):
    """
    Greedy assignment: fill stations with highest-priority eligible task that fits.
    Returns number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    adj, in_deg = _build_graph(instance)
    rem_preds = in_deg.copy()
    
    assigned = np.zeros(n, dtype=bool)
    station_time = 0.0
    num_stations = 1
    
    remaining = n
    while remaining > 0:
        # Find eligible tasks
        eligible = [i for i in range(n) if not assigned[i] and rem_preds[i] == 0]
        
        if not eligible:
            break
        
        # Among eligible, find those that fit in current station
        remaining_capacity = cycle_time - station_time
        fit = [i for i in eligible if times[i] <= remaining_capacity + 1e-9]
        
        if fit:
            # Pick highest priority among those that fit
            best = max(fit, key=lambda i: priority_scores[i])
        else:
            # Close current station, start new one
            station_time = 0.0
            num_stations += 1
            remaining_capacity = cycle_time
            fit = [i for i in eligible if times[i] <= remaining_capacity + 1e-9]
            if not fit:
                best = max(eligible, key=lambda i: priority_scores[i])
            else:
                best = max(fit, key=lambda i: priority_scores[i])
        
        assigned[best] = True
        station_time += times[best]
        remaining -= 1
        
        # Update successor counts
        for s in adj[best]:
            rem_preds[s] -= 1
    
    return num_stations


def _generate_candidates(instance):
    """Generate multiple candidate priority vectors."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    adj, in_deg = _build_graph(instance)
    topo = _topo_sort(n, in_deg, adj)
    
    candidates = []
    
    # 1. Longest processing time
    candidates.append(np.asarray(times, dtype=float))
    
    # 2. Longest path to end (rank)
    bwd = _longest_path_backward(times, topo, adj)
    candidates.append(bwd)
    
    # 3. Critical path (forward + backward)
    fwd = _longest_path_forward(times, topo, adj)
    crit = fwd + bwd - times
    candidates.append(crit)
    
    # 4. Modified rank (successor count)
    succ_count = _count_successors(n, adj)
    candidates.append(succ_count.astype(float))
    
    # 5. Composite weighted
    p_time = _normalize(times)
    p_rank = _normalize(bwd)
    p_crit = _normalize(crit)
    p_succ = _normalize(succ_count.astype(float))
    composite = 0.35 * p_time + 0.3 * p_rank + 0.2 * p_crit + 0.15 * p_succ
    candidates.append(composite)
    
    # 6. Time * successor count
    candidates.append(times * (1.0 + succ_count.astype(float)))
    
    # 7. Backward path weighted by time
    candidates.append(bwd * (1.0 + 0.1 * times))
    
    # 8. Inverse predecessor count weighted
    pred_counts = _count_predecessors(instance)
    denom = 1.0 + pred_counts.astype(float)
    candidates.append(times / denom)
    
    # 9. Forward path
    candidates.append(fwd)
    
    # 10. Time * critical path
    candidates.append(times * crit)
    
    # 11. Downstream time (total work remaining including self)
    downstream = _downstream_time(times, adj)
    candidates.append(downstream)
    
    # 12. Downstream time normalized by cycle time (station pressure)
    ct = instance["cycle_time"]
    station_pressure = downstream / max(ct, 1e-9)
    candidates.append(station_pressure)
    
    # 13. Time squared (emphasize very long tasks)
    candidates.append(times ** 2)
    
    # 14. Rank position in topological order (earlier = higher priority)
    topo_pos = np.zeros(n, dtype=float)
    for rank, idx in enumerate(topo):
        topo_pos[idx] = float(n - rank)
    candidates.append(topo_pos)
    
    # 15. Critical Path Density: time / slack
    # Slack = cycle_time - (own_time + longest_downstream_path_excluding_self)
    # This prioritizes tasks with least scheduling flexibility
    longest_downstream_excl = bwd - times  # longest path from successors
    slack = ct - (times + longest_downstream_excl)
    slack = np.maximum(slack, 0.1)  # avoid division by zero
    cpd = times / slack
    candidates.append(cpd)
    
    # 16. CPD * time (emphasize both density and absolute time)
    candidates.append(cpd * times)
    
    # 17. CPD + rank
    candidates.append(cpd + bwd)
    
    # 18. Downstream * (1 + 0.5 * CPD_normalized)
    cpd_norm = _normalize(cpd)
    cand_18 = downstream * (1.0 + 0.5 * cpd_norm)
    candidates.append(cand_18)
    
    # 19. bwd * downstream
    candidates.append(bwd * downstream)
    
    # 20. Soft-max style
    tau = 0.3 * times.max() + 1e-9
    candidates.append(np.exp(times / tau))
    
    # 21. RPV (Remaining Processing Volume) = downstream time
    # Already covered by #11, but add variant
    candidates.append(downstream * (1.0 + 0.1 * succ_count.astype(float)))
    
    # 22. Bottom level (longest path to end) * successor count
    candidates.append(bwd * (1.0 + 0.5 * succ_count.astype(float)))
    
    # 23. Weighted: 0.4*CPD + 0.3*rank + 0.2*downstream + 0.1*time
    composite2 = 0.4 * _normalize(cpd) + 0.3 * p_rank + 0.2 * _normalize(downstream) + 0.1 * p_time
    candidates.append(composite2)
    
    return candidates


def _local_search(instance, start_priority, cycle_time, time_budget):
    """
    Local search with swap moves and occasional 2-opt style reordering.
    Uses a priority vector representation and evaluates via greedy assignment.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    best_priority = start_priority.copy()
    best_stations = _assign_stations(instance, best_priority, cycle_time)
    
    # Build order from priority
    order = np.argsort(-best_priority)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < time_budget and no_improve < max_no_improve:
        improved = False
        
        # Try adjacent swaps
        for i in range(n - 1):
            if time.time() > time_budget:
                break
            
            a, b = order[i], order[i + 1]
            mod_priority = best_priority.copy()
            mod_priority[a], mod_priority[b] = best_priority[b], best_priority[a]
            
            try:
                ns = _assign_stations(instance, mod_priority, cycle_time)
                if ns < best_stations:
                    best_stations = ns
                    best_priority = mod_priority
                    order = np.argsort(-best_priority)
                    improved = True
                    no_improve = 0
                    if time.time() > time_budget:
                        break
                else:
                    no_improve += 1
            except Exception:
                no_improve += 1
        
        if not improved:
            # Try random non-adjacent swaps
            rng = np.random.RandomState(hash(tuple(best_priority)) % 2**31)
            for _ in range(50):
                if time.time() > time_budget:
                    break
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    continue
                a, b = order[i], order[j]
                mod_priority = best_priority.copy()
                mod_priority[a], mod_priority[b] = best_priority[b], best_priority[a]
                try:
                    ns = _assign_stations(instance, mod_priority, cycle_time)
                    if ns < best_stations:
                        best_stations = ns
                        best_priority = mod_priority
                        order = np.argsort(-best_priority)
                        improved = True
                        no_improve = 0
                    else:
                        no_improve += 1
                except Exception:
                    no_improve += 1
        
        if not improved:
            no_improve += 10  # accelerate convergence
    
    return best_priority, best_stations


def priority(instance):
    """
    Score each assembly task using Critical Path Density heuristic
    with local search refinement.
    """
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Generate candidate priority vectors
    candidates = _generate_candidates(instance)
    
    # Evaluate each candidate and track the best
    best_priority = candidates[0]
    best_num_stations = _assign_stations(instance, candidates[0], cycle_time)
    
    for idx in range(1, len(candidates)):
        if time.time() > _DEADLINE - 10:
            break
        try:
            ns = _assign_stations(instance, candidates[idx], cycle_time)
            if ns < best_num_stations:
                best_num_stations = ns
                best_priority = candidates[idx]
        except Exception:
            continue
    
    # Local search refinement
    if time.time() < _DEADLINE - 15:
        time_budget = _DEADLINE - 15
        best_priority, best_num_stations = _local_search(
            instance, best_priority, cycle_time, time_budget
        )
    
    return best_priority
# EVOLVE-BLOCK-END
