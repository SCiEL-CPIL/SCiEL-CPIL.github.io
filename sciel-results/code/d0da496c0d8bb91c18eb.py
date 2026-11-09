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


def _downstream_station_requirement(times, adj, cycle_time):
    """
    For each node, compute the minimum number of stations its downstream
    subtree (including itself) would require if each station held at most
    cycle_time of work. This is an admissible lower bound on stations needed
    for that subtree.

    We approximate by: total_downstream_time / cycle_time, rounded up.
    total_downstream_time = sum of times of the node and all its successors.
    """
    n = len(times)
    # Compute total downstream time (including self) via reverse topological order
    # We need a topological order; use the adj to do a DFS post-order or use
    # the fact that successors come after in topo order.
    # Simpler: for each node, sum times of all nodes reachable (including self).
    # For n=30 this is O(n^2 * avg_succ) which is fine.
    downstream_time = np.zeros(n, dtype=float)
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
        downstream_time[i] = total

    # Minimum stations needed for downstream subtree
    if cycle_time <= 0:
        return np.ones(n, dtype=float)
    stations_req = np.ceil(downstream_time / cycle_time)
    # Ensure at least 1
    stations_req = np.maximum(stations_req, 1.0)
    return stations_req


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
                # Should not happen if times[i] <= cycle_time
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


def _assign_stations_with_order(instance, task_order, cycle_time):
    """
    Assign stations following a given task order (index into task_order = priority rank).
    task_order[i] is the task to consider at priority rank i (higher rank = higher priority).
    Returns number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    adj, in_deg = _build_graph(instance)
    rem_preds = in_deg.copy()
    
    assigned = np.zeros(n, dtype=bool)
    station_time = 0.0
    num_stations = 1
    
    remaining = n
    # Create a priority map from task -> rank
    rank_map = np.zeros(n, dtype=int)
    for rank, task in enumerate(task_order):
        rank_map[task] = rank
    
    while remaining > 0:
        # Find eligible tasks
        eligible = [i for i in range(n) if not assigned[i] and rem_preds[i] == 0]
        
        if not eligible:
            break
        
        # Among eligible, find those that fit in current station
        remaining_capacity = cycle_time - station_time
        fit = [i for i in eligible if times[i] <= remaining_capacity + 1e-9]
        
        if fit:
            # Pick lowest rank (highest priority) among those that fit
            best = min(fit, key=lambda i: rank_map[i])
        else:
            # Close current station, start new one
            station_time = 0.0
            num_stations += 1
            remaining_capacity = cycle_time
            fit = [i for i in eligible if times[i] <= remaining_capacity + 1e-9]
            if not fit:
                best = min(eligible, key=lambda i: rank_map[i])
            else:
                best = min(fit, key=lambda i: rank_map[i])
        
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
    
    # 11. Bottleneck density boost
    downstream_req = _downstream_station_requirement(times, adj, instance["cycle_time"])
    bottleneck_ratio = times / downstream_req
    ratio_norm = _normalize(bottleneck_ratio)
    boost_strength = 1.5
    cand_11 = times * (1.0 + boost_strength * ratio_norm)
    candidates.append(cand_11)
    
    # 12. Critical path * bottleneck ratio
    cand_12 = crit * (1.0 + 0.5 * ratio_norm)
    candidates.append(cand_12)
    
    # 13. Backward path with stronger bottleneck emphasis
    cand_13 = bwd * (1.0 + 1.0 * ratio_norm)
    candidates.append(cand_13)
    
    # 14. Downstream time (total work remaining including self)
    downstream_time = np.zeros(n, dtype=float)
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
        downstream_time[i] = total
    candidates.append(downstream_time)
    
    # 15. Downstream time normalized by cycle time (station pressure)
    ct = instance["cycle_time"]
    station_pressure = downstream_time / max(ct, 1e-9)
    candidates.append(station_pressure)
    
    # 16. Time squared (emphasize very long tasks)
    candidates.append(times ** 2)
    
    # 17. Rank position in topological order (earlier = higher priority)
    topo_pos = np.zeros(n, dtype=float)
    for rank, idx in enumerate(topo):
        topo_pos[idx] = float(n - rank)  # earlier tasks get higher score
    candidates.append(topo_pos)
    
    # 18. Combined: downstream_time * (1 + 0.5 * ratio_norm)
    cand_18 = downstream_time * (1.0 + 0.5 * ratio_norm)
    candidates.append(cand_18)
    
    # 19. bwd * downstream_time (both signals)
    cand_19 = bwd * downstream_time
    candidates.append(cand_19)
    
    # 20. Soft-max style: exp(times / tau) to smooth emphasis
    tau = 0.3 * times.max() + 1e-9
    cand_20 = np.exp(times / tau)
    candidates.append(cand_20)
    
    # 21. Positional weight: tasks that unlock many successors get higher priority
    # This rewards tasks whose completion enables the most downstream work
    pos_weight = succ_count.astype(float) * (1.0 + 0.5 * _normalize(times))
    candidates.append(pos_weight)
    
    # 22. Soft-min surrogate: smooth approximation to station count minimization
    # Use log-sum-exp style smoothing on the backward path
    bwd_smooth = bwd.copy()
    tau_sm = 0.1 * bwd.max() + 1e-9
    # Soft-max of (bwd / tau) to create a smooth ranking
    max_bwd = bwd.max()
    exp_bwd = np.exp((bwd - max_bwd) / tau_sm)
    soft_rank = tau_sm * np.log(exp_bwd.sum() + 1e-12) + bwd
    candidates.append(soft_rank)
    
    # 23. Downstream station pressure: total downstream work / cycle_time
    # Tasks with high downstream pressure should be scheduled early
    ds_pressure = downstream_time / max(ct, 1e-9)
    # Blend with time to break ties
    cand_23 = 0.7 * ds_pressure + 0.3 * _normalize(times)
    candidates.append(cand_23)
    
    # 24. Epigraph-style: minimize max(bwd, downstream_time/ct) via smooth max
    smooth_max_tau = 0.05 * max(bwd.max(), downstream_time.max() / max(ct, 1e-9)) + 1e-9
    logsumexp = smooth_max_tau * np.log(
        np.exp(bwd / smooth_max_tau) + np.exp(downstream_time / (smooth_max_tau * max(ct, 1e-9)))
    )
    candidates.append(logsumexp)
    
    # 25. Composite with positional weight and soft signals
    p_pos = _normalize(pos_weight)
    p_soft = _normalize(soft_rank)
    p_ds = _normalize(cand_23)
    composite2 = 0.25 * p_time + 0.25 * p_rank + 0.15 * p_pos + 0.15 * p_soft + 0.1 * p_ds + 0.1 * p_succ
    candidates.append(composite2)
    
    # 26. Time-weighted successor count (stronger emphasis on unlocking work)
    cand_26 = times * (1.0 + 2.0 * _normalize(succ_count.astype(float)))
    candidates.append(cand_26)
    
    # 27. Backward path with positional boost
    cand_27 = bwd * (1.0 + 0.5 * _normalize(succ_count.astype(float)))
    candidates.append(cand_27)
    
    # 28. Critical path with soft smoothing
    crit_smooth = crit + 0.1 * smooth_max_tau * np.log(
        np.exp(bwd / smooth_max_tau) + np.exp(fwd / smooth_max_tau)
    )
    candidates.append(crit_smooth)
    
    # 29. Downstream time with time-weighted boost
    cand_29 = downstream_time * (1.0 + 0.3 * _normalize(times))
    candidates.append(cand_29)
    
    # 30. Hybrid: geometric mean of time and backward path
    geo_mean = np.sqrt(times * (bwd + 1e-9))
    candidates.append(geo_mean)
    
    # 31. New: Smooth max of (bwd, downstream_time/ct, times) with varying tau
    for tau_factor in [0.02, 0.05, 0.1]:
        tau_val = tau_factor * max(bwd.max(), downstream_time.max() / max(ct, 1e-9), times.max()) + 1e-9
        # Use log-sum-exp for smooth max of three terms
        a = bwd / tau_val
        b = downstream_time / (tau_val * max(ct, 1e-9))
        c = times / tau_val
        max_val = np.maximum(np.maximum(a, b), c)
        logsumexp3 = tau_val * (max_val + np.log(
            np.exp(a - max_val) + np.exp(b - max_val) + np.exp(c - max_val)
        ) + 1e-12)
        candidates.append(logsumexp3)
    
    # 32. New: Weighted harmonic mean of time and backward path
    w_harmonic = 2.0 * times * (bwd + 1e-9) / (times + bwd + 1e-9)
    candidates.append(w_harmonic)
    
    # 33. New: Downstream time with critical path boost
    cand_33 = downstream_time * (1.0 + 0.3 * _normalize(crit))
    candidates.append(cand_33)
    
    # 34. New: Priority based on slack (cycle_time - time) inverted
    # Tasks with less slack (tighter fit) get higher priority
    slack = ct - times
    slack_norm = _normalize(slack)
    cand_34 = 1.0 - slack_norm  # invert: less slack = higher priority
    # Blend with time to break ties
    cand_34 = 0.7 * cand_34 + 0.3 * p_time
    candidates.append(cand_34)
    
    # 35. New: Composite with all signals, tuned weights
    composite3 = (0.2 * p_time + 0.2 * p_rank + 0.15 * p_crit + 
                  0.15 * p_succ + 0.1 * p_pos + 0.1 * _normalize(cand_34) + 0.1 * _normalize(geo_mean))
    candidates.append(composite3)
    
    return candidates


def priority(instance):
    """
    Score each assembly task. Stations are filled with the highest-priority 
    eligible task that fits the cycle time.
    
    Uses a model-and-optimize approach: generate parameterized priority vectors
    and select the one that minimizes station count.
    """
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Generate candidate priority vectors
    candidates = _generate_candidates(instance)
    
    # Evaluate each candidate and track the best
    best_priority = candidates[0].copy()
    best_num_stations = _assign_stations(instance, candidates[0], cycle_time)
    
    for idx in range(1, len(candidates)):
        if time.time() > _DEADLINE - 10:
            break
        try:
            ns = _assign_stations(instance, candidates[idx], cycle_time)
            if ns < best_num_stations:
                best_num_stations = ns
                best_priority = candidates[idx].copy()
        except Exception:
            continue
    
    # Local optimization using task order representation
    if time.time() < _DEADLINE - 15:
        rng = np.random.RandomState(42)
        # Create task order from current best priority
        order = np.argsort(-best_priority)
        best_order = order.copy()
        
        # Phase 1: Systematic pairwise swap search with wider neighborhood
        improved = True
        max_passes = 20
        for pass_num in range(max_passes):
            if time.time() > _DEADLINE - 15:
                break
            improved = False
            for i in range(n):
                if time.time() > _DEADLINE - 15:
                    break
                for j in range(i + 1, n):
                    if time.time() > _DEADLINE - 15:
                        break
                    # Try swapping tasks at positions i and j
                    mod_order = best_order.copy()
                    mod_order[i], mod_order[j] = mod_order[j], mod_order[i]
                    
                    # Build priority from order
                    mod_priority = np.zeros(n, dtype=float)
                    for rank, task in enumerate(mod_order):
                        mod_priority[task] = float(n - rank)
                    
                    try:
                        ns = _assign_stations(instance, mod_priority, cycle_time)
                        if ns < best_num_stations:
                            best_num_stations = ns
                            best_order = mod_order
                            # Update best_priority
                            best_priority = np.zeros(n, dtype=float)
                            for rank, task in enumerate(best_order):
                                best_priority[task] = float(n - rank)
                            improved = True
                            if time.time() > _DEADLINE - 15:
                                break
                    except Exception:
                        pass
                if time.time() > _DEADLINE - 15:
                    break
            if not improved:
                break
        
        # Phase 2: Insertion moves (move a task to a different position)
        if time.time() < _DEADLINE - 15:
            improved = True
            for pass_num in range(10):
                if time.time() > _DEADLINE - 15:
                    break
                improved = False
                for i in range(n):
                    if time.time() > _DEADLINE - 15:
                        break
                    for j in range(n):
                        if i == j or time.time() > _DEADLINE - 15:
                            continue
                        # Move task at position i to position j
                        mod_order = list(best_order)
                        task = mod_order.pop(i)
                        mod_order.insert(j, task)
                        
                        # Build priority from order
                        mod_priority = np.zeros(n, dtype=float)
                        for rank, task in enumerate(mod_order):
                            mod_priority[task] = float(n - rank)
                        
                        try:
                            ns = _assign_stations(instance, mod_priority, cycle_time)
                            if ns < best_num_stations:
                                best_num_stations = ns
                                best_order = np.array(mod_order)
                                # Update best_priority
                                best_priority = np.zeros(n, dtype=float)
                                for rank, task in enumerate(best_order):
                                    best_priority[task] = float(n - rank)
                                improved = True
                                if time.time() > _DEADLINE - 15:
                                    break
                        except Exception:
                            pass
                    if time.time() > _DEADLINE - 15:
                        break
                if not improved:
                    break
        
        # Phase 3: Random restarts with perturbation
        if time.time() < _DEADLINE - 10:
            for restart in range(30):
                if time.time() > _DEADLINE - 10:
                    break
                # Create a perturbed version of the best order
                mod_order = list(best_order)
                # Swap 2-3 random pairs
                for _ in range(2 + rng.randint(0, 2)):
                    i = rng.randint(0, n)
                    j = rng.randint(0, n)
                    if i != j:
                        mod_order[i], mod_order[j] = mod_order[j], mod_order[i]
                
                # Build priority from order
                mod_priority = np.zeros(n, dtype=float)
                for rank, task in enumerate(mod_order):
                    mod_priority[task] = float(n - rank)
                
                try:
                    ns = _assign_stations(instance, mod_priority, cycle_time)
                    if ns < best_num_stations:
                        best_num_stations = ns
                        best_order = np.array(mod_order)
                        best_priority = np.zeros(n, dtype=float)
                        for rank, task in enumerate(best_order):
                            best_priority[task] = float(n - rank)
                except Exception:
                    continue
        
        # Phase 4: Fine-grained noise on the final priority
        if time.time() < _DEADLINE - 5:
            rng2 = np.random.RandomState(123)
            for seed in range(40):
                if time.time() > _DEADLINE - 5:
                    break
                noise_scale = (0.01 + 0.01 * (seed % 3) / 2.0) * (best_priority.max() - best_priority.min() + 1e-9)
                noisy = best_priority + rng2.normal(0, noise_scale, size=n)
                try:
                    ns = _assign_stations(instance, noisy, cycle_time)
                    if ns < best_num_stations:
                        best_num_stations = ns
                        best_priority = noisy
                except Exception:
                    continue
    
    return best_priority
# EVOLVE-BLOCK-END
