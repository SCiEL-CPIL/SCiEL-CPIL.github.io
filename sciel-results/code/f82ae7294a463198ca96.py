"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 143.0
_START_TIME = None

def _build_graph(instance):
    n = len(instance["durations"])
    pred_lists = []
    for p in instance.get("predecessors", [None] * n):
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                successors[p].append(i)
                in_degree[i] += 1
                
    # Topological Sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) < n:
        top_order = list(range(n))
        
    # CPM Forward/Backward
    durations = np.asarray(instance["durations"], dtype=float)
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        for v in successors[u]:
            if es[v] < ef[u]:
                es[v] = ef[u]
        ef[u] = es[u] + durations[u]
        
    project_end = np.max(ef) if n > 0 else 0.0
    lf = np.full(n, project_end)
    ls = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            if lf[u] > ls[v]:
                lf[u] = ls[v]
        ls[u] = lf[u] - durations[u]
        
    total_float = ls - es
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            candidate = bottom_level[v] + durations[v]
            if candidate > bottom_level[u]:
                bottom_level[u] = candidate
                
    # Resource Metrics
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is not None and capacities is not None:
        demand_matrix = np.zeros((n, len(capacities)), dtype=float)
        if isinstance(demands, list):
            if len(demands) > 0:
                if isinstance(demands[0], dict):
                    for i, d in enumerate(demands):
                        for r, val in d.items():
                            if r < len(capacities):
                                demand_matrix[i, r] = val
                else:
                    dm = np.asarray(demands, dtype=float)
                    if dm.shape[0] == n:
                        if dm.ndim == 1:
                            demand_matrix[:, 0] = dm
                        else:
                            cols = min(dm.shape[1], len(capacities))
                            demand_matrix[:, :cols] = dm[:, :cols]
        capacities_arr = np.asarray(capacities, dtype=float)
        safe_caps = np.where(capacities_arr > 0, capacities_arr, 1.0)
        res_pressure = np.sum(demand_matrix / safe_caps, axis=1)
        max_res_pressure = np.max(demand_matrix / safe_caps, axis=1)
        
        # Dynamic Resource Contention Feature (Inspiration 1 Strategy)
        # For each activity i, look at all unscheduled activities j (initially all)
        # that share any resource. Calculate the max utilization ratio demand[j]/capacity[r]
        # for resources shared between i and j. This proxies how much "pressure" i faces
        # or how much it contributes to the bottleneck pool.
        # We compute a "contention score" for each node:
        # contention[i] = max over j != i of (max over r where demand[i,r]>0 and demand[j,r]>0 of demand[j,r]/cap[r])
        # This is O(N^2 * R). N=30, R=4. 900*4 = 3600 ops. Very fast.
        
        f_contention = np.zeros(n)
        for i in range(n):
            max_cont = 0.0
            # Find resources used by i
            used_res_i = np.where(demand_matrix[i] > 0)[0]
            if len(used_res_i) == 0:
                continue
            for j in range(n):
                if i == j:
                    continue
                # Check if j shares any resource with i
                # Efficient check: intersection of non-zero indices
                # Since R is small, just loop through used_res_i
                for r in used_res_i:
                    if demand_matrix[j, r] > 0:
                        ratio = demand_matrix[j, r] / safe_caps[r]
                        if ratio > max_cont:
                            max_cont = ratio
            f_contention[i] = max_cont
            
        c_max = np.max(f_contention)
        if c_max > 0:
            f_contention = f_contention / c_max
        else:
            f_contention = np.zeros(n)
    else:
        res_pressure = np.zeros(n)
        max_res_pressure = np.zeros(n)
        f_contention = np.zeros(n)

    # Features
    # 1. Neg Float (higher is better)
    float_max = np.max(total_float)
    if float_max > 0:
        f_float = 1.0 - (total_float / float_max)
    else:
        f_float = np.ones(n)
        
    # 2. Bottom Level (higher is better)
    bl_max = np.max(bottom_level)
    if bl_max > 0:
        f_bl = bottom_level / bl_max
    else:
        f_bl = np.ones(n)
        
    # 3. Resource Pressure (higher is better)
    rp_max = np.max(res_pressure)
    if rp_max > 0:
        f_rp = res_pressure / rp_max
    else:
        f_rp = np.zeros(n)
        
    # 4. Successor Count (more successors often means higher priority to avoid bottlenecks)
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    sc_max = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    f_sc = succ_count / sc_max

    # 5. Duration (longer activities usually higher priority)
    d_max = np.max(durations) if np.max(durations) > 0 else 1.0
    f_dur = durations / d_max
    
    # 6. Weighted Succ Float (sum of floats of successors, normalized)
    ws_float = np.zeros(n)
    for u in range(n):
        for v in successors[u]:
            ws_float[u] += total_float[v]
    wf_max = np.max(ws_float)
    if wf_max > 0:
        f_wf = 1.0 - (ws_float / wf_max) # Invert so lower succ float = higher priority
    else:
        f_wf = np.ones(n)
        
    # 7. Dynamic Contention
    # Already computed above if resources exist

    features = np.vstack([
        f_float, 
        f_bl, 
        f_rp, 
        f_sc, 
        f_dur, 
        f_wf,
        f_contention
    ]).T # Shape (n, 7)
    
    return features, successors, durations, res_pressure, safe_caps, demand_matrix if 'demand_matrix' in locals() else None, capacities_arr if 'capacities_arr' in locals() else None

def _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, priorities, return_delays=False):
    n = len(durations)
    # Stable sort by priority
    order = sorted(range(n), key=lambda i: -priorities[i])
    
    t_now = 0.0
    remaining = [True] * n
    start_times = np.full(n, np.nan)
    finish_times = np.full(n, np.nan)
    active_tasks = [] # list of [finish_time, task_id]
    res_usage = np.zeros(len(capacities_arr))
    delays = np.zeros(n)  # delay = actual_start - earliest_start (from CPM)
    
    # Pre-compute ES from CPM for delay calculation
    in_deg = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                in_deg[i] += 1
    es = np.zeros(n)
    for u in order:
        for v in successors[u]:
            if es[v] < es[u] + durations[u]:
                es[v] = es[u] + durations[u]
    
    while any(remaining):
        # Find eligible tasks
        candidates = []
        for i in range(n):
            if not remaining[i]:
                continue
            # Check preds
            ready = True
            for p in pred_lists[i]:
                if np.isnan(finish_times[p]) or finish_times[p] > t_now:
                    ready = False
                    break
            if ready:
                candidates.append(i)
                
        if not candidates:
            # Jump to next event
            if not active_tasks:
                break
            next_finish = min(t for t, _ in active_tasks)
            for ft, tid in list(active_tasks):
                if ft <= next_finish + 1e-9:
                    active_tasks.remove([ft, tid])
                    res_usage -= demand_matrix[tid]
            t_now = next_finish
            continue
            
        # Sort candidates by priority
        candidates.sort(key=lambda x: -priorities[x])
        
        started_any = False
        for i in candidates:
            # Check resources
            if np.all(demand_matrix[i] <= res_usage + capacities_arr - 1e-9):
                res_usage += demand_matrix[i]
                start_times[i] = t_now
                finish_times[i] = t_now + durations[i]
                delays[i] = max(0.0, t_now - es[i])
                remaining[i] = False
                active_tasks.append([finish_times[i], i])
                started_any = True
                
        if not started_any:
            if not active_tasks:
                break
            next_finish = min(t for t, _ in active_tasks)
            for ft, tid in list(active_tasks):
                if ft <= next_finish + 1e-9:
                    active_tasks.remove([ft, tid])
                    res_usage -= demand_matrix[tid]
            t_now = next_finish
            continue
            
    max_time = np.nanmax(finish_times) if not np.all(np.isnan(finish_times)) else 0.0
    if return_delays:
        return max_time, delays
    return max_time

def priority(instance):
    """
    Score each activity using a Model-and-Optimize approach with dynamic contention features
    and simulation-based delay features. Uses smooth surrogate for objective.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        durations = np.asarray(instance["durations"], dtype=float)
        return durations

    n = len(instance["durations"])
    if n == 0:
        return np.array([])
        
    # Build features and data structures
    features, successors, durations, res_pressure, safe_caps, demand_matrix, capacities_arr = _build_graph(instance)
    
    # Reconstruct pred_lists for simulation
    pred_lists = []
    for p in instance.get("predecessors", [None] * n):
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    # --- Simulation-based delay feature ---
    # Run a baseline schedule (e.g., CPM-based priority) to get delays
    # These delays capture resource conflicts that static CPM misses.
    baseline_priorities = features @ np.array([0.4, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05])
    _, baseline_delays = _simulate_schedule_fast(
        pred_lists, successors, durations, demand_matrix, capacities_arr, baseline_priorities, return_delays=True
    )
    
    # Normalize delays to [0,1]
    d_max = np.max(baseline_delays)
    if d_max > 1e-9:
        f_delay = baseline_delays / d_max
    else:
        f_delay = np.zeros(n)
    
    # Also compute a "resource contention during schedule" feature:
    # For each activity, how many other activities were waiting when it started?
    # Approximate: delay * resource_pressure (high delay + high resource use = critical)
    f_delay_res = f_delay * res_pressure
    dr_max = np.max(f_delay_res)
    if dr_max > 1e-9:
        f_delay_res = f_delay_res / dr_max
    else:
        f_delay_res = np.zeros(n)
    
    # Add new features
    features = np.hstack([features, f_delay[:, None], f_delay_res[:, None]])
    n_features = features.shape[1]
    
    # --- Smooth surrogate objective ---
    # Instead of raw makespan, use a soft-min over a set of "proxy makespans"
    # derived from different priority perturbations. This smooths the landscape.
    # Actually, better: use the makespan itself but with a small penalty for
    # variance in delays (encourages balanced scheduling).
    
    def objective(weights):
        w_sum = np.sum(weights)
        if w_sum < 1e-9:
            w_norm = np.ones(n_features) / n_features
        else:
            w_norm = weights / w_sum
            
        priorities = features @ w_norm
        makespan, delays = _simulate_schedule_fast(
            pred_lists, successors, durations, demand_matrix, capacities_arr, priorities, return_delays=True
        )
        # Smooth penalty: log-sum-exp of delays (soft-max proxy)
        # This penalizes schedules with a few very large delays
        if np.max(delays) > 1e-9:
            # Temperature-scaled soft-max of delays
            temp = 2.0  # lower = sharper, higher = smoother
            shifted = delays / temp
            shifted = shifted - np.max(shifted)  # numerical stability
            soft_max = temp * np.log(np.sum(np.exp(shifted)))
            penalty = 0.05 * soft_max
        else:
            penalty = 0.0
        return makespan + penalty
    
    # --- Coordinate Descent with adaptive steps ---
    # Initial weights: emphasize float, bottom level, contention, and delay features
    # Feature order: [f_float, f_bl, f_rp, f_sc, f_dur, f_wf, f_contention, f_delay, f_delay_res]
    w_current = np.array([0.25, 0.20, 0.08, 0.04, 0.04, 0.04, 0.15, 0.12, 0.08])
    w_current = np.clip(w_current, 0, None)
    w_current = w_current / np.sum(w_current)
    
    best_makespan = objective(w_current)
    best_weights = w_current.copy()
    
    np.random.seed(42)
    
    # Phase 1: Broad search
    max_evals_p1 = 6000
    evals_done = 1
    
    while evals_done < max_evals_p1:
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
            
        d = np.random.randint(n_features)
        progress = evals_done / max_evals_p1
        step_size = 0.12 * (1.0 - 0.6 * progress)
        
        w_new = best_weights.copy()
        w_new[d] += np.random.uniform(-step_size, step_size)
        w_new = np.clip(w_new, 0, None)
        
        w_sum = np.sum(w_new)
        if w_sum < 1e-9:
            w_new = best_weights.copy()
        else:
            w_new = w_new / w_sum
            
        makespan = objective(w_new)
        evals_done += 1
        
        if makespan < best_makespan - 1e-6:
            best_makespan = makespan
            best_weights = w_new.copy()
    
    # Phase 2: Medium polish
    if time.time() - _START_TIME < _TIME_LIMIT - 5:
        max_evals_p2 = 3000
        evals_p2 = 0
        while evals_p2 < max_evals_p2:
            if time.time() - _START_TIME > _TIME_LIMIT - 4:
                break
                
            n_perturb = np.random.choice([1, 2], p=[0.6, 0.4])
            dims = np.random.choice(n_features, size=n_perturb, replace=False)
            
            w_new = best_weights.copy()
            for d in dims:
                w_new[d] += np.random.uniform(-0.04, 0.04)
            w_new = np.clip(w_new, 0, None)
            
            w_sum = np.sum(w_new)
            if w_sum < 1e-9:
                w_new = best_weights.copy()
            else:
                w_new = w_new / w_sum
                
            makespan = objective(w_new)
            evals_p2 += 1
            
            if makespan < best_makespan - 1e-7:
                best_makespan = makespan
                best_weights = w_new.copy()
    
    # Phase 3: High-precision polish (tighter tolerance, more iterations)
    if time.time() - _START_TIME < _TIME_LIMIT - 2:
        max_evals_p3 = 2000
        evals_p3 = 0
        while evals_p3 < max_evals_p3:
            if time.time() - _START_TIME > _TIME_LIMIT - 1:
                break
                
            # Very fine perturbations on 1-2 dims
            n_perturb = np.random.choice([1, 2], p=[0.5, 0.5])
            dims = np.random.choice(n_features, size=n_perturb, replace=False)
            
            w_new = best_weights.copy()
            for d in dims:
                w_new[d] += np.random.uniform(-0.015, 0.015)
            w_new = np.clip(w_new, 0, None)
            
            w_sum = np.sum(w_new)
            if w_sum < 1e-9:
                w_new = best_weights.copy()
            else:
                w_new = w_new / w_sum
                
            makespan = objective(w_new)
            evals_p3 += 1
            
            if makespan < best_makespan - 1e-8:
                best_makespan = makespan
                best_weights = w_new.copy()
    
    final_priorities = features @ best_weights
    return final_priorities

# EVOLVE-BLOCK-END
