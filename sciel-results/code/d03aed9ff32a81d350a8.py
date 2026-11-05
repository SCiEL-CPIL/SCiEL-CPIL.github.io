"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 148.0
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
    else:
        res_pressure = np.zeros(n)
        max_res_pressure = np.zeros(n)

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

    # 7. Dynamic Resource Contention: for each activity, compute max demand/capacity ratio
    #    among all activities that share any resource with it. This identifies tasks that
    #    compete for scarce resources.
    f_rc = np.zeros(n)
    if demand_matrix is not None and capacities_arr is not None:
        n_res = len(capacities_arr)
        # For each resource, find activities that use it
        for r in range(n_res):
            # Activities using resource r
            using_r = np.where(demand_matrix[:, r] > 0)[0]
            if len(using_r) > 1:
                # For each activity using r, contention = max(demand/cap) of other activities using r
                for i in using_r:
                    others = [j for j in using_r if j != i]
                    if others:
                        max_ratio = np.max(demand_matrix[others, r] / safe_caps[r])
                        if max_ratio > f_rc[i]:
                            f_rc[i] = max_ratio
        rc_max = np.max(f_rc)
        if rc_max > 0:
            f_rc = f_rc / rc_max
    
    # 8. Critical Path Length to End (longest path from this activity to any terminal)
    #    This is related to bottom_level but computed differently for robustness
    f_cp = np.zeros(n)
    # Compute longest path from each node to any terminal node
    # Terminal nodes have no successors
    for u in reversed(top_order):
        if not successors[u]:
            f_cp[u] = durations[u]
        else:
            max_succ = max(f_cp[v] + durations[v] for v in successors[u])
            f_cp[u] = durations[u] + max_succ
    cp_max = np.max(f_cp) if np.max(f_cp) > 0 else 1.0
    f_cp = f_cp / cp_max

    # 9. Duration-weighted resource opportunity cost:
    #    duration * max(demand/capacity) captures time-occupancy on scarce resources.
    #    Long activities that consume a large fraction of a scarce resource are costly.
    f_roc = np.zeros(n)
    if demand_matrix is not None and capacities_arr is not None:
        ratio_matrix = demand_matrix / safe_caps[np.newaxis, :]  # (n, n_res)
        max_ratio_per_task = np.max(ratio_matrix, axis=1)
        f_roc = durations * max_ratio_per_task
        roc_max = np.max(f_roc)
        if roc_max > 0:
            f_roc = f_roc / roc_max

    features = np.vstack([
        f_float, 
        f_bl, 
        f_rp, 
        f_sc, 
        f_dur, 
        f_wf,
        f_rc,
        f_cp,
        f_roc
    ]).T # Shape (n, 9)
    
    return features, successors, durations, res_pressure, safe_caps, demand_matrix if 'demand_matrix' in locals() else None, capacities_arr if 'capacities_arr' in locals() else None

def _simulate_serial_schedule(instance, priorities, successors, durations, demand_matrix, capacities_arr, safe_caps):
    n = len(durations)
    # Sort by priority (descending). Stable sort by index for tie-breaking consistency
    # Using np.argsort with kind='stable' isn't directly available in all versions without specifying, 
    # but standard sorts in numpy are stable since 1.15? No, quicksort is not. 
    # We need a stable sort to ensure deterministic behavior and reproducibility.
    # indices = np.argsort(-priorities, kind='stable')
    
    # Manual stable sort for safety
    order = sorted(range(n), key=lambda i: -priorities[i])
    
    start_times = np.full(n, -1)
    current_time = 0.0
    remaining = [True] * n
    finished = [False] * n
    
    # Track resource usage over time
    # Since n is small, we can just iterate through time or events.
    # Event-based simulation is more efficient.
    
    events = []
    # We will simulate step by step.
    # At each step, check if any task can start.
    # If not, jump to the next finish event.
    
    time = 0.0
    active_resources = np.zeros(len(capacities_arr))
    active_tasks = {} # task_id -> finish_time
    
    while any(remaining):
        # Try to start tasks
        started_any = False
        for i in order:
            if not remaining[i]:
                continue
            # Check precedence
            preds_done = True
            # We need pred lists, but we passed successors. Let's assume we can derive or just check if start_time of preds is set.
            # Actually, simpler: check if all predecessors have finished.
            # We don't have pred_lists passed. Let's refactor to pass pred_lists or compute on fly.
            # For speed, let's pre-compute pred_lists in main function and pass.
            pass # Placeholder, see updated function below
            
    # This simple loop is inefficient if we don't manage events well.
    # Better approach: 
    # 1. Identify eligible tasks (all preds finished).
    # 2. Among eligible, sort by priority.
    # 3. Try to start highest priority task if resources allow.
    # 4. If no tasks can start, jump time to min finish time of active tasks.
    
    # Let's rewrite simulation properly
    time = 0.0
    active = [] # list of (finish_time, task_id)
    res_usage = np.zeros(len(capacities_arr))
    
    # We need pred_lists for this simulation
    # I will modify _simulate to take pred_lists
    
    # Re-implementing simulation logic here for clarity in the final code block
    # This inner function is discarded, the real one is below.
    pass

def _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, priorities):
    n = len(durations)
    # Stable sort by priority
    order = sorted(range(n), key=lambda i: -priorities[i])
    
    time = 0.0
    remaining = [True] * n
    start_times = np.full(n, np.nan)
    finish_times = np.full(n, np.nan)
    active_tasks = [] # list of [finish_time, task_id]
    res_usage = np.zeros(len(capacities_arr))
    
    # To make it faster, we don't re-scan the whole 'order' list every time if we can avoid it.
    # But n=30 is small. Scanning 30 items is trivial.
    
    while any(remaining):
        # Find eligible tasks
        # Eligible if all preds are finished
        # To avoid O(N^2) per step, we could maintain a set of ready tasks.
        # Given N=30, O(N^2) is 900 ops. Even 10k steps is 9M ops, which is < 0.1s in Python.
        
        candidates = []
        for i in range(n):
            if not remaining[i]:
                continue
            # Check preds
            ready = True
            for p in pred_lists[i]:
                if np.isnan(finish_times[p]) or finish_times[p] > time:
                    ready = False
                    break
            if ready:
                candidates.append(i)
                
        if not candidates:
            # Jump to next event
            if not active_tasks:
                # Deadlock or error?
                break
            next_finish = min(t for t, _ in active_tasks)
            # Process all tasks finishing at next_finish
            for ft, tid in list(active_tasks):
                if ft <= next_finish + 1e-9:
                    active_tasks.remove([ft, tid])
                    res_usage -= demand_matrix[tid]
            time = next_finish
            continue
            
        # Sort candidates by priority (already in 'order' but we filtered)
        # 'candidates' is in index order. We need to sort by priority.
        candidates.sort(key=lambda x: -priorities[x])
        
        started_any = False
        for i in candidates:
            # Check resources
            if np.all(demand_matrix[i] <= res_usage + capacities_arr - 1e-9):
                # Start task i
                res_usage += demand_matrix[i]
                start_times[i] = time
                finish_times[i] = time + durations[i]
                remaining[i] = False
                active_tasks.append([finish_times[i], i])
                started_any = True
                # Do not break, try to start more tasks at same time if possible
                # (Parallel start is allowed if resources allow)
                
        if not started_any:
            # No candidate could start due to resources
            if not active_tasks:
                # Deadlock
                break
            next_finish = min(t for t, _ in active_tasks)
            for ft, tid in list(active_tasks):
                if ft <= next_finish + 1e-9:
                    active_tasks.remove([ft, tid])
                    res_usage -= demand_matrix[tid]
            time = next_finish
            continue
            
    # If any tasks are still remaining (shouldn't happen), assign max time
    max_time = np.nanmax(finish_times) if not np.all(np.isnan(finish_times)) else 0.0
    return max_time

def priority(instance):
    """
    Score each activity using a Model-and-Optimize approach.
    We define a feature space and optimize weights to minimize makespan.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Fallback
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
            
    # Get number of features
    n_features = features.shape[1]
    
    # Define objective function: Minimize Makespan
    def objective(weights):
        # Normalize weights to sum to 1 (simplex)
        w_sum = np.sum(weights)
        if w_sum < 1e-9:
            w_norm = np.ones(n_features) / n_features
        else:
            w_norm = weights / w_sum
            
        priorities = features @ w_norm
        makespan = _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, priorities)
        return makespan
    
    # Nelder-Mead style simplex search in the positive orthant, projected to simplex.
    # Features order: [f_float, f_bl, f_rp, f_sc, f_dur, f_wf, f_rc, f_cp, f_roc]
    # Initial weights emphasizing critical path, float, and resource opportunity cost
    w_current = np.array([0.20, 0.15, 0.05, 0.05, 0.05, 0.10, 0.10, 0.20, 0.10])
    w_current = np.clip(w_current, 0, None)
    w_sum_init = np.sum(w_current)
    if w_sum_init > 0:
        w_current = w_current / w_sum_init
    else:
        w_current = np.ones(n_features) / n_features
    
    best_makespan = objective(w_current)
    best_weights = w_current.copy()
    
    # Nelder-Mead simplex in weight space (positive orthant, projected to simplex)
    # Build initial simplex around w_current
    alpha = 0.1  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    # Initialize simplex: n_features+1 vertices
    simplex = [w_current.copy()]
    for i in range(n_features):
        v = w_current.copy()
        v[i] += 0.05
        v = np.clip(v, 0, None)
        s = np.sum(v)
        if s > 0:
            v = v / s
        simplex.append(v)
    
    # Evaluate all vertices
    simplex_f = [objective(simplex[i]) for i in range(n_features + 1)]
    
    # Track best
    for i in range(n_features + 1):
        if simplex_f[i] < best_makespan:
            best_makespan = simplex_f[i]
            best_weights = simplex[i].copy()
    
    evals_done = n_features + 1
    
    # Nelder-Mead iterations
    max_nm_iters = 500
    no_improve = 0
    max_no_improve = 50
    
    for _ in range(max_nm_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 3:
            break
        
        # Sort simplex by function value
        idx = np.argsort(simplex_f)
        simplex = [simplex[i] for i in idx]
        simplex_f = [simplex_f[i] for i in idx]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        # Project centroid to simplex (positive, sum=1)
        centroid = np.clip(centroid, 0, None)
        cs = np.sum(centroid)
        if cs < 1e-9:
            centroid = np.ones(n_features) / n_features
        else:
            centroid = centroid / cs
        
        worst = simplex[-1]
        worst_f = simplex_f[-1]
        best_f = simplex_f[0]
        
        # Reflection
        x_ref = centroid + alpha * (centroid - worst)
        x_ref = np.clip(x_ref, 0, None)
        rs = np.sum(x_ref)
        if rs < 1e-9:
            x_ref = centroid
        else:
            x_ref = x_ref / rs
        
        f_ref = objective(x_ref)
        evals_done += 1
        
        if f_ref < best_f:
            # Expansion
            x_exp = centroid + gamma * (x_ref - centroid)
            x_exp = np.clip(x_exp, 0, None)
            es = np.sum(x_exp)
            if es < 1e-9:
                x_exp = x_ref
            else:
                x_exp = x_exp / es
            
            f_exp = objective(x_exp)
            evals_done += 1
            
            if f_exp < f_ref:
                simplex[-1] = x_exp
                simplex_f[-1] = f_exp
            else:
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
        elif f_ref < worst_f:
            # Accept reflection
            simplex[-1] = x_ref
            simplex_f[-1] = f_ref
        else:
            # Contraction
            if f_ref < worst_f:
                # Outside contraction
                x_con = centroid + rho * (worst - centroid)
            else:
                # Inside contraction
                x_con = centroid + rho * (worst - centroid)
            
            x_con = np.clip(x_con, 0, None)
            cs2 = np.sum(x_con)
            if cs2 < 1e-9:
                x_con = centroid
            else:
                x_con = x_con / cs2
            
            f_con = objective(x_con)
            evals_done += 1
            
            if f_con < min(f_ref, worst_f):
                simplex[-1] = x_con
                simplex_f[-1] = f_con
            else:
                # Shrink
                for i in range(1, n_features + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], 0, None)
                    ss = np.sum(simplex[i])
                    if ss < 1e-9:
                        simplex[i] = simplex[0].copy()
                    else:
                        simplex[i] = simplex[i] / ss
                    simplex_f[i] = objective(simplex[i])
                    evals_done += 1
        
        # Update best
        for i in range(n_features + 1):
            if simplex_f[i] < best_makespan:
                best_makespan = simplex_f[i]
                best_weights = simplex[i].copy()
        
        # Check convergence
        if abs(worst_f - best_f) < 1e-6:
            no_improve += 1
        else:
            no_improve = 0
        
        if no_improve > max_no_improve:
            break
    
    # Stage 2: Fine coordinate descent polish
    # Use remaining time for high-precision polish
    polish_start = time.time()
    polish_budget = max(100, int((_TIME_LIMIT - 3 - (polish_start - _START_TIME)) / 0.005))
    polish_evals = 0
    polish_no_improve = 0
    max_polish_no_improve = 30
    
    while polish_evals < polish_budget and polish_no_improve < max_polish_no_improve:
        if time.time() - _START_TIME > _TIME_LIMIT - 2:
            break
        
        # Try each dimension with small steps
        improved = False
        for d in range(n_features):
            if time.time() - _START_TIME > _TIME_LIMIT - 2:
                break
            
            for direction in [-1, 1]:
                step = 0.02
                w_new = best_weights.copy()
                w_new[d] += direction * step
                w_new = np.clip(w_new, 0, None)
                ws = np.sum(w_new)
                if ws < 1e-9:
                    continue
                w_new = w_new / ws
                
                ms = objective(w_new)
                polish_evals += 1
                evals_done += 1
                
                if ms < best_makespan - 1e-7:
                    best_makespan = ms
                    best_weights = w_new.copy()
                    improved = True
                    break
        
        if not improved:
            polish_no_improve += 1
            # Try smaller steps
            for d in range(n_features):
                if time.time() - _START_TIME > _TIME_LIMIT - 2:
                    break
                for direction in [-1, 1]:
                    step = 0.005
                    w_new = best_weights.copy()
                    w_new[d] += direction * step
                    w_new = np.clip(w_new, 0, None)
                    ws = np.sum(w_new)
                    if ws < 1e-9:
                        continue
                    w_new = w_new / ws
                    
                    ms = objective(w_new)
                    polish_evals += 1
                    evals_done += 1
                    
                    if ms < best_makespan - 1e-8:
                        best_makespan = ms
                        best_weights = w_new.copy()
                        improved = True
                        break
                if improved:
                    break
        else:
            polish_no_improve = 0
    
    # Final check
    final_priorities = features @ best_weights
    return final_priorities

# EVOLVE-BLOCK-END
