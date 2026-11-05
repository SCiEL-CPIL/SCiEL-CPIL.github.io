"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 145.0
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
        demand_matrix = None
        capacities_arr = None

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

    features = np.vstack([
        f_float, 
        f_bl, 
        f_rp, 
        f_sc, 
        f_dur, 
        f_wf,
        f_rc,
        f_cp
    ]).T # Shape (n, 8)
    
    return features, successors, durations, res_pressure, safe_caps, demand_matrix, capacities_arr

def _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, priorities):
    n = len(durations)
    if n == 0:
        return 0.0
    
    # Pre-compute pred_count for faster eligibility check
    pred_count = np.array([len(p) for p in pred_lists], dtype=int)
    remaining_preds = pred_count.copy()
    
    time = 0.0
    remaining = [True] * n
    finish_times = np.full(n, np.nan)
    active_tasks = [] # list of [finish_time, task_id]
    res_usage = np.zeros(len(capacities_arr))
    completed = [False] * n
    
    # Priority sort order (descending)
    order = sorted(range(n), key=lambda i: (-priorities[i], i))
    
    # Use a heap for active tasks to efficiently find min finish time
    import heapq
    active_heap = [] # (finish_time, task_id)
    
    while any(remaining):
        # Process all tasks that finish at current time
        while active_heap and active_heap[0][0] <= time + 1e-9:
            ft, tid = heapq.heappop(active_heap)
            if ft > time + 1e-9:
                heapq.heappush(active_heap, (ft, tid))
                break
            completed[tid] = True
            res_usage -= demand_matrix[tid]
            # Update remaining preds for successors
            for s in successors[tid]:
                if remaining[s]:
                    remaining_preds[s] -= 1
                    if remaining_preds[s] == 0:
                        # s becomes eligible
                        pass # We'll check in candidate scan
        
        # Find eligible candidates
        candidates = []
        for i in range(n):
            if remaining[i] and remaining_preds[i] == 0:
                candidates.append(i)
        
        if not candidates:
            # No eligible tasks. Jump to next finish event.
            if not active_heap:
                break
            next_finish = active_heap[0][0]
            time = next_finish
            continue
        
        # Sort candidates by priority
        candidates.sort(key=lambda x: -priorities[x])
        
        started_any = False
        for i in candidates:
            # Check resources
            if np.all(demand_matrix[i] <= res_usage + capacities_arr - 1e-9):
                # Start task i
                res_usage += demand_matrix[i]
                finish_times[i] = time + durations[i]
                remaining[i] = False
                heapq.heappush(active_heap, (finish_times[i], i))
                started_any = True
        
        if not started_any:
            # No candidate could start due to resources
            if not active_heap:
                break
            next_finish = active_heap[0][0]
            time = next_finish
            continue
    
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
        # Normalize weights to sum to 1
        w_sum = np.sum(weights)
        if w_sum < 1e-9:
            w_norm = np.ones(n_features) / n_features
        else:
            w_norm = weights / w_sum
            
        priorities = features @ w_norm
        makespan = _simulate_schedule_fast(pred_lists, successors, durations, demand_matrix, capacities_arr, priorities)
        return makespan
        
    # Local Search / Coordinate Descent
    # Initial weights emphasizing resource contention and critical path
    # Features order: [f_float, f_bl, f_rp, f_sc, f_dur, f_wf, f_rc, f_cp]
    w_current = np.array([0.25, 0.2, 0.1, 0.05, 0.05, 0.1, 0.15, 0.1])
    w_current = np.clip(w_current, 0, None)
    w_sum_init = np.sum(w_current)
    if w_sum_init > 0:
        w_current = w_current / w_sum_init
    else:
        w_current = np.ones(n_features) / n_features
    
    best_makespan = objective(w_current)
    best_weights = w_current.copy()
    
    np.random.seed(42)
    
    # Stage 1: Coordinate descent with adaptive step size
    max_cd_evals = 5000
    evals_done = 1
    no_improve_count = 0
    max_no_improve = 300
    step_size = 0.15
    
    while evals_done < max_cd_evals and no_improve_count < max_no_improve:
        # Check time
        if time.time() - _START_TIME > _TIME_LIMIT - 3:
            break
            
        # Coordinate Descent step: Perturb one weight
        d = np.random.randint(n_features)
        
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
            no_improve_count = 0
            # Increase step size slightly on improvement
            step_size = min(step_size * 1.05, 0.25)
        else:
            no_improve_count += 1
            # Decrease step size when stuck
            if no_improve_count % 50 == 0:
                step_size *= 0.9
    
    # Stage 2: Random restarts with local search
    restarts = 8
    evals_per_restart = 400
    
    for r in range(restarts):
        if time.time() - _START_TIME > _TIME_LIMIT - 3:
            break
        
        # Start from a random point in the simplex
        w_restart = np.random.dirichlet(np.ones(n_features) * 2.0)
        
        # Local search from this restart
        local_best = w_restart.copy()
        local_makespan = objective(local_best)
        evals_done += 1
        
        local_no_improve = 0
        local_step = 0.1
        
        for _ in range(evals_per_restart):
            if time.time() - _START_TIME > _TIME_LIMIT - 3:
                break
            
            d = np.random.randint(n_features)
            w_new = local_best.copy()
            w_new[d] += np.random.uniform(-local_step, local_step)
            w_new = np.clip(w_new, 0, None)
            
            w_sum = np.sum(w_new)
            if w_sum < 1e-9:
                w_new = local_best.copy()
            else:
                w_new = w_new / w_sum
                
            makespan = objective(w_new)
            evals_done += 1
            
            if makespan < local_makespan - 1e-6:
                local_makespan = makespan
                local_best = w_new.copy()
                local_no_improve = 0
                local_step = min(local_step * 1.05, 0.2)
            else:
                local_no_improve += 1
                if local_no_improve % 30 == 0:
                    local_step *= 0.9
        
        if local_makespan < best_makespan - 1e-6:
            best_makespan = local_makespan
            best_weights = local_best.copy()
    
    # Final check
    final_priorities = features @ best_weights
    return final_priorities

# EVOLVE-BLOCK-END
