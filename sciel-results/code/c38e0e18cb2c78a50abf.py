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
    
    return features, successors, durations, res_pressure, safe_caps, demand_matrix, capacities_arr, pred_lists

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
    res_usage = np.zeros(len(capacities_arr))
    
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
            res_usage -= demand_matrix[tid]
            # Update remaining preds for successors
            for s in successors[tid]:
                if remaining[s]:
                    remaining_preds[s] -= 1
        
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

def _compute_dynamic_bottleneck_features(pred_lists, successors, durations, demand_matrix, capacities_arr, pred_lists_out):
    """
    Simulate a resource-feasible greedy schedule to identify which activities 
    actually cause resource contention (bottlenecks), then compute dynamic 
    resource-weighted criticality scores.
    """
    n = len(durations)
    if n == 0:
        return np.zeros(n), np.zeros(n)
    
    # Initial greedy schedule using simple priority (duration descending)
    initial_priorities = -durations  # longer tasks first
    # Run simulation to get finish times
    pred_count = np.array([len(p) for p in pred_lists], dtype=int)
    remaining_preds = pred_count.copy()
    
    time = 0.0
    remaining = [True] * n
    finish_times = np.full(n, np.nan)
    res_usage = np.zeros(len(capacities_arr))
    
    import heapq
    active_heap = []
    
    while any(remaining):
        while active_heap and active_heap[0][0] <= time + 1e-9:
            ft, tid = heapq.heappop(active_heap)
            if ft > time + 1e-9:
                heapq.heappush(active_heap, (ft, tid))
                break
            res_usage -= demand_matrix[tid]
            for s in successors[tid]:
                if remaining[s]:
                    remaining_preds[s] -= 1
        
        candidates = []
        for i in range(n):
            if remaining[i] and remaining_preds[i] == 0:
                candidates.append(i)
        
        if not candidates:
            if not active_heap:
                break
            next_finish = active_heap[0][0]
            time = next_finish
            continue
        
        candidates.sort(key=lambda x: -initial_priorities[x])
        
        started_any = False
        for i in candidates:
            if np.all(demand_matrix[i] <= res_usage + capacities_arr - 1e-9):
                res_usage += demand_matrix[i]
                finish_times[i] = time + durations[i]
                remaining[i] = False
                heapq.heappush(active_heap, (finish_times[i], i))
                started_any = True
        
        if not started_any:
            if not active_heap:
                break
            next_finish = active_heap[0][0]
            time = next_finish
            continue
    
    # Now compute dynamic bottleneck features based on this schedule
    # Feature 1: Resource contention at start time - how much of each resource is used when this task starts
    # Feature 2: Criticality in the dynamic schedule (position on the "dynamic critical path")
    
    # For each task, compute the maximum resource utilization at its start time
    # We need to track resource usage over time
    # Simplified: For each task, compute how "tight" the resource situation is when it starts
    
    f_bottleneck = np.zeros(n)
    f_dynamic_crit = np.zeros(n)
    
    # Track resource usage events
    # For each task, find the maximum resource pressure at its start time
    start_times = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(finish_times[i]):
            start_times[i] = finish_times[i] - durations[i]
    
    # Compute resource usage over time intervals
    # For each task, determine the resource pressure at its start time
    # This is expensive, so we approximate: for each task, look at how many other tasks
    # are active at its start time and share resources
    
    for i in range(n):
        if np.isnan(start_times[i]):
            continue
        
        # Count how many other tasks are active at start_times[i] and share resources
        contention = 0.0
        for j in range(n):
            if j == i or np.isnan(start_times[j]) or np.isnan(finish_times[j]):
                continue
            if start_times[j] <= start_times[i] < finish_times[j]:
                # Task j is active when task i starts
                # Check if they share resources
                shared = np.sum((demand_matrix[i] > 0) & (demand_matrix[j] > 0))
                if shared > 0:
                    # Add pressure based on shared resources
                    for r in range(len(capacities_arr)):
                        if demand_matrix[i, r] > 0 and demand_matrix[j, r] > 0:
                            contention += (demand_matrix[i, r] + demand_matrix[j, r]) / capacities_arr[r]
        
        f_bottleneck[i] = contention
    
    # Normalize
    if np.max(f_bottleneck) > 0:
        f_bottleneck = f_bottleneck / np.max(f_bottleneck)
    
    # Dynamic criticality: compute the longest path from each task to the end in the dynamic schedule
    # Use the actual finish times to determine dependencies
    # A task is on the dynamic critical path if delaying it would delay the project end
    
    # Compute slack in the dynamic schedule
    # For each task, slack = (latest start) - (earliest start)
    # Latest start can be computed backward from the project end
    
    project_end = np.nanmax(finish_times) if not np.all(np.isnan(finish_times)) else 0.0
    
    # Backward pass to compute latest start times
    ls = np.full(n, project_end)
    for u in reversed(range(n)):
        # Find all successors
        succs = successors[u]
        if succs:
            min_ls_succ = np.inf
            for v in succs:
                if not np.isnan(ls[v]):
                    min_ls_succ = min(min_ls_succ, ls[v])
            if min_ls_succ < np.inf:
                ls[u] = min_ls_succ - durations[u]
        else:
            ls[u] = project_end - durations[u]
    
    # Slack = latest start - earliest start
    slack = ls - start_times
    slack = np.where(np.isnan(slack), 0, slack)
    slack = np.where(slack < 0, 0, slack)
    
    # Dynamic criticality = 1 - normalized slack (lower slack = higher criticality)
    if np.max(slack) > 0:
        f_dynamic_crit = 1.0 - (slack / np.max(slack))
    else:
        f_dynamic_crit = np.ones(n)
    
    return f_bottleneck, f_dynamic_crit

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
    features, successors, durations, res_pressure, safe_caps, demand_matrix, capacities_arr, pred_lists = _build_graph(instance)
            
    # Compute dynamic bottleneck features
    f_bottleneck, f_dynamic_crit = _compute_dynamic_bottleneck_features(
        pred_lists, successors, durations, demand_matrix, capacities_arr, pred_lists
    )
    
    # Add dynamic features to the feature matrix
    features = np.hstack([features, f_bottleneck.reshape(-1, 1), f_dynamic_crit.reshape(-1, 1)])
    
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
    # Features order: [f_float, f_bl, f_rp, f_sc, f_dur, f_wf, f_rc, f_cp, f_bottleneck, f_dynamic_crit]
    w_current = np.array([0.15, 0.15, 0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15])
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
    restarts = 6
    evals_per_restart = 300
    
    for r in range(restarts):
        if time.time() - _START_TIME > _TIME_LIMIT - 3:
            break
        
        # Start from a random point in the simplex, biased toward dynamic features
        base_alpha = np.ones(n_features) * 2.0
        # Increase weight for dynamic bottleneck features (last two)
        base_alpha[-2:] = 4.0
        w_restart = np.random.dirichlet(base_alpha)
        
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
