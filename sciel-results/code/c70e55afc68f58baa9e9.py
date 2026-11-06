"""RCPSP priority via fast NumPy SGS with Nelder-Mead optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque
from heapq import heappush, heappop


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy:
    1. Compute rich features (slack, duration, resource criticality, successors, predecessors).
    2. Use fast NumPy-vectorized SGS simulation.
    3. Optimize linear feature weights via Nelder-Mead simplex method.
    """
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill

    # --- Extract instance components ---
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred"]:
        if key in instance:
            preds = instance[key]
            break
    if preds is None:
        preds = [[] for _ in range(n)]

    succs = None
    for key in ["successors", "succs", "successors_list", "succ"]:
        if key in instance:
            succs = instance[key]
            break

    demands = None
    for key in ["demands", "resource_demands", "resources", "dem", "resource"]:
        if key in instance:
            demands = instance[key]
            break
    if demands is None:
        demands = np.zeros((n, 1))

    capacities = None
    for key in ["capacities", "capacity", "resource_capacities", "caps", "cap"]:
        if key in instance:
            capacities = instance[key]
            break
    if capacities is None:
        capacities = [1.0]

    demands = np.asarray(demands, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
    R = demands.shape[1]
    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim == 0:
        capacities = np.full(R, float(capacities))

    # Build successor lists if not provided
    if succs is None:
        succs = [set() for _ in range(n)]
        for j in range(n):
            for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
                idx = int(i)
                if 0 <= idx < n:
                    succs[idx].add(j)
        succs = [list(s) for s in succs]

    # Build pred sets for fast lookup
    pred_set = [set() for _ in range(n)]
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            for i in p:
                idx = int(i)
                if 0 <= idx < n:
                    pred_set[j].add(idx)

    # --- Topological order ---
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        in_degree[j] = len(pred_set[j])

    q = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # --- Forward pass ---
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        for v in succs[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
        EF[u] = ES[u] + durations[u]

    makespan_forward = float(np.max(EF)) if n > 0 else 0.0

    # --- Backward pass ---
    LS = np.zeros(n)
    LF = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) == 0:
            LF[u] = makespan_forward
        else:
            LF[u] = float(np.min([LS[v] for v in succs[u]]))
        LS[u] = LF[u] - durations[u]

    slack = LS - ES

    # --- Compute features for each activity ---
    eps = 1e-9
    
    # Precompute some stats
    max_demand_ratio = np.zeros(n)
    total_demand = np.zeros(n)
    for i in range(n):
        total_demand[i] = np.sum(demands[i])
        for r in range(R):
            if capacities[r] > 0:
                ratio = demands[i, r] / capacities[r]
                if ratio > max_demand_ratio[i]:
                    max_demand_ratio[i] = ratio

    # Min slack of successors
    min_succ_slack = np.full(n, 1e6)
    for i in range(n):
        if succs[i]:
            min_succ_slack[i] = np.min([slack[v] for v in succs[i]])
        else:
            min_succ_slack[i] = 0.0 # No successors, slack doesn't matter as much for "blocking"

    # Total duration of predecessors
    pred_duration_sum = np.zeros(n)
    for i in range(n):
        if pred_set[i]:
            pred_duration_sum[i] = np.sum([durations[p] for p in pred_set[i]])

    features = np.zeros((n, 8))
    for i in range(n):
        s = slack[i] + eps
        d = durations[i] + eps
        features[i, 0] = 1.0 / s          # 1/Slack
        features[i, 1] = durations[i]     # Duration
        features[i, 2] = max_demand_ratio[i] # Resource Criticality (Max)
        features[i, 3] = total_demand[i]  # Resource Criticality (Total)
        features[i, 4] = float(len(succs[i])) # Successor Count
        features[i, 5] = min_succ_slack[i]    # Min Successor Slack
        features[i, 6] = float(len(pred_set[i])) # Predecessor Count
        features[i, 7] = pred_duration_sum[i]   # Predecessor Duration

    # Normalize features to [0, 1]
    feat_min = np.min(features, axis=0)
    feat_max = np.max(features, axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range < 1e-12] = 1.0
    features_norm = (features - feat_min) / feat_range

    # --- Fast NumPy SGS Simulator ---
    def simulate_sgs(prio_scores):
        """
        Serial SGS using NumPy vectorization for resource checks.
        """
        if n == 0:
            return 0.0
        
        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        
        unscheduled = set(range(n))
        
        # Track unscheduled predecessors count
        unscheduled_preds_count = np.zeros(n, dtype=int)
        for j in range(n):
            unscheduled_preds_count[j] = len(pred_set[j])
        
        # Priority heap for eligible activities (negative priority for max-heap)
        eligible_heap = []
        
        # Initially, activities with no predecessors are eligible
        for j in range(n):
            if unscheduled_preds_count[j] == 0:
                heappush(eligible_heap, (-prio_scores[j], j))
        
        # Pre-allocate masks for speed
        scheduled_mask = np.zeros(n, dtype=bool)
        
        while unscheduled:
            if not eligible_heap:
                # Fallback
                for j in unscheduled:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break
            
            # Get highest priority eligible activity
            neg_prio, j = heappop(eligible_heap)
            
            if j not in unscheduled:
                continue  # stale entry
            
            # Compute earliest start from precedence
            es_j = 0.0
            for p in pred_set[j]:
                cand = start_t[p] + durations[p]
                if cand > es_j:
                    es_j = cand
            
            # Find candidate times
            # Only consider finish times of scheduled activities that are >= es_j
            # And es_j itself
            
            # Get indices of scheduled activities
            sched_indices = np.where(scheduled_mask)[0]
            
            cand_times = [es_j]
            if len(sched_indices) > 0:
                f_times = finish_t[sched_indices]
                valid_f = f_times[f_times >= es_j]
                if len(valid_f) > 0:
                    cand_times.extend(valid_f.tolist())
            
            cand_times = sorted(set(cand_times))
            
            best_start = None
            for t in cand_times:
                if t < es_j - 1e-9:
                    continue
                
                end_t = t + durations[j]
                
                # Check resource feasibility
                # Active if start_k < end_t and finish_k > t
                # We only check scheduled activities
                
                if len(sched_indices) == 0:
                    best_start = t
                    break
                
                s_t = start_t[sched_indices]
                f_t = finish_t[sched_indices]
                
                active_mask = (s_t < end_t) & (f_t > t)
                
                if np.any(active_mask):
                    # Sum demands of active activities
                    # demands[sched_indices] is (M, R)
                    # active_mask is (M,)
                    # We need to sum demands of rows where active_mask is True
                    active_dems = demands[sched_indices][active_mask]
                    total_demand = np.sum(active_dems, axis=0)
                    
                    if np.all(total_demand + demands[j] <= capacities + 1e-9):
                        best_start = t
                        break
                else:
                    best_start = t
                    break
            
            if best_start is None:
                best_start = es_j
            
            start_t[j] = best_start
            finish_t[j] = best_start + durations[j]
            unscheduled.discard(j)
            scheduled_mask[j] = True
            
            # Update eligibility of successors
            for v in succs[j]:
                unscheduled_preds_count[v] -= 1
                if unscheduled_preds_count[v] == 0 and v in unscheduled:
                    heappush(eligible_heap, (-prio_scores[v], v))
        
        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return makespan

    # --- Optimize weights using Nelder-Mead ---
    n_features = features_norm.shape[1]

    # Initialize weights
    # Heuristic: 1/slack is usually most important
    w0 = np.zeros(n_features)
    w0[0] = 1.0
    w0[1] = 0.5
    w0[2] = 0.5
    
    best_weights = w0.copy()
    best_ms = simulate_sgs(features_norm @ best_weights)
    
    # Nelder-Mead implementation
    def nelder_mead(fun, x0, max_iter=1000, timeout=100):
        # Simple Nelder-Mead
        n = len(x0)
        # Initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            xi = x0.copy()
            xi[i] += 0.1 if x0[i] == 0 else 0.1 * x0[i]
            simplex.append(xi)
        
        # Evaluate initial simplex
        f_values = [fun(x) for x in simplex]
        
        # Sort
        for i in range(n + 1):
            min_idx = i + np.argmin(f_values[i:])
            simplex[i], simplex[min_idx] = simplex[min_idx], simplex[i]
            f_values[i], f_values[min_idx] = f_values[min_idx], f_values[i]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        start_nm = time.time()
        for it in range(max_iter):
            if time.time() - start_nm > timeout:
                break
            
            # Reflect
            x_worst = simplex[-1]
            f_worst = f_values[-1]
            
            centroid = np.sum(simplex[:-1], axis=0) / n
            x_r = centroid + alpha * (centroid - x_worst)
            f_r = fun(x_r)
            
            if f_r < f_values[0]:
                # Expand
                x_e = centroid + gamma * (x_r - centroid)
                f_e = fun(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contract
                x_c = centroid + rho * (x_worst - centroid)
                f_c = fun(x_c)
                if f_c < f_worst:
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        f_values[i] = fun(simplex[i])
            
            # Sort
            for i in range(n + 1):
                min_idx = i + np.argmin(f_values[i:])
                simplex[i], simplex[min_idx] = simplex[min_idx], simplex[i]
                f_values[i], f_values[min_idx] = f_values[min_idx], f_values[i]
        
        # Return best
        best_idx = np.argmin(f_values)
        return simplex[best_idx], f_values[best_idx]

    # Wrapper for Nelder-Mead
    def objective(weights):
        scores = features_norm @ weights
        return simulate_sgs(scores)

    # Run Nelder-Mead with time limit
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        # Use a subset of time for NM
        nm_time = min(remaining_time - 5, 60)
        best_w, best_f = nelder_mead(objective, best_weights, timeout=nm_time)
        
        # Update best
        if best_f < best_ms:
            best_ms = best_f
            best_weights = best_w

    # Final polish with coordinate descent if time remains
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        step_sizes = np.ones(n_features) * 0.1
        no_improve = 0
        while time.time() - start_time < DEADLINE - 2:
            improved = False
            for fi in range(n_features):
                for direction in [1, -1]:
                    trial_w = best_weights.copy()
                    trial_w[fi] += direction * step_sizes[fi]
                    trial_w = np.clip(trial_w, -10, 10)
                    trial_ms = simulate_sgs(features_norm @ trial_w)
                    if trial_ms < best_ms:
                        best_weights = trial_w
                        best_ms = trial_ms
                        improved = True
                        no_improve = 0
                        step_sizes[fi] *= 1.2
                        break
                    else:
                        step_sizes[fi] *= 0.8
            if not improved:
                no_improve += 1
                if no_improve > 20:
                    break

    best_scores = features_norm @ best_weights
    return best_scores
# EVOLVE-BLOCK-END
