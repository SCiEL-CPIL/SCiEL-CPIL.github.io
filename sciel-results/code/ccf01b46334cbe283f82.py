"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def _simulate(priority_scores, times, predecessors, successors, n, cycle_time):
    """
    Simulate greedy station filling with given priority scores.
    Returns number of stations used.
    """
    completed = np.zeros(n, dtype=bool)
    stations = 0
    current_load = 0.0
    
    # Precompute predecessor sets for fast lookup
    pred_sets = [set(predecessors[i]) if i < len(predecessors) else set() for i in range(n)]
    
    while not np.all(completed):
        # Find eligible tasks: all preds completed, fits in remaining cycle time
        eligible = []
        for task in range(n):
            if completed[task]:
                continue
            if all(completed[p] for p in pred_sets[task]):
                if times[task] <= cycle_time - current_load:
                    eligible.append(task)
        
        if not eligible:
            # Move to next station
            stations += 1
            current_load = 0.0
            for task in range(n):
                if completed[task]:
                    continue
                if all(completed[p] for p in pred_sets[task]):
                    if times[task] <= cycle_time:
                        eligible.append(task)
        
        if not eligible:
            # Infeasible or bug; break to avoid infinite loop
            stations += 1
            break
        
        # Pick highest priority
        best_task = max(eligible, key=lambda t: priority_scores[t])
        completed[best_task] = True
        current_load += times[best_task]
    
    return stations

def _solve_lp_duals(times, predecessors, successors, n, cycle_time):
    """
    Solve LP relaxation to get dual variables for priority scoring.
    
    Formulation:
    Primal (minimize number of stations):
      Minimize: sum_s z_s
      Subject to:
        sum_s x_{i,s} = 1 for all i           (each task in exactly one station)
        sum_i t_i * x_{i,s} <= C * z_s for all s  (capacity)
        x_{i,s} - x_{p,s-1} <= 0 for all (p,i) in P, s>=2  (precedence)
        x_{i,s} >= 0, z_s >= 0
    
    The dual variables associated with the assignment constraints (sum_s x_{i,s} = 1)
    give the marginal cost of assigning task i, which serves as its priority score.
    
    For computational tractability with n=30 tasks, we use a reduced formulation:
    We solve a smaller LP that captures the essential structure.
    """
    from scipy.optimize import linprog
    
    # Compute topological order
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        preds = predecessors[j] if j < len(predecessors) else []
        in_degree[j] = len(preds)
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree.copy()
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                queue.append(v)
    
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in topo_order]
        topo_order.extend(remaining)
    
    # Positional weight (longest path to end)
    pos_weights = times.copy()
    for u in reversed(topo_order):
        for v in successors[u]:
            pos_weights[u] += pos_weights[v]
    
    # Depth from start
    depth_from_start = np.zeros(n)
    for u in topo_order:
        for v in successors[u]:
            depth_from_start[v] = max(depth_from_start[v], depth_from_start[u] + times[u])
    
    # Immediate successor/predecessor counts
    isc = np.array([len(successors[i]) for i in range(n)], dtype=float)
    ipc = np.array([len(predecessors[i]) if i < len(predecessors) else 0 for i in range(n)], dtype=float)
    
    # Determine number of stations needed (lower bound)
    total_time = np.sum(times)
    min_stations_lb = int(np.ceil(total_time / cycle_time))
    # Also consider critical path
    critical_path_len = np.max(pos_weights) if n > 0 else 0
    min_stations_cp = int(np.ceil(critical_path_len / cycle_time))
    S = max(min_stations_lb, min_stations_cp, 1)
    # Add a small buffer
    S = min(S + 2, n)  # Cap at n
    
    # Build LP:
    # Variables: x_{i,s} for i in range(n), s in range(S)
    #           z_s for s in range(S)
    # Total variables: n*S + S
    
    num_x = n * S
    num_z = S
    num_vars = num_x + num_z
    
    # Objective: minimize sum of z_s
    c = np.zeros(num_vars)
    for s in range(S):
        c[num_x + s] = 1.0
    
    # Constraints:
    # 1. Assignment: sum_s x_{i,s} = 1 for each i
    #    A_eq rows: n rows, each with 1s in the S positions for task i
    A_eq = np.zeros((n, num_vars))
    b_eq = np.ones(n)
    for i in range(n):
        for s in range(S):
            A_eq[i, i * S + s] = 1.0
    
    # 2. Capacity: sum_i t_i * x_{i,s} <= C * z_s for each s
    #    sum_i t_i * x_{i,s} - C * z_s <= 0
    A_ub_cap = np.zeros((S, num_vars))
    b_ub_cap = np.zeros(S)
    for s in range(S):
        for i in range(n):
            A_ub_cap[s, i * S + s] = times[i]
        A_ub_cap[s, num_x + s] = -cycle_time
    
    # 3. Precedence: x_{i,s} - x_{p,s-1} <= 0 for each (p,i) in P, s >= 1 (0-indexed: s>=1)
    #    x_{i,s} - x_{p,s-1} <= 0
    prec_constraints = []
    prec_bounds = []
    for s in range(1, S):
        for j in range(n):
            preds = predecessors[j] if j < len(predecessors) else []
            for p in preds:
                row = np.zeros(num_vars)
                row[j * S + s] = 1.0       # x_{j,s}
                row[p * S + (s-1)] = -1.0   # -x_{p,s-1}
                prec_constraints.append(row)
                prec_bounds.append(0.0)
    
    # Combine upper bound constraints
    if prec_constraints:
        A_ub = np.vstack([A_ub_cap] + prec_constraints)
        b_ub = np.concatenate([b_ub_cap, prec_bounds])
    else:
        A_ub = A_ub_cap
        b_ub = b_ub_cap
    
    # Variable bounds: x >= 0, z >= 0
    bounds = [(0, None)] * num_vars
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
        
        if result.success:
            # Dual variables from equality constraints (assignment constraints)
            # These represent the marginal cost of each task's assignment
            dual_assignment = result.ineqlin if hasattr(result, 'ineqlin') else None
            
            # Actually, we need the dual of the equality constraints
            # In scipy's linprog with HiGHS, the dual values for equality constraints
            # are in result.ineqlin or we need to extract them differently
            
            # Let's use the shadow prices / dual values
            # The dual variables for the assignment constraints give us what we want
            # For equality constraints, the dual is in result.ineqlin (if available)
            # or we can compute it from the solution
            
            # Alternative: use the dual values from the Lagrangian
            # The priority for task i is the dual value of constraint sum_s x_{i,s} = 1
            # This can be extracted from the LP solution
            
            # For HiGHS, dual values for equality constraints are in result.ineqlin
            # But actually they might be in a different attribute. Let's try to get them.
            
            # If we can't easily extract duals, use a heuristic based on the solution
            # The optimal x_{i,s} tells us which station each task goes to
            # Tasks that are "tight" (constrained) get higher priority
            
            # Extract assignment from LP solution
            x_opt = result.x[:num_x].reshape(n, S)
            
            # Priority: based on how "critical" each task is
            # Use positional weight as primary, adjusted by LP solution
            # Tasks assigned to earlier stations with high positional weight are most critical
            
            # Compute a refined priority using the LP structure
            # The dual of the capacity constraints gives the "price" of station capacity
            # Tasks that consume capacity in bottleneck stations should have higher priority
            
            # For now, use a combination:
            # - Positional weight (dominant)
            # - LP-based adjustment: tasks that are "binding" in the LP get boosted
            
            # Get dual values for capacity constraints if available
            # In HiGHS, these might be in result.ineqlin
            cap_duals = None
            if hasattr(result, 'ineqlin') and result.ineqlin is not None:
                # ineqlin contains duals for inequality constraints
                # First S are capacity constraints
                cap_duals = result.ineqlin[:S]
            
            # Build priority
            max_pw = np.max(pos_weights) if np.max(pos_weights) > 0 else 1
            max_t = np.max(times) if np.max(times) > 0 else 1
            max_isc = np.max(isc) if np.max(isc) > 0 else 1
            max_ipc = np.max(ipc) if np.max(ipc) > 0 else 1
            max_dfs = np.max(depth_from_start) if np.max(depth_from_start) > 0 else 1
            
            # Base priority from positional weight
            priority = pos_weights / max_pw
            
            # Add time component
            priority += 0.3 * (times / max_t)
            
            # Add successor count (bottleneck indicator)
            priority += 0.1 * (isc / max_isc)
            
            # Subtract predecessor count (flexibility)
            priority -= 0.05 * (ipc / max_ipc)
            
            # Add depth component
            priority += 0.1 * (depth_from_start / max_dfs)
            
            # If we have capacity duals, use them to adjust
            if cap_duals is not None and np.any(cap_duals > 1e-6):
                # Tasks assigned to stations with high dual values are more critical
                station_criticality = np.zeros(n)
                for i in range(n):
                    for s in range(S):
                        if x_opt[i, s] > 0.5:  # Task i assigned to station s
                            station_criticality[i] += cap_duals[s] * times[i]
                
                if np.max(station_criticality) > 1e-6:
                    priority += 0.2 * (station_criticality / np.max(station_criticality))
            
            # Normalize
            if np.max(priority) - np.min(priority) > 1e-6:
                priority = (priority - np.min(priority)) / (np.max(priority) - np.min(priority))
            
            return priority, pos_weights, depth_from_start, isc, ipc
    
    except Exception:
        pass
    
    # Fallback: heuristic priority
    max_pw = np.max(pos_weights) if np.max(pos_weights) > 0 else 1
    max_t = np.max(times) if np.max(times) > 0 else 1
    max_isc = np.max(isc) if np.max(isc) > 0 else 1
    max_ipc = np.max(ipc) if np.max(ipc) > 0 else 1
    max_dfs = np.max(depth_from_start) if np.max(depth_from_start) > 0 else 1
    
    priority = (0.55 * (pos_weights / max_pw) + 
                0.30 * (times / max_t) + 
                0.10 * (isc / max_isc) + 
                -0.05 * (ipc / max_ipc) + 
                0.10 * (depth_from_start / max_dfs))
    
    if np.max(priority) - np.min(priority) > 1e-6:
        priority = (priority - np.min(priority)) / (np.max(priority) - np.min(priority))
    
    return priority, pos_weights, depth_from_start, isc, ipc

def _compute_features(times, predecessors, successors, n):
    """Compute various priority features for tasks."""
    # Topological order
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        preds = predecessors[j] if j < len(predecessors) else []
        in_degree[j] = len(preds)
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree.copy()
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                queue.append(v)
    
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in topo_order]
        topo_order.extend(remaining)
    
    # Positional weight (longest path to end)
    pos_weights = times.copy()
    for u in reversed(topo_order):
        for v in successors[u]:
            pos_weights[u] += pos_weights[v]
    
    # Depth from start (longest path from any source to i)
    depth_from_start = np.zeros(n)
    for u in topo_order:
        for v in successors[u]:
            depth_from_start[v] = max(depth_from_start[v], depth_from_start[u] + times[u])
    
    # Immediate counts
    isc = np.array([len(successors[i]) for i in range(n)], dtype=float)
    ipc = np.array([len(predecessors[i]) if i < len(predecessors) else 0 for i in range(n)], dtype=float)
    
    # Normalize features to [0, 1]
    max_pw = np.max(pos_weights) if np.max(pos_weights) > 0 else 1
    max_t = np.max(times) if np.max(times) > 0 else 1
    max_dfs = np.max(depth_from_start) if np.max(depth_from_start) > 0 else 1
    max_isc = np.max(isc) if np.max(isc) > 0 else 1
    max_ipc = np.max(ipc) if np.max(ipc) > 0 else 1
    
    features = {
        "pos_weights": pos_weights / max_pw,
        "times": times / max_t,
        "isc": isc / max_isc,
        "ipc": ipc / max_ipc,
        "depth_from_start": depth_from_start / max_dfs,
        "raw_pos_weights": pos_weights,
        "raw_times": times,
        "raw_isc": isc,
        "raw_ipc": ipc,
        "raw_depth": depth_from_start
    }
    
    return features, topo_order

def _local_search_weights(features, times, predecessors, successors, n, cycle_time, 
                          start_weights=None, max_iter=50, deadline=None):
    """
    Local search over weight combinations to find priority vector that minimizes stations.
    """
    feature_keys = ["pos_weights", "times", "isc", "ipc", "depth_from_start"]
    num_features = len(feature_keys)
    
    # Initialize weights
    if start_weights is None:
        weights = np.array([0.55, 0.30, 0.10, -0.05, 0.10])
    else:
        weights = start_weights.copy()
    
    # Compute initial priority
    feature_matrix = np.column_stack([features[k] for k in feature_keys])
    priority = feature_matrix @ weights
    
    # Normalize to avoid numerical issues
    if np.max(priority) - np.min(priority) > 1e-6:
        priority = (priority - np.min(priority)) / (np.max(priority) - np.min(priority))
    
    best_stations = _simulate(priority, times, predecessors, successors, n, cycle_time)
    best_weights = weights.copy()
    
    # Local search: perturb weights and keep if improvement
    step_size = 0.05
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        for dim in range(num_features):
            # Try increasing weight
            trial_weights = best_weights.copy()
            trial_weights[dim] += step_size
            trial_priority = feature_matrix @ trial_weights
            if np.max(trial_priority) - np.min(trial_priority) > 1e-6:
                trial_priority = (trial_priority - np.min(trial_priority)) / (np.max(trial_priority) - np.min(trial_priority))
            trial_stations = _simulate(trial_priority, times, predecessors, successors, n, cycle_time)
            if trial_stations < best_stations:
                best_stations = trial_stations
                best_weights = trial_weights
                improved = True
                break
        
        # Try decreasing weight
        for dim in range(num_features):
            trial_weights = best_weights.copy()
            trial_weights[dim] -= step_size
            trial_priority = feature_matrix @ trial_weights
            if np.max(trial_priority) - np.min(trial_priority) > 1e-6:
                trial_priority = (trial_priority - np.min(trial_priority)) / (np.max(trial_priority) - np.min(trial_priority))
            trial_stations = _simulate(trial_priority, times, predecessors, successors, n, cycle_time)
            if trial_stations < best_stations:
                best_stations = trial_stations
                best_weights = trial_weights
                improved = True
                break
        
        if not improved:
            # Try smaller step
            step_size *= 0.7
            if step_size < 0.01:
                break
    
    # Compute final priority from best weights
    final_priority = feature_matrix @ best_weights
    if np.max(final_priority) - np.min(final_priority) > 1e-6:
        final_priority = (final_priority - np.min(final_priority)) / (np.max(final_priority) - np.min(final_priority))
    
    return final_priority, best_stations

def priority(instance):
    start_time = time.time()
    deadline = 115  # Safety margin before 120s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([])
    
    cycle_time = instance.get("cycle_time", 40)
    
    # Build successor graph
    successors = [[] for _ in range(n)]
    for j in range(n):
        preds = predecessors[j] if j < len(predecessors) else []
        for i in preds:
            successors[i].append(j)
    
    # Compute features
    features, topo_order = _compute_features(times, predecessors, successors, n)
    
    # Start with LP-inspired priority
    lp_priority, pos_weights, depth_from_start, isc, ipc = _solve_lp_duals(
        times, predecessors, successors, n, cycle_time
    )
    lp_stations = _simulate(lp_priority, times, predecessors, successors, n, cycle_time)
    
    # Check time
    if time.time() - start_time > deadline - 10:
        return lp_priority
    
    # Try local search to find better weight combination
    try:
        # Use normalized features for local search
        feature_keys = ["pos_weights", "times", "isc", "ipc", "depth_from_start"]
        feature_matrix = np.column_stack([features[k] for k in feature_keys])
        
        # Initial weights from LP dual intuition
        init_weights = np.array([0.55, 0.30, 0.10, -0.05, 0.10])
        
        # Run local search
        search_priority, search_stations = _local_search_weights(
            features, times, predecessors, successors, n, cycle_time,
            start_weights=init_weights,
            max_iter=40,
            deadline=deadline - 5
        )
        
        # Check if search improved
        if search_stations < lp_stations:
            return search_priority
        else:
            return lp_priority
            
    except Exception:
        return lp_priority
# EVOLVE-BLOCK-END
