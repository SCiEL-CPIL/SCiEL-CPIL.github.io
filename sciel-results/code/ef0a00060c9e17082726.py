"""Assembly line balancing with model-and-optimize priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0
    
    successors = defaultdict(list)
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    eligible = set(i for i in range(n) if unassigned_preds[i] == 0)
    
    if n > 0 and not eligible:
        return 10**9
    
    station_count = 1
    current_load = 0
    
    while eligible:
        # Find the highest-priority task that fits
        best_task = -1
        best_pri = -np.inf
        remaining = cycle_time - current_load
        
        for task in eligible:
            if times[task] <= remaining:
                if priority_scores[task] > best_pri:
                    best_pri = priority_scores[task]
                    best_task = task
        
        if best_task >= 0:
            eligible.discard(best_task)
            current_load += times[best_task]
            
            for succ in successors[best_task]:
                unassigned_preds[succ] -= 1
                if unassigned_preds[succ] == 0:
                    eligible.add(succ)
        else:
            # No eligible task fits; start new station
            station_count += 1
            current_load = 0
            # Check if any eligible task exceeds cycle time
            for t in eligible:
                if times[t] > cycle_time:
                    return 10**9
    
    return station_count


def _compute_features(times, predecessors, cycle_time):
    """Compute various features for each task."""
    n = len(times)
    if n == 0:
        return np.zeros((n, 0))
    
    successors = defaultdict(list)
    pred_map = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            pred_map[i].append(pred)
    
    # Topological sort
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    topo_order = []
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    if len(topo_order) != n:
        return None  # Cycle detected
    
    # Critical path (longest path from task to end)
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            cp[u] = max(cp[u], times[u] + cp[v])
    
    # Position in topological order (0-indexed, earlier = smaller)
    topo_pos = np.zeros(n, dtype=float)
    for idx, u in enumerate(topo_order):
        topo_pos[u] = idx
    
    # Number of direct successors
    num_direct_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Number of direct predecessors
    num_direct_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Transitive successor count using BFS
    def count_trans(start, adj):
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for s in adj[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        return len(visited)
    
    num_trans_succ = np.array([count_trans(i, successors) for i in range(n)], dtype=float)
    num_trans_pred = np.array([count_trans(i, pred_map) for i in range(n)], dtype=float)
    
    # Slack: cycle_time - time (positive means there's room)
    slack = cycle_time - times
    
    # Feature matrix
    features = np.column_stack([
        times,          # 0: processing time
        cp,             # 1: critical path
        topo_pos,       # 2: topological position
        num_direct_succ, # 3: direct successors
        num_direct_pred, # 4: direct predecessors
        num_trans_succ,  # 5: transitive successors
        num_trans_pred,  # 6: transitive predecessors
        slack,          # 7: slack
        times * cp,     # 8: time * critical path
        num_trans_succ / (n + 1),  # 9: normalized transitive successors
        1.0 / (1.0 + num_direct_pred),  # 10: inverse pred factor
    ])
    
    return features


def _fit_and_optimize(times, predecessors, cycle_time, deadline):
    """
    Model-and-optimize: parameterize priority as linear combination of features,
    fit coefficients by trying many combinations and selecting the best.
    """
    n = len(times)
    if n == 0:
        return np.array([])
    
    features = _compute_features(times, predecessors, cycle_time)
    if features is None:
        # Cycle in predecessors; return random priorities
        rng = np.random.RandomState(42)
        return rng.rand(n)
    
    num_features = features.shape[1]
    
    # Precompute some candidate weight vectors
    candidates = []
    
    # Individual features
    for f in range(num_features):
        candidates.append(np.eye(num_features)[f])
    
    # All ones (average)
    candidates.append(np.ones(num_features) / num_features)
    
    # Random directions
    rng = np.random.RandomState(42)
    for _ in range(50):
        w = rng.randn(num_features)
        candidates.append(w / (np.linalg.norm(w) + 1e-9))
    
    # Grid search over key weights
    time_w = [0.0, 0.5, 1.0, 1.5, 2.0]
    cp_w = [0.0, 0.5, 1.0, 1.5, 2.0]
    succ_w = [0.0, 0.5, 1.0]
    pred_w = [0.0, -1.0, -2.0]
    
    for tw in time_w:
        for cw in cp_w:
            for sw in succ_w:
                for pw in pred_w:
                    w = np.zeros(num_features)
                    w[0] = tw      # time
                    w[1] = cw      # cp
                    w[4] = pw      # num_direct_pred (negative to favor fewer)
                    w[5] = sw      # num_trans_succ
                    norm = np.linalg.norm(w)
                    if norm > 1e-9:
                        w = w / norm
                    candidates.append(w)
    
    # Also try: high time, high cp, low topological position
    for tw in [1.0, 2.0]:
        for cw in [1.0, 2.0]:
            for tpw in [-1.0, -2.0]:
                w = np.zeros(num_features)
                w[0] = tw
                w[1] = cw
                w[2] = tpw  # topological position (negative to favor earlier)
                norm = np.linalg.norm(w)
                if norm > 1e-9:
                    w = w / norm
                candidates.append(w)
    
    # Remove duplicates approximately
    unique_candidates = [candidates[0]]
    for c in candidates[1:]:
        is_dup = False
        for uc in unique_candidates:
            if np.allclose(c, uc, atol=1e-6):
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
    
    candidates = unique_candidates
    
    best_score = float('inf')
    best_priority = None
    best_weights = None
    
    for idx, w in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        
        pri = features @ w
        n_stations = _greedy_balance(times, predecessors, cycle_time, pri)
        
        if n_stations < best_score:
            best_score = n_stations
            best_priority = pri
            best_weights = w
            if best_score == 1:
                break
    
    if best_priority is None:
        best_priority = features[:, 1]
    
    # Try to improve with local search: perturb best weights
    if best_weights is not None and time.time() < deadline - 15:
        current_best = best_score
        current_weights = best_weights.copy()
        
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for dim in range(num_features):
                for delta in [-0.5, -0.2, 0.2, 0.5]:
                    new_w = current_weights.copy()
                    old_val = new_w[dim]
                    new_w[dim] = old_val + delta
                    
                    pri = features @ new_w
                    n_stations = _greedy_balance(times, predecessors, cycle_time, pri)
                    
                    if n_stations < current_best:
                        current_best = n_stations
                        current_weights = new_w
                        improved = True
                        if current_best == 1:
                            break
                if current_best == 1:
                    break
            if not improved:
                break
            if current_best == 1:
                break
        
        best_priority = features @ current_weights
        best_score = current_best
    
    return best_priority


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Deadline guard: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Model-and-optimize approach
    best_priority = _fit_and_optimize(times, predecessors, cycle_time, deadline)
    
    # If time remains, try a few more random perturbations of the best
    if time.time() < deadline - 5:
        rng = np.random.RandomState(42 + int(time.time() % 1000))
        current_best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)
        
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            # Small random perturbation
            perturbation = rng.randn(n) * 0.1
            trial_priority = best_priority + perturbation
            trial_score = _greedy_balance(times, predecessors, cycle_time, trial_priority)
            if trial_score < current_best_score:
                current_best_score = trial_score
                best_priority = trial_priority
                if current_best_score == 1:
                    break
    
    return best_priority
# EVOLVE-BLOCK-END
