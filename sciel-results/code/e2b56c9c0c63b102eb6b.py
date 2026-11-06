"""Assembly line balancing with advanced feature-based priority scoring and simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque
import math

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


def _compute_advanced_features(times, predecessors, cycle_time):
    """Compute advanced features including graph centrality."""
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
    cp_to_end = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp_to_end[u] = times[u]
        for v in successors[u]:
            cp_to_end[u] = max(cp_to_end[u], times[u] + cp_to_end[v])
    
    # Critical path (longest path from start to task)
    cp_from_start = np.zeros(n, dtype=float)
    for u in topo_order:
        cp_from_start[u] = times[u]
        for v in successors[u]:
            cp_from_start[v] = max(cp_from_start[v], cp_from_start[u] + times[v])
    
    # Total critical path length
    max_cp = max(cp_to_end) if n > 0 else 0
    
    # Position in topological order
    topo_pos = np.zeros(n, dtype=float)
    for idx, u in enumerate(topo_order):
        topo_pos[u] = idx / max(1, n - 1) # Normalized 0 to 1
    
    # Direct and transitive counts
    num_direct_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    num_direct_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
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
    
    # Slack: How much room is there if we start as early as possible?
    # Earliest start = cp_from_start - times
    # Latest start = cycle_time - cp_to_end
    # Slack = Latest start - Earliest start = cycle_time - cp_to_end - (cp_from_start - times)
    # Note: cp_from_start includes the task time.
    # Let's define Slack as: Cycle Time - (cp_from_start + cp_to_end - times)
    # This is the total slack available for this task.
    total_slack = cycle_time - (cp_from_start + cp_to_end - times)
    
    # Betweenness Centrality (approximate for DAGs)
    # For each pair (s, t), find shortest paths. Count how many pass through u.
    # This is O(V * (V+E)) which is fine for V=30.
    betweenness = np.zeros(n, dtype=float)
    # We only care about paths that respect precedence.
    # Actually, standard betweenness is on the graph structure.
    # Let's compute it on the DAG.
    
    def shortest_paths_dag(start):
        """Returns number of shortest paths from start to all nodes, and order of nodes."""
        dist = np.full(n, np.inf)
        num_paths = np.zeros(n, dtype=float)
        dist[start] = 0
        num_paths[start] = 1
        # Topological order is sufficient for DAG shortest paths
        for u in topo_order:
            if dist[u] == np.inf:
                continue
            for v in successors[u]:
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    num_paths[v] = num_paths[u]
                elif dist[v] == dist[u] + 1:
                    num_paths[v] += num_paths[u]
        return dist, num_paths

    for s in range(n):
        dist_s, num_s = shortest_paths_dag(s)
        # Backwards accumulation
        betweenness_local = np.zeros(n, dtype=float)
        # We need to accumulate from end to start
        # The formula for betweenness in DAG is slightly different than undirected.
        # C_B(v) = sum_{s < v < t} sigma_st(v) / sigma_st
        # We can iterate t in reverse topo order.
        for t in reversed(topo_order):
            if t == s:
                continue
            # Accumulate contributions from t to predecessors
            # This requires careful bookkeeping of which paths go through which nodes.
            # Given the small size, we can just count paths explicitly for simplicity if needed, 
            # but let's try a simpler proxy: Weighted degree.
            # Or simply use the number of pairs (s,t) such that v is on a shortest path.
            # For small N, explicit counting is faster to implement correctly.
            
    # Explicit Betweenness Count (O(N^3) worst case, but N=30 is tiny)
    def count_on_shortest_path(s, t, v):
        """Check if v is on ANY shortest path from s to t."""
        if s == v or t == v:
            return False
        dist_st, _ = shortest_paths_dag(s)
        dist_sv, _ = shortest_paths_dag(s)
        dist_vt, _ = shortest_paths_dag(v)
        # dist_vt is from v to all nodes. We need dist from v to t.
        # My shortest_paths_dag calculates dist from start to all.
        # So dist_vt[t] is dist(v, t).
        if dist_st[t] == np.inf:
            return False
        if dist_sv[v] + dist_vt[t] == dist_st[t]:
            return True
        return False

    for v in range(n):
        count = 0
        for s in range(n):
            for t in range(n):
                if s == v or t == v or s == t:
                    continue
                if count_on_shortest_path(s, t, v):
                    count += 1
        betweenness[v] = count
    
    # Feature matrix
    features = np.column_stack([
        times,                  # 0
        cp_to_end,              # 1
        cp_from_start,          # 2
        topo_pos,               # 3
        num_direct_succ,        # 4
        num_direct_pred,        # 5
        num_trans_succ,         # 6
        num_trans_pred,         # 7
        total_slack,            # 8
        betweenness,            # 9
        times * betweenness,    # 10
        1.0 / (1.0 + total_slack), # 11: Inverse slack (higher is more critical)
    ])
    
    return features


def _fit_and_optimize_sa(times, predecessors, cycle_time, deadline):
    """
    Simulated Annealing to optimize priority weights.
    """
    n = len(times)
    if n == 0:
        return np.array([])
    
    features = _compute_advanced_features(times, predecessors, cycle_time)
    if features is None:
        rng = np.random.RandomState(42)
        return rng.rand(n)
    
    num_features = features.shape[1]
    
    # Normalize features to have zero mean and unit variance to help with weight scales
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1.0
    features_norm = (features - mean) / std
    
    # Initial weights: Try a few heuristic starts
    rng = np.random.RandomState(42)
    
    # Heuristic 1: Maximize Slack (low slack = high priority) -> Minimize Slack
    # In our features, index 8 is total_slack. We want high priority for low slack.
    # So weight for slack should be negative.
    w1 = np.zeros(num_features)
    w1[8] = -1.0
    w1[9] = 1.0 # Betweenness
    
    # Heuristic 2: Maximize Time
    w2 = np.zeros(num_features)
    w2[0] = 1.0
    
    # Heuristic 3: Random
    w3 = rng.randn(num_features)
    
    candidates = [w1, w2, w3]
    
    best_w = None
    best_score = float('inf')
    
    for w in candidates:
        pri = features_norm @ w
        score = _greedy_balance(times, predecessors, cycle_time, pri)
        if score < best_score:
            best_score = score
            best_w = w.copy()
    
    # Simulated Annealing
    current_w = best_w.copy()
    current_score = best_score
    T = 1.0
    T_min = 0.01
    alpha = 0.95
    steps = 0
    max_steps = 5000
    
    while T > T_min and steps < max_steps:
        if time.time() > deadline - 10:
            break
            
        steps += 1
        
        # Propose new weights
        new_w = current_w + rng.randn(num_features) * 0.1
        new_pri = features_norm @ new_w
        new_score = _greedy_balance(times, predecessors, cycle_time, new_pri)
        
        # Acceptance criterion
        if new_score < current_score:
            current_w = new_w
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_w = current_w.copy()
        else:
            # Accept with probability exp(-(new_score - current_score) / T)
            # Since score is integer, the difference is at least 1.
            delta = new_score - current_score
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                current_w = new_w
                current_score = new_score
        
        T *= alpha
        
        if best_score == 1:
            break
    
    return features_norm @ best_w


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_priority = _fit_and_optimize_sa(times, predecessors, cycle_time, deadline)
    
    # Final polish: Random perturbations
    if time.time() < deadline - 5:
        rng = np.random.RandomState(42 + int(time.time() % 1000))
        current_best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)
        
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            perturbation = rng.randn(n) * 0.05
            trial_priority = best_priority + perturbation
            trial_score = _greedy_balance(times, predecessors, cycle_time, trial_priority)
            if trial_score < current_best_score:
                current_best_score = trial_score
                best_priority = trial_priority
                if current_best_score == 1:
                    break
    
    return best_priority
# EVOLVE-BLOCK-END
