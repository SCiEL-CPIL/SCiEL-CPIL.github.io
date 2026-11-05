"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _simulate(priority_scores, times, pred, cycle_time):
    """Simulate the greedy ALB: fill stations with highest-priority eligible task that fits."""
    n = len(times)
    pred_sets = [set(p) for p in pred]
    completed = set()
    num_stations = 0
    remaining = set(range(n))
    
    while remaining:
        num_stations += 1
        remaining_time = cycle_time
        
        # Find all eligible tasks: all predecessors completed
        eligible = [t for t in remaining if pred_sets[t].issubset(completed)]
        if not eligible:
            # Should not happen in a valid DAG, but guard against it
            return num_stations + 1000000
        
        # Sort eligible by priority (descending), then by time (descending) as tie-breaker
        eligible.sort(key=lambda t: (-priority_scores[t], -times[t], t))
        
        # Greedily assign tasks to this station
        station_tasks = []
        for t in eligible:
            if times[t] <= remaining_time:
                station_tasks.append(t)
                remaining_time -= times[t]
        
        # If no task fits (shouldn't happen if times <= cycle_time), force assign highest priority
        if not station_tasks:
            t = eligible[0]
            station_tasks = [t]
        
        for t in station_tasks:
            completed.add(t)
            remaining.discard(t)
    
    return num_stations


def priority(instance):
    """
    Score each assembly task for the greedy assembly line balancing heuristic.
    Uses a refined Critical Path Density approach with local optimization.
    """
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin before 120s kill
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    pred = instance["predecessors"]
    
    # Build successor graph
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Topological sort
    in_degree = [0] * n
    for i in range(n):
        in_degree[i] = len(pred[i])
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Precompute features for each task
    # Feature 1: Longest path from task to end (downstream distance)
    dist = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        max_succ = 0.0
        for v in succ[u]:
            max_succ = max(max_succ, dist[v])
        dist[u] = times[u] + max_succ
    
    # Feature 2: Task duration
    # Feature 3: Number of immediate successors
    n_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Feature 4: Total number of transitive successors
    total_succ = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0
        for v in succ[u]:
            s += 1 + total_succ[v]
        total_succ[u] = s
    
    # Feature 5: Longest path from start to task (upstream distance)
    up_dist = np.zeros(n, dtype=float)
    for u in top_order:
        max_pred = 0.0
        for p in pred[u]:
            max_pred = max(max_pred, up_dist[p])
        up_dist[u] = max_pred + times[u]
    
    # Feature 6: Normalized position in topological order
    topo_pos = np.zeros(n, dtype=float)
    for idx, u in enumerate(top_order):
        topo_pos[u] = idx / max(n - 1, 1)
    
    # Feature 7: Number of immediate predecessors
    n_pred = np.array([len(pred[i]) for i in range(n)], dtype=float)
    
    # Feature 8: Sum of times of immediate successors
    succ_time_sum = np.zeros(n, dtype=float)
    for u in range(n):
        for v in succ[u]:
            succ_time_sum[u] += times[v]
    
    # Feature 9: Product of time and downstream dist (bottleneck indicator)
    bottleneck = times * dist
    
    # Feature 10: Ratio of time to cycle time
    time_ratio = times / max(cycle_time, 1e-9)
    
    # Feature 11: Critical Path Density - ratio of task time to downstream path
    # This identifies tasks that are a large fraction of their downstream path
    cpd = np.zeros(n, dtype=float)
    for i in range(n):
        if dist[i] > 1e-9:
            cpd[i] = times[i] / dist[i]
        else:
            cpd[i] = 1.0 if times[i] > 0 else 0.0
    
    # Feature 12: Fan-out weighted by downstream distance (branching importance)
    fanout_weighted = np.zeros(n, dtype=float)
    for i in range(n):
        for v in succ[i]:
            fanout_weighted[i] += dist[v]
    
    # Feature 13: Upstream-downstream balance
    # Tasks with balanced upstream and downstream positions may be more critical
    balance = np.zeros(n, dtype=float)
    for i in range(n):
        if up_dist[i] > 1e-9 and dist[i] > 1e-9:
            balance[i] = min(up_dist[i], dist[i]) / max(up_dist[i], dist[i])
    
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    # Normalize all features
    feats = np.column_stack([
        normalize(dist),
        normalize(times),
        normalize(n_succ),
        normalize(total_succ),
        normalize(up_dist),
        normalize(topo_pos),
        normalize(n_pred),
        normalize(succ_time_sum),
        normalize(bottleneck),
        normalize(time_ratio),
        normalize(cpd),
        normalize(fanout_weighted),
        normalize(balance),
    ])
    
    # Number of features
    d = feats.shape[1]
    
    def simulate_with_weights(w):
        scores = feats @ w
        return _simulate(scores, times, pred, cycle_time)
    
    # Initial candidate weight vectors - focused on Critical Path Density
    candidates = []
    
    # 1. Pure dist
    w = np.zeros(d); w[0] = 1.0
    candidates.append(w)
    
    # 2. Pure times
    w = np.zeros(d); w[1] = 1.0
    candidates.append(w)
    
    # 3. Dist + times
    w = np.zeros(d); w[0] = 0.5; w[1] = 0.5
    candidates.append(w)
    
    # 4. Critical Path Density focused
    w = np.zeros(d); w[10] = 0.5; w[0] = 0.3; w[2] = 0.2
    candidates.append(w)
    
    # 5. CPD + fan-out weighted
    w = np.zeros(d); w[10] = 0.4; w[11] = 0.3; w[0] = 0.2; w[1] = 0.1
    candidates.append(w)
    
    # 6. Bottleneck focused
    w = np.zeros(d); w[8] = 0.7; w[0] = 0.3
    candidates.append(w)
    
    # 7. Upstream + downstream
    w = np.zeros(d); w[0] = 0.4; w[4] = 0.3; w[1] = 0.3
    candidates.append(w)
    
    # 8. All equal
    w = np.ones(d) / d
    candidates.append(w)
    
    # 9. Heavy on dist, light on others
    w = np.zeros(d); w[0] = 0.6; w[1] = 0.2; w[3] = 0.1; w[8] = 0.1
    candidates.append(w)
    
    # 10. Balance focused
    w = np.zeros(d); w[12] = 0.4; w[0] = 0.3; w[1] = 0.2; w[10] = 0.1
    candidates.append(w)
    
    # 11. CPD + bottleneck
    w = np.zeros(d); w[10] = 0.5; w[8] = 0.3; w[0] = 0.2
    candidates.append(w)
    
    # 12. Fan-out weighted + dist
    w = np.zeros(d); w[11] = 0.4; w[0] = 0.4; w[2] = 0.2
    candidates.append(w)
    
    # Evaluate all candidates
    best_stations = float('inf')
    best_w = candidates[0].copy()
    for w in candidates:
        st = simulate_with_weights(w)
        if st < best_stations:
            best_stations = st
            best_w = w.copy()
    
    # Deadline check
    if time.time() - start_time > DEADLINE - 10:
        scores = feats @ best_w
        return scores
    
    # Local optimization: coordinate descent / random perturbation
    best_stations_local = best_stations
    best_w_local = best_w.copy()
    
    # Random perturbation search
    rng = np.random.RandomState(42)
    time_per_trial = 0.02
    remaining_time = DEADLINE - 5 - (time.time() - start_time)
    max_trials = int(remaining_time / time_per_trial)
    max_trials = min(max_trials, 3000)
    
    for trial in range(max_trials):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Perturb the best weights
        w_new = best_w_local + rng.normal(0, 0.08, size=d)
        # Normalize to prevent large values
        w_norm = np.linalg.norm(w_new)
        if w_norm > 1e-12:
            w_new = w_new / w_norm * np.linalg.norm(best_w_local)
        
        st = simulate_with_weights(w_new)
        if st < best_stations_local:
            best_stations_local = st
            best_w_local = w_new.copy()
    
    # Coordinate descent polish
    for i in range(d):
        if time.time() - start_time > DEADLINE - 5:
            break
        for delta in [0.03, -0.03, 0.06, -0.06, 0.1, -0.1]:
            if time.time() - start_time > DEADLINE - 5:
                break
            w_new = best_w_local.copy()
            w_new[i] += delta
            st = simulate_with_weights(w_new)
            if st < best_stations_local:
                best_stations_local = st
                best_w_local = w_new.copy()
    
    # Final scores
    scores = feats @ best_w_local
    
    # Add tiny tie-breaker to ensure deterministic ordering
    scores = scores + 1e-8 * times
    
    return scores
# EVOLVE-BLOCK-END
