"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 140.0  # safety margin before 150s kill


def _greedy_assign(times, preds, cycle_time, priority_scores, succ=None, remaining_preds_init=None):
    """Simulate the greedy station filling with given priority scores."""
    n = len(times)
    if n == 0:
        return 0

    if succ is None:
        succ = [[] for _ in range(n)]
        for i in range(n):
            for p in preds[i]:
                succ[p].append(i)

    if remaining_preds_init is None:
        remaining_preds_init = [len(preds[i]) for i in range(n)]

    remaining_preds = remaining_preds_init[:]
    # Use a set for faster removal and check if eligible
    # Note: Python sets are unordered, so we need to find the best eligible task dynamically
    # For n=30, a list is fine, but let's use a list and mark removed items or just filter.
    # Filtering a list of size <=30 is very fast.
    eligible_list = [i for i in range(n) if remaining_preds[i] == 0]

    stations = 0
    remaining = n
    while remaining > 0:
        stations += 1
        load = 0.0
        while True:
            best_task = -1
            best_prio = -1e18
            # Find the highest priority task that fits
            for i in eligible_list:
                if times[i] <= cycle_time - load + 1e-9:
                    if priority_scores[i] > best_prio:
                        best_prio = priority_scores[i]
                        best_task = i
            
            if best_task == -1:
                break
            
            load += times[best_task]
            remaining -= 1
            # Remove best_task from eligible_list
            # List removal is O(n), n is small (30)
            eligible_list.remove(best_task)
            
            for s in succ[best_task]:
                remaining_preds[s] -= 1
                if remaining_preds[s] == 0:
                    eligible_list.append(s)

    return stations


def _build_graph(preds, n):
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
    remaining_preds_init = [len(preds[i]) for i in range(n)]
    return succ, remaining_preds_init


def _compute_features(times, preds, n, succ, remaining_preds_init):
    """Compute multiple priority features."""
    # Topological order
    in_degree = remaining_preds_init[:]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for s in succ[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)

    # Bottom-level rank (longest path from task to end)
    bottom_level = np.zeros(n, dtype=float)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        max_succ = 0.0
        for s in succ[node]:
            if bottom_level[s] > max_succ:
                max_succ = bottom_level[s]
        bottom_level[node] = times[node] + max_succ

    # Top-level rank (longest path from start to task)
    top_level = np.zeros(n, dtype=float)
    for i in range(len(topo_order)):
        node = topo_order[i]
        max_pred = 0.0
        for p in preds[node]:
            if top_level[p] + times[p] > max_pred:
                max_pred = top_level[p] + times[p]
        top_level[node] = max_pred

    # Number of successors
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)

    # Number of predecessors
    num_pred = np.array(remaining_preds_init, dtype=float)

    # Sum of predecessor times (weight)
    pred_weight = np.zeros(n, dtype=float)
    for i in range(n):
        for p in preds[i]:
            pred_weight[i] += times[p]

    # Position in topological order
    pos_in_topo = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        pos_in_topo[node] = idx

    return {
        'bottom_level': bottom_level,
        'top_level': top_level,
        'num_succ': num_succ,
        'num_pred': num_pred,
        'pred_weight': pred_weight,
        'pos_in_topo': pos_in_topo,
        'times': times.copy(),
    }


def _normalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def priority(instance):
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    if n == 0:
        return np.array([])

    succ, remaining_preds_init = _build_graph(preds, n)
    features = _compute_features(times, preds, n, succ, remaining_preds_init)

    bl_norm = _normalize(features['bottom_level'])
    t_norm = _normalize(features['times'])
    ns_norm = _normalize(features['num_succ'])
    tl_norm = _normalize(features['top_level'])
    np_norm = _normalize(features['num_pred'])
    pw_norm = _normalize(features['pred_weight'])
    pos_norm = _normalize(features['pos_in_topo'])

    # Initial priority: weighted combination
    # Emphasize bottom-level (critical path to end) and time
    base_weights = [
        (1.0, 0.5, 0.3, 0.2, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0),
        (2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0),
        (1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1),
        (1.5, 0.8, 0.4, 0.3, 0.1, 0.1, 0.0),
        (1.0, 0.7, 0.3, 0.3, 0.1, 0.1, 0.0),
    ]

    best_score = None
    best_stations = float('inf')

    for w in base_weights:
        if time.time() - start_time > _DEADLINE:
            break
        s = (w[0] * bl_norm + w[1] * t_norm + w[2] * ns_norm + 
             w[3] * tl_norm + w[4] * np_norm + w[5] * pw_norm + w[6] * pos_norm)
        st = _greedy_assign(times, preds, cycle_time, s, succ, remaining_preds_init)
        if st < best_stations:
            best_stations = st
            best_score = s.copy()

    if best_score is None:
        # Fallback if no weights were tried due to timeout
        best_score = bl_norm
        best_stations = _greedy_assign(times, preds, cycle_time, best_score, succ, remaining_preds_init)

    # Simulated Annealing
    rng = np.random.default_rng(42)
    current_score = best_score
    current_stations = best_stations
    T = 0.5  # Initial temperature
    alpha = 0.99  # Cooling rate
    
    # Track best found
    global_best_score = current_score.copy()
    global_best_stations = current_stations
    no_improve_count = 0
    max_no_improve = 500  # Restart if stuck
    
    steps = 0
    max_steps = 50000  # Limit total steps to avoid excessive time, deadline check is primary
    
    while time.time() - start_time < _DEADLINE and steps < max_steps:
        # Generate neighbor
        # 1. Single task perturbation
        # 2. Pairwise swap
        if rng.random() < 0.7:
            idx1 = rng.integers(0, n)
            trial_score = current_score.copy()
            trial_score[idx1] += rng.normal(0, 0.1)
        else:
            idx1 = rng.integers(0, n)
            idx2 = rng.integers(0, n)
            if idx1 == idx2:
                continue
            trial_score = current_score.copy()
            trial_score[idx1], trial_score[idx2] = current_score[idx2], current_score[idx1]
        
        trial_stations = _greedy_assign(times, preds, cycle_time, trial_score, succ, remaining_preds_init)
        
        # Acceptance criteria
        if trial_stations < current_stations:
            # Improvement
            current_score = trial_score
            current_stations = trial_stations
            no_improve_count = 0
            if trial_stations < global_best_stations:
                global_best_stations = trial_stations
                global_best_score = trial_score.copy()
        else:
            no_improve_count += 1
            # Accept worse solution with probability exp(-delta/T)
            delta = trial_stations - current_stations
            if delta > 0 and T > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_score = trial_score
                    current_stations = trial_stations
            # Do not accept if delta == 0 to avoid cycling, or accept with low prob
            elif delta == 0 and rng.random() < 0.1:
                current_score = trial_score
        
        T *= alpha
        
        # Restart logic: If stuck, reset to global best and increase T
        if no_improve_count > max_no_improve:
            current_score = global_best_score.copy()
            current_stations = global_best_stations
            T = 0.5
            no_improve_count = 0
            max_no_improve = 1000 # Increase patience next time
            
        steps += 1

    return global_best_score
# EVOLVE-BLOCK-END
