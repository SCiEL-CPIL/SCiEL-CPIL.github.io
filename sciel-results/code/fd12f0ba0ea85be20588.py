"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s kill


def _build_graph(preds, n):
    """Build successor lists and initial remaining predecessor counts."""
    succ = [[] for _ in range(n)]
    remaining_preds_init = [0] * n
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            remaining_preds_init[i] += 1
    return succ, remaining_preds_init


def _greedy_assign(times, preds, cycle_time, priority_scores, succ=None, remaining_preds_init=None):
    """Simulate the greedy station filling with given priority scores.
    Returns number of stations used."""
    n = len(times)
    if n == 0:
        return 0

    if succ is None:
        succ, remaining_preds_init = _build_graph(preds, n)

    remaining_preds = remaining_preds_init[:]
    eligible_list = [i for i in range(n) if remaining_preds[i] == 0]

    stations = 0
    remaining = n
    while remaining > 0:
        stations += 1
        load = 0.0
        while True:
            best_task = -1
            best_prio = -1e18
            for i in eligible_list:
                if times[i] <= cycle_time - load + 1e-9:
                    if priority_scores[i] > best_prio:
                        best_prio = priority_scores[i]
                        best_task = i
            if best_task == -1:
                break
            load += times[best_task]
            remaining -= 1
            eligible_list.remove(best_task)
            for s in succ[best_task]:
                remaining_preds[s] -= 1
                if remaining_preds[s] == 0:
                    eligible_list.append(s)

    return stations


def _compute_topo_order(succ, remaining_preds_init, n):
    """Compute topological order using Kahn's algorithm."""
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
    return topo_order


def _compute_features(times, preds, n, succ, remaining_preds_init):
    """Compute multiple priority features for the tasks."""
    topo_order = _compute_topo_order(succ, remaining_preds_init, n)

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

    # Position in topological order
    pos_in_topo = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        pos_in_topo[node] = idx

    # Downstream workload sum (sum of times of all successors, not just critical path)
    downstream_work = np.zeros(n, dtype=float)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        total_succ = 0.0
        for s in succ[node]:
            total_succ += times[s] + downstream_work[s]
        downstream_work[node] = times[node] + total_succ

    return {
        'bottom_level': bottom_level,
        'top_level': top_level,
        'num_succ': num_succ,
        'num_pred': num_pred,
        'pos_in_topo': pos_in_topo,
        'downstream_work': downstream_work,
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

    # Normalize features
    bl_norm = _normalize(features['bottom_level'])
    t_norm = _normalize(times)
    ns_norm = _normalize(features['num_succ'])
    tl_norm = _normalize(features['top_level'])
    dw_norm = _normalize(features['downstream_work'])

    # Initial priority: weighted combination emphasizing critical path and time
    # Higher score = higher priority
    score = 1.0 * bl_norm + 0.5 * t_norm + 0.3 * ns_norm + 0.2 * tl_norm + 0.3 * dw_norm

    best_score = score.copy()
    best_stations = _greedy_assign(times, preds, cycle_time, score, succ, remaining_preds_init)

    # Try several weight combinations
    weight_combos = [
        (1.0, 0.5, 0.3, 0.2, 0.3),
        (1.0, 1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0, 0.0, 0.0),
        (1.0, 0.5, 0.0, 0.5, 0.0),
        (1.0, 0.0, 0.0, 0.0, 1.0),
        (1.5, 0.5, 0.5, 0.0, 0.5),
        (1.0, 1.0, 0.5, 0.0, 0.5),
        (0.5, 1.0, 0.5, 0.5, 0.5),
        (1.0, 0.2, 0.2, 0.2, 0.2),
        (2.0, 0.5, 0.5, 0.0, 0.5),
        (1.0, 0.5, 0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, 0.5),
        (0.5, 0.5, 0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0, 0.0, 1.0),
        (1.0, 0.5, 0.3, 0.3, 0.3),
    ]

    for w_bl, w_t, w_ns, w_tl, w_dw in weight_combos:
        if time.time() - start_time > _DEADLINE:
            break
        s = w_bl * bl_norm + w_t * t_norm + w_ns * ns_norm + w_tl * tl_norm + w_dw * dw_norm
        st = _greedy_assign(times, preds, cycle_time, s, succ, remaining_preds_init)
        if st < best_stations:
            best_stations = st
            best_score = s.copy()

    # Local search: perturb the best score and see if we can improve
    if time.time() - start_time < _DEADLINE - 10:
        rng = np.random.RandomState(42)
        for _ in range(50):
            if time.time() - start_time > _DEADLINE:
                break
            perturbed = best_score + rng.normal(0, 0.1, n)
            st = _greedy_assign(times, preds, cycle_time, perturbed, succ, remaining_preds_init)
            if st < best_stations:
                best_stations = st
                best_score = perturbed.copy()

    return best_score
# EVOLVE-BLOCK-END
