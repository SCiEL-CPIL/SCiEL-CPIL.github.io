# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _topo_order(n, pred_lists):
    """Return topological order and successor adjacency."""
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            succ[p].append(i)
            indeg[i] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order, succ


def _downstream_stats(n, order, succ, times):
    """Compute longest_down and total_descendant_time via reverse topo."""
    ld = [0.0] * n
    td = [0.0] * n
    for u in reversed(order):
        best_ld = 0.0
        tot = 0.0
        for v in succ[u]:
            if times[v] + ld[v] > best_ld:
                best_ld = times[v] + ld[v]
            tot += times[v] + td[v]
        ld[u] = best_ld
        td[u] = tot
    return ld, td


def _greedy_stations(n, times, pred_lists, succ, cycle_time, priority_vec):
    """
    Simulate the greedy station-filling.
    Returns number of stations.
    """
    done = [False] * n
    remaining_count = n
    stations = 0
    
    while remaining_count > 0:
        stations += 1
        cap = cycle_time
        
        eligible = []
        for t in range(n):
            if done[t]:
                continue
            is_eligible = True
            for p in pred_lists[t]:
                if not done[p]:
                    is_eligible = False
                    break
            if is_eligible and times[t] <= cap:
                eligible.append(t)
        
        if not eligible:
            return 1000
        
        eligible.sort(key=lambda t: (-priority_vec[t], -times[t], t))
        
        assigned_any = False
        for t in eligible:
            if times[t] <= cap:
                cap -= times[t]
                done[t] = True
                remaining_count -= 1
                assigned_any = True
        
        if not assigned_any:
            return 1000
    
    return stations


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    pred_lists = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    if n == 0:
        return np.array([], dtype=float)

    t0 = time.time()
    DEADLINE = 135.0

    order, succ = _topo_order(n, pred_lists)
    ld, td = _downstream_stats(n, order, succ, times)

    lu = [0.0] * n
    for u in order:
        for v in succ[u]:
            c = lu[u] + times[u]
            if c > lu[v]:
                lu[v] = c

    cp_len = max(lu[i] + times[i] + ld[i] for i in range(n)) + 1e-9
    max_t = max(times) + 1e-9
    max_ld = max(ld) + 1e-9 if max(ld) > 0 else 1.0
    max_td = max(td) + 1e-9 if max(td) > 0 else 1.0
    max_lu = max(lu) + 1e-9 if max(lu) > 0 else 1.0
    max_succ = max([len(succ[i]) for i in range(n)]) + 1e-9
    max_pred = max([len(pred_lists[i]) for i in range(n)]) + 1e-9

    lu_a = np.array(lu, dtype=float)
    ld_a = np.array(ld, dtype=float)
    td_a = np.array(td, dtype=float)
    succ_a = np.array([len(succ[i]) for i in range(n)], dtype=float)
    pred_a = np.array([len(pred_lists[i]) for i in range(n)], dtype=float)
    cp_a = (lu_a + times + ld_a) / cp_len

    trans_succ = np.zeros(n, dtype=float)
    for u in reversed(order):
        for v in succ[u]:
            trans_succ[u] += 1 + trans_succ[v]
    max_trans = max(trans_succ) + 1e-9 if max(trans_succ) > 0 else 1.0

    norm_succ = succ_a / max_succ
    norm_time = times / max_t
    norm_ld = ld_a / max_ld
    norm_td = td_a / max_td
    norm_lu = lu_a / max_lu

    slack_a = cp_len - (lu_a + times + ld_a)
    min_slack = min(slack_a)
    max_slack = max(slack_a)
    range_slack = max_slack - min_slack + 1e-9
    norm_slack = (slack_a - min_slack) / range_slack

    candidates = []
    candidates.append(times.copy())
    candidates.append(norm_succ * 0.6 + norm_time * 0.4)
    candidates.append(norm_time * 0.5 + norm_ld * 0.5)
    candidates.append(norm_time * 0.4 + cp_a * 0.4 + norm_ld * 0.2)
    candidates.append(norm_ld * 0.5 + norm_td * 0.3 + norm_time * 0.2)
    candidates.append(norm_succ * 0.3 + norm_time * 0.3 + cp_a * 0.2 + norm_ld * 0.2)
    candidates.append((1.0 - pred_a / max_pred) * 0.3 + norm_time * 0.4 + cp_a * 0.3)
    combined = norm_ld * 0.6 + norm_time * 0.4
    idx = np.argsort(-combined)
    ranks = np.zeros(n)
    for r, t in enumerate(idx):
        ranks[t] = r
    max_r = max(n - 1, 1)
    candidates.append(1.0 - (ranks / max_r) ** 0.7)
    candidates.append(norm_time * cp_a * 0.5 + norm_time * 0.3 + norm_ld * 0.2)
    candidates.append(norm_td * 0.4 + norm_time * 0.3 + cp_a * 0.3)
    candidates.append(cp_a.copy())
    candidates.append((1.0 - norm_slack) * 0.5 + norm_time * 0.5)
    candidates.append(norm_lu * 0.5 + norm_time * 0.5)
    candidates.append(norm_time * 0.3 + norm_ld * 0.3 + norm_succ * 0.2 + cp_a * 0.2)
    candidates.append((trans_succ / max_trans) * 0.5 + norm_time * 0.3 + cp_a * 0.2)
    candidates.append(norm_time * norm_succ * 0.4 + norm_time * 0.3 + norm_ld * 0.3)
    candidates.append((1.0 - norm_slack) * 0.4 + norm_time * 0.4 + norm_succ * 0.2)
    cp_pos = lu_a / cp_len
    candidates.append(cp_pos * 0.3 + (1 - cp_pos) * 0.2 + norm_time * 0.5)
    sort_key = ld_a * 1000 + times
    idx19 = np.argsort(-sort_key)
    ranks19 = np.zeros(n)
    for r, t in enumerate(idx19):
        ranks19[t] = r
    max_r19 = max(n - 1, 1)
    candidates.append(1.0 - (ranks19 / max_r19) ** 0.5)
    candidates.append((times ** 2 / (max_t ** 2)) * 0.5 + norm_succ * 0.3 + cp_a * 0.2)

    best_priority = candidates[0]
    best_stations = _greedy_stations(n, times, pred_lists, succ, cycle_time, candidates[0])

    for c in candidates[1:]:
        if time.time() - t0 > DEADLINE:
            break
        s = _greedy_stations(n, times, pred_lists, succ, cycle_time, c)
        if s < best_stations:
            best_stations = s
            best_priority = c

    if time.time() - t0 < DEADLINE - 10:
        best_p = best_priority.copy()
        best_s = best_stations
        rng = np.random.default_rng(42)
        for _ in range(100):
            if time.time() - t0 > DEADLINE:
                break
            noise = rng.normal(0, 0.02, n)
            perturbed = best_p + noise
            s = _greedy_stations(n, times, pred_lists, succ, cycle_time, perturbed)
            if s < best_s:
                best_s = s
                best_p = perturbed
        if best_s < best_stations:
            best_priority = best_p

    return best_priority
# EVOLVE-BLOCK-END
