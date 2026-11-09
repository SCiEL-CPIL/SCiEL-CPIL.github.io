# EVOLVE-BLOCK-START
import time
import heapq
import numpy as np
from collections import deque

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def _serial_sgs(durations, demands, capacities, predecessors, priority_vec):
    n = len(durations)
    if n == 0:
        return 0.0
    remaining_preds = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for j in range(n):
        pred_list = predecessors[j]
        if pred_list:
            remaining_preds[j] = len(pred_list)
            for p in pred_list:
                successors[p].append(j)
    heap = []
    for j in range(n):
        if remaining_preds[j] == 0:
            heapq.heappush(heap, (-float(priority_vec[j]), j))
    start = np.zeros(n)
    finish = np.zeros(n)
    res_free = np.zeros(len(capacities))
    scheduled = 0
    while heap:
        _, j = heapq.heappop(heap)
        ps = 0.0
        for p in predecessors[j]:
            if finish[p] > ps:
                ps = finish[p]
        rs = 0.0
        for k in range(len(capacities)):
            if demands[j][k] > 0 and res_free[k] > rs:
                rs = res_free[k]
        st = ps if ps > rs else rs
        start[j] = st
        finish[j] = st + durations[j]
        for k in range(len(capacities)):
            if demands[j][k] > 0 and finish[j] > res_free[k]:
                res_free[k] = finish[j]
        scheduled += 1
        for v in successors[j]:
            remaining_preds[v] -= 1
            if remaining_preds[v] == 0:
                heapq.heappush(heap, (-float(priority_vec[v]), v))
    if scheduled < n:
        return 1e9
    return float(np.max(finish))


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    if n == 0:
        return np.array([])

    successors = [[] for _ in range(n)]
    for j in range(n):
        if predecessors[j]:
            for p in predecessors[j]:
                successors[p].append(j)

    # Topological order
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        in_deg[j] = len(predecessors[j]) if predecessors[j] else 0
    q = deque([j for j in range(n) if in_deg[j] == 0])
    topo = []
    indeg2 = in_deg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                q.append(v)
    if len(topo) < n:
        for j in range(n):
            if j not in topo:
                topo.append(j)

    rev_topo = list(reversed(topo))

    # Longest path to end (critical tail)
    longest_path_to_end = np.zeros(n, dtype=float)
    for i in rev_topo:
        max_succ = 0.0
        for j in successors[i]:
            cand = durations[j] + longest_path_to_end[j]
            if cand > max_succ:
                max_succ = cand
        longest_path_to_end[i] = durations[i] + max_succ

    # Earliest start (forward pass)
    es = np.zeros(n)
    for u in topo:
        for v in successors[u]:
            cand = es[u] + durations[u]
            if cand > es[v]:
                es[v] = cand

    # Latest finish / latest start
    ms_ub = np.max(es + durations) if n > 0 else 0.0
    lf = np.full(n, ms_ub)
    for u in rev_topo:
        for v in successors[u]:
            cand = lf[v] - durations[v]
            if cand < lf[u]:
                lf[u] = cand
    ls = lf - durations

    # Slack
    slack = np.maximum(lf - es - durations, 0.0)

    # Resource pressure components
    n_res = len(capacities)
    max_ratio = np.zeros(n)
    sum_ratio = np.zeros(n)
    for j in range(n):
        mx = 0.0
        s = 0.0
        for k in range(n_res):
            if capacities[k] > 0:
                r = demands[j][k] / capacities[k]
                if r > mx:
                    mx = r
                s += r
        max_ratio[j] = mx
        sum_ratio[j] = s

    # Log-sum-exp soft-max of resource ratios (smooth surrogate for max)
    # This captures multi-resource pressure smoothly
    max_ratio_scale = np.max(max_ratio) if n_res > 0 else 1.0
    if max_ratio_scale > 0:
        # soft-max over resources for each activity
        log_sum_exp = np.zeros(n)
        for j in range(n):
            ratios = []
            for k in range(n_res):
                if capacities[k] > 0:
                    ratios.append(demands[j][k] / capacities[k])
            if ratios:
                ratios = np.array(ratios)
                m = np.max(ratios)
                if m > 0:
                    log_sum_exp[j] = m + np.log(np.sum(np.exp(ratios - m)))
                else:
                    log_sum_exp[j] = 0.0
        # Normalize: this is a smooth measure of total resource pressure
        soft_max_pressure = log_sum_exp
    else:
        soft_max_pressure = np.zeros(n)

    # Resource-weighted tail: longest path where each activity's contribution
    # is weighted by its resource demand intensity
    res_weighted_tail = np.zeros(n)
    for i in rev_topo:
        max_succ = 0.0
        for j in successors[i]:
            # Weight successor's tail by resource pressure
            weight = 1.0 + 0.3 * max_ratio[j]
            cand = durations[j] * weight + res_weighted_tail[j]
            if cand > max_succ:
                max_succ = cand
        res_weighted_tail[i] = durations[i] * (1.0 + 0.3 * max_ratio[i]) + max_succ

    succ_count = np.zeros(n)
    for j in range(n):
        succ_count[j] = len(successors[j])

    def _norm(x):
        m = np.max(x)
        if m > 0:
            return x / m
        return np.zeros_like(x)

    norm_cp = _norm(longest_path_to_end)
    norm_max_ratio = _norm(max_ratio)
    norm_sum_ratio = _norm(sum_ratio)
    norm_soft_max = _norm(soft_max_pressure)
    norm_res_tail = _norm(res_weighted_tail)
    norm_succ = _norm(succ_count)
    norm_slack_inv = _norm(1.0 / (1.0 + slack)) if np.any(slack > 0) else np.zeros(n)

    def _base_score():
        # Core: critical path length, boosted by resource pressure
        return norm_cp * (1.0 + 0.5 * norm_max_ratio + 0.3 * norm_soft_max)

    if _remaining_budget() < 30.0:
        return _base_score()

    best_score = _base_score()
    best_ms = _serial_sgs(durations, demands, capacities, predecessors, best_score)

    # Grid search over weight combinations
    # Weights: (w_max_ratio, w_soft_max, w_res_tail, w_sum_ratio)
    w_max_vals = [0.0, 0.3, 0.5, 0.7, 1.0]
    w_soft_vals = [0.0, 0.2, 0.4, 0.6]
    w_tail_vals = [0.0, 0.2, 0.4, 0.6]
    w_sum_vals = [0.0, 0.2, 0.4]

    base = norm_cp
    evaluated = 0
    max_evals = 250
    for wm in w_max_vals:
        if _remaining_budget() < 30.0:
            break
        for ws in w_soft_vals:
            if _remaining_budget() < 30.0:
                break
            for wt in w_tail_vals:
                if _remaining_budget() < 30.0 or evaluated >= max_evals:
                    break
                for wsum in w_sum_vals:
                    if _remaining_budget() < 30.0 or evaluated >= max_evals:
                        break
                    cand = base * (1.0 + wm * norm_max_ratio + ws * norm_soft_max
                                   + wt * norm_res_tail + wsum * norm_sum_ratio)
                    ms = _serial_sgs(durations, demands, capacities, predecessors, cand)
                    evaluated += 1
                    if ms < best_ms:
                        best_ms = ms
                        best_score = cand

    # Finer polish around best if budget remains
    if _remaining_budget() > 45.0:
        for wm2 in [0.4, 0.6, 0.8]:
            if _remaining_budget() < 30.0:
                break
            for ws2 in [0.3, 0.5, 0.7]:
                if _remaining_budget() < 30.0:
                    break
                for wt2 in [0.3, 0.5, 0.7]:
                    if _remaining_budget() < 30.0:
                        break
                    for wsum2 in [0.1, 0.3, 0.5]:
                        if _remaining_budget() < 30.0:
                            break
                        cand = base * (1.0 + wm2 * norm_max_ratio + ws2 * norm_soft_max
                                       + wt2 * norm_res_tail + wsum2 * norm_sum_ratio)
                        ms = _serial_sgs(durations, demands, capacities, predecessors, cand)
                        if ms < best_ms:
                            best_ms = ms
                            best_score = cand

    return best_score
# EVOLVE-BLOCK-END
