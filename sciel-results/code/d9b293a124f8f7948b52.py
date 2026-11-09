# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from collections import deque

def _schedule_makespan(durations, demands, capacities, preds, priority_scores,
                       successors=None, in_degree=None, res_idx=None):
    """Fast serial schedule generation scheme. Returns the makespan.

    Resource feasibility is checked at exact event points (start/end times of
    already-scheduled activities) rather than integer-stepping, which is both
    faster and exact for the standard SSAS.
    """
    n = len(durations)
    n_res = len(capacities)
    if successors is None or in_degree is None:
        in_degree = np.zeros(n, dtype=int)
        successors = [[] for _ in range(n)]
        for i in range(n):
            for p in preds[i]:
                successors[p].append(i)
                in_degree[i] += 1
    if res_idx is None:
        res_idx = []
        for i in range(n):
            res_idx.append([j for j in range(n_res) if demands[i][j] > 0])

    remaining = in_degree.copy()
    sched_start = np.zeros(n)
    sched_end = np.zeros(n)
    scheduled = [False] * n
    scheduled_count = 0
    active = []  # list of scheduled activity indices

    while scheduled_count < n:
        eligible = [i for i in range(n) if not scheduled[i] and remaining[i] == 0]
        if not eligible:
            break
        eligible.sort(key=lambda i: priority_scores[i], reverse=True)
        best = eligible[0]

        # Earliest start from precedence
        es = 0.0
        for p in preds[best]:
            if sched_end[p] > es:
                es = sched_end[p]
        start = es

        # Resource feasibility: try candidate times in increasing order
        # Candidate times: es, then start times of active activities >= es,
        # then end times of active activities >= es.
        if res_idx[best]:
            # Collect candidate times
            candidates = [es]
            for k in active:
                if sched_start[k] >= es - 1e-9:
                    candidates.append(sched_start[k])
                if sched_end[k] >= es - 1e-9:
                    candidates.append(sched_end[k])
            # Deduplicate and sort
            candidates = sorted(set(candidates))

            found = False
            for t in candidates:
                if t < es - 1e-9:
                    continue
                ok = True
                for j in res_idx[best]:
                    used = 0.0
                    for k in active:
                        if sched_start[k] < t and sched_end[k] > t:
                            used += demands[k][j]
                    if used + demands[best][j] > capacities[j] + 1e-9:
                        ok = False
                        break
                if ok:
                    start = t
                    found = True
                    break
            if not found:
                # Fallback: use the last candidate (shouldn't happen for valid instances)
                start = candidates[-1] if candidates else es
        # else: no resource demand, start at es

        sched_start[best] = start
        sched_end[best] = start + durations[best]
        scheduled[best] = True
        active.append(best)
        scheduled_count += 1
        for v in successors[best]:
            remaining[v] -= 1

    return float(max(sched_end))


def _compute_features(instance):
    """Compute a rich set of heuristic features for each activity."""
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    n_res = len(capacities)

    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological order
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    topo_order = []
    rem = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            rem[v] -= 1
            if rem[v] == 0:
                queue.append(v)
    if len(topo_order) != n:
        topo_order = list(range(n))
    reverse_topo = list(reversed(topo_order))

    # Hard CPL
    cpl = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            cpl[u] = durations[u]
        else:
            cpl[u] = durations[u] + max(cpl[v] for v in successors[u])

    # Smooth CPL (log-sum-exp)
    T = 0.5
    scpl = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            scpl[u] = durations[u]
        else:
            sv = [scpl[v] for v in successors[u]]
            m = max(sv)
            se = sum(math.exp((x - m) / T) for x in sv)
            scpl[u] = durations[u] + m + T * math.log(se)

    # Earliest start
    es = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            es[u] = 0.0
        else:
            es[u] = max(es[p] + durations[p] for p in preds[u])
    project_length = max(es[i] + durations[i] for i in range(n))

    # Slack
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = (es[i] + durations[i] + cpl[i]) - project_length

    # Resource features
    res_sum = np.zeros(n, dtype=float)
    res_max = np.zeros(n, dtype=float)
    res_prod = np.zeros(n, dtype=float)
    res_bottleneck = np.zeros(n, dtype=float)
    res_contention = np.zeros(n, dtype=float)  # NEW: duration-weighted bottleneck
    for i in range(n):
        s = 0.0
        mx = 0.0
        pr = 1.0
        bn = 0.0
        ct = 0.0
        for j in range(n_res):
            r = demands[i][j] / capacities[j] if capacities[j] > 0 else 0.0
            s += r
            if r > mx:
                mx = r
            pr *= (1.0 + r)
            bn += r ** 1.5
            ct += r * durations[i]  # duration-weighted resource pressure
        res_sum[i] = s
        res_max[i] = mx
        res_prod[i] = pr
        res_bottleneck[i] = bn
        res_contention[i] = ct

    # Depth, pred/succ counts
    pred_count = np.array([len(preds[i]) for i in range(n)], dtype=float)
    depth = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            depth[u] = 0.0
        else:
            depth[u] = max(depth[p] for p in preds[u]) + 1.0
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Resource-weighted tail
    res_tail = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            res_tail[u] = res_max[u]
        else:
            res_tail[u] = res_max[u] + max(res_tail[v] for v in successors[u])

    # Smooth resource tail
    T2 = 0.3
    res_tail_smooth = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            res_tail_smooth[u] = res_max[u]
        else:
            sv = [res_tail_smooth[v] for v in successors[u]]
            m = max(sv)
            se = sum(math.exp((x - m) / T2) for x in sv)
            res_tail_smooth[u] = res_max[u] + m + T2 * math.log(se)

    # Contention-weighted tail (NEW)
    res_cont_tail = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            res_cont_tail[u] = res_contention[u]
        else:
            res_cont_tail[u] = res_contention[u] + max(res_cont_tail[v] for v in successors[u])

    # Normalize
    def norm(arr):
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    feats = {
        'cpl': norm(cpl),
        'scpl': norm(scpl),
        'slack_crit': 1.0 - norm(slack),
        'res_sum': norm(res_sum),
        'res_max': norm(res_max),
        'res_prod': norm(res_prod),
        'res_bottleneck': norm(res_bottleneck),
        'res_contention': norm(res_contention),
        'res_tail': norm(res_tail),
        'res_tail_smooth': norm(res_tail_smooth),
        'res_cont_tail': norm(res_cont_tail),
        'dur': norm(durations),
        'depth': norm(depth),
        'pred_count': norm(pred_count),
        'succ_count': norm(succ_count),
    }
    return feats, topo_order, successors


def priority(instance):
    start_time = time.time()
    DEADLINE = 115.0  # safety margin under 120s

    feats, topo_order, successors = _compute_features(instance)
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    n_res = len(capacities)

    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1

    res_idx = []
    for i in range(n):
        res_idx.append([j for j in range(n_res) if demands[i][j] > 0])

    names = list(feats.keys())
    X = np.column_stack([feats[k] for k in names])
    n_feat = len(names)

    # Baseline weights
    base = np.zeros(n_feat)
    base[names.index('cpl')] = 1.0
    base[names.index('scpl')] = 0.3
    base[names.index('res_sum')] = 0.3
    base[names.index('res_bottleneck')] = 0.4
    base[names.index('res_contention')] = 0.5
    base[names.index('res_tail')] = 0.5
    base[names.index('res_tail_smooth')] = 0.4
    base[names.index('res_cont_tail')] = 0.3
    base[names.index('slack_crit')] = 0.2
    base_weights = base
    base_scores = X @ base_weights

    def eval_w(w):
        scores = X @ w
        try:
            return _schedule_makespan(durations, demands, capacities, preds, scores,
                                      successors=successors, in_degree=in_degree,
                                      res_idx=res_idx)
        except Exception:
            return float('inf')

    best_makespan = eval_w(base_weights)
    best_w = base_weights.copy()
    best_scores = base_scores.copy()

    # Coordinate descent with adaptive step
    step = 0.5
    no_improve_rounds = 0
    while step >= 0.03 and time.time() - start_time < DEADLINE - 10.0:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for k in range(n_feat):
            if time.time() - start_time > DEADLINE:
                break
            for delta in (step, -step, 0.5 * step, -0.5 * step):
                w = best_w.copy()
                w[k] += delta
                if (w < -0.5).any() or (w > 3.0).any():
                    continue
                m = eval_w(w)
                if m < best_makespan - 1e-9:
                    best_makespan = m
                    best_w = w.copy()
                    best_scores = X @ w
                    improved = True
                    break
        if not improved:
            no_improve_rounds += 1
            if no_improve_rounds >= 2:
                break
            step *= 0.7
        else:
            no_improve_rounds = 0

    # Random restarts
    rng = np.random.default_rng(42)
    while time.time() - start_time < DEADLINE - 5.0:
        w = base_weights.copy()
        w += rng.normal(0, 0.4, size=n_feat)
        w = np.clip(w, -0.5, 3.0)
        m = eval_w(w)
        if m < best_makespan - 1e-9:
            best_makespan = m
            best_w = w.copy()
            best_scores = X @ w

    # Final polish: tight local search around best
    if time.time() - start_time < DEADLINE - 2.0:
        rng2 = np.random.default_rng(7)
        for _ in range(200):
            if time.time() - start_time > DEADLINE:
                break
            w = best_w + rng2.normal(0, 0.05, size=n_feat)
            w = np.clip(w, -0.5, 3.0)
            m = eval_w(w)
            if m < best_makespan - 1e-9:
                best_makespan = m
                best_w = w.copy()
                best_scores = X @ w

    return best_scores
# EVOLVE-BLOCK-END
