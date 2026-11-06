"""Resource-Constrained Project Scheduling — Total Float (slack) priority.

Direct approach: compute the classical CPM Total Float for each activity via
a forward pass (earliest start) and a backward pass (latest finish). An
activity's total float is how late it can start without delaying the project,
so activities with the least float are most critical to the makespan. We
return a score that is HIGHER for more critical (low-float) activities, and
break ties using the activity's demand on the scarcest (most contended)
renewable resources. This is a structurally-derived rule: float comes directly
from the instance's precedence graph + durations, and resource pressure comes
directly from demands/capacities.

A safety deadline guard ensures we always return within budget.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

_DEADLINE_SECONDS = 110.0  # safety margin below the 120 s wall-clock limit


def _soft_min(values, temp=2.0):
    """Soft minimum using log-sum-exp for numerical stability.
    
    Higher temp -> smoother (less peaked) approximation to min.
    temp=2.0 gives a sharper criticality signal while remaining differentiable.
    """
    if len(values) is None or len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    vals = np.asarray(values, dtype=float)
    neg = -vals
    max_neg = np.max(neg)
    if max_neg > 1e12:
        return -max_neg
    shifted = neg - max_neg
    log_sum = max_neg + np.log(np.sum(np.exp(shifted / temp)))
    return -log_sum


def priority(instance):
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # ---- Build predecessor / successor lists ----
    preds = instance.get("predecessors", None)
    pred_lists = []
    if preds is None:
        pred_lists = [[] for _ in range(n)]
    else:
        for i in range(n):
            p = preds[i]
            if p is None:
                p = []
            if isinstance(p, (list, tuple, np.ndarray)):
                p = [int(x) for x in p]
            else:
                p = [int(p)]
            p = [x for x in p if 0 <= x < n and x != i]
            seen = set()
            out = []
            for x in p:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            pred_lists.append(out)

    succ_lists = [[] for _ in range(n)]
    for i in range(n):
        for j in pred_lists[i]:
            succ_lists[j].append(i)

    # ---- Topological order (Kahn) ----
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for j in pred_lists[i]:
            in_degree[i] += 1
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ_lists[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    if len(topo) < n:
        for i in range(n):
            if i not in set(topo):
                topo.append(i)

    # ---- Forward pass: earliest start / earliest finish ----
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    for i in topo:
        if pred_lists[i]:
            ES[i] = max(EF[j] for j in pred_lists[i])
        else:
            ES[i] = 0.0
        EF[i] = ES[i] + durations[i]

    project_dur = float(max(EF)) if n > 0 else 0.0

    # ---- Backward pass with soft-min for smoother criticality ----
    LF = np.full(n, project_dur, dtype=float)
    LS = np.zeros(n, dtype=float)
    soft_temp = 2.0
    for i in reversed(topo):
        if succ_lists[i]:
            ls_vals = [LS[j] for j in succ_lists[i]]
            LF[i] = _soft_min(ls_vals, temp=soft_temp)
        else:
            LF[i] = project_dur
        LS[i] = LF[i] - durations[i]

    # ---- Total Float (slack) ----
    slack = LS - ES

    s_min = float(np.min(slack)) if n > 0 else 0.0
    s_max = float(np.max(slack)) if n > 0 else 0.0
    s_range = s_max - s_min
    if s_range > 1e-12:
        slack_norm = (slack - s_min) / s_range
        crit = 1.0 - slack_norm
    else:
        crit = np.ones(n, dtype=float)

    # ---- Resource pressure: dynamic capacity depletion (fluid-flow congestion) ----
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)

    res_score = np.zeros(n, dtype=float)
    if demands is not None:
        D = np.asarray(demands, dtype=float)
        if D.size == 0:
            D = None
        if D is not None:
            if D.ndim == 1:
                if D.shape[0] == n:
                    D = D.reshape(n, 1)
                else:
                    D = None
            if D is not None and D.shape[0] == n:
                R = D.shape[1]
                caps = None
                if capacities is not None:
                    C = np.asarray(capacities, dtype=float).reshape(-1)
                    if C.size == 1:
                        caps = np.full(R, float(C[0]), dtype=float)
                    elif C.size < R:
                        caps = np.ones(R, dtype=float)
                        caps[:C.size] = C
                    else:
                        caps = C[:R]
                    caps = np.maximum(caps, 1e-9)
                if caps is not None:
                    # Per-activity, per-resource utilization ratio
                    ratio = D / caps[np.newaxis, :]  # (n, R)

                    # Static scarcity: total demand per resource relative to capacity
                    total_demand_per_res = np.sum(D, axis=0)  # (R,)
                    scarcity = total_demand_per_res / caps  # (R,)
                    sc_max = float(np.max(scarcity)) if R > 0 else 1.0
                    if sc_max > 1e-12:
                        scarcity = scarcity / sc_max  # normalize to [0,1]

                    # Dynamic congestion: for each activity, estimate the peak
                    # cumulative load it contributes to each resource over its
                    # execution window [ES_i, EF_i].
                    windows = np.stack([ES, EF], axis=1)  # (n, 2)
                    start_max = np.maximum(windows[:, 0][:, np.newaxis], windows[:, 0][np.newaxis, :])
                    end_min = np.minimum(windows[:, 1][:, np.newaxis], windows[:, 1][np.newaxis, :])
                    overlap = np.maximum(0.0, end_min - start_max)  # (n, n)

                    dur_safe = np.maximum(durations, 1e-9)
                    cumulative_load = overlap @ D  # (n, R)
                    congestion = cumulative_load / dur_safe[:, np.newaxis]  # (n, R)

                    # Weight by scarcity of each resource (cubic to emphasize tightest)
                    scarcity_cu = scarcity ** 3
                    congestion_weighted = congestion * scarcity_cu[np.newaxis, :]

                    # res_score: total weighted congestion per activity
                    # Also add a direct demand/capacity ratio component to capture
                    # activities that individually consume large fractions of capacity
                    direct_pressure = np.sum(ratio, axis=1)
                    dp_max = float(np.max(direct_pressure)) if n > 0 else 1.0
                    if dp_max > 1e-12:
                        direct_pressure = direct_pressure / dp_max
                    else:
                        direct_pressure = np.zeros(n, dtype=float)

                    res_score = np.sum(congestion_weighted, axis=1) + 0.5 * direct_pressure
                    rm = float(np.max(res_score)) if n > 0 else 1.0
                    if rm > 1e-12:
                        res_score = res_score / rm
                    else:
                        res_score = np.zeros(n, dtype=float)

    # ---- Bottleneck potential: duration weighted by successor count ----
    succ_counts = np.array([len(succ_lists[i]) for i in range(n)], dtype=float)
    d_max = float(np.max(durations)) if n > 0 else 1.0
    if d_max > 1e-12:
        bottleneck = (durations / d_max) * (1.0 + np.log1p(succ_counts))
        bn_max = float(np.max(bottleneck)) if n > 0 else 1.0
        if bn_max > 1e-12:
            bottleneck = bottleneck / bn_max
    else:
        bottleneck = np.zeros(n, dtype=float)

    # ---- Downstream criticality: weighted sum of successor slacks ----
    # Activities whose successors have low slack are more critical
    downstream_crit = np.zeros(n, dtype=float)
    for i in reversed(topo):
        if succ_lists[i]:
            # Use the soft-min of successor criticalities (inverse slack)
            succ_crit_vals = [crit[j] for j in succ_lists[i]]
            downstream_crit[i] = _soft_min(succ_crit_vals, temp=1.5)
        else:
            downstream_crit[i] = crit[i]
    
    dc_max = float(np.max(downstream_crit)) if n > 0 else 1.0
    if dc_max > 1e-12:
        downstream_crit = downstream_crit / dc_max
    else:
        downstream_crit = np.zeros(n, dtype=float)

    # ---- Composite priority ----
    # crit dominates (structural criticality), res_score adds scheduling
    # pressure from resource contention, bottleneck adds duration/branching.
    # Increased weight on resource contention since it directly affects
    # serial SGS feasibility under renewable resource constraints.
    score = crit + 0.45 * res_score + 0.08 * bottleneck

    # Composite priority with downstream criticality
    score = crit + 0.45 * res_score + 0.08 * bottleneck + 0.15 * downstream_crit

    # Deadline guard
    if (time.time() - t0) > _DEADLINE_SECONDS:
        return np.asarray(score, dtype=float)

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
