"""RCPSP priority via local search over priority rules with serial SGS evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy: Local search over priority rule combinations.
    1. Compute base features: backward slack, resource criticality, duration,
       predecessor/successor counts.
    2. Generate candidate priority vectors as weighted combinations of features.
    3. Evaluate each candidate via serial SGS simulation.
    4. Keep the best; perform local perturbation around it.
    """
    start_time = time.time()
    DEADLINE = 118.0  # safety margin before 120s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # --- Extract instance components ---
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred"]:
        if key in instance:
            preds = instance[key]
            break

    succs = None
    for key in ["successors", "succs", "successors_list", "succ"]:
        if key in instance:
            succs = instance[key]
            break

    demands = None
    for key in ["demands", "resource_demands", "resources", "dem", "resource"]:
        if key in instance:
            demands = instance[key]
            break

    capacities = None
    for key in ["capacities", "capacity", "resource_capacities", "caps", "cap"]:
        if key in instance:
            capacities = instance[key]
            break

    if preds is None:
        preds = [[] for _ in range(n)]
    if demands is None:
        demands = np.zeros((n, 1))
    if capacities is None:
        capacities = [1.0]

    demands = np.asarray(demands, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
    R = demands.shape[1]
    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim == 0:
        capacities = np.full(R, float(capacities))

    # Build successor lists if not provided
    if succs is None:
        succs = [set() for _ in range(n)]
        for j in range(n):
            for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
                idx = int(i)
                if 0 <= idx < n:
                    succs[idx].add(j)
        succs = [list(s) for s in succs]

    # --- Topological order ---
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
            idx = int(i)
            if 0 <= idx < n:
                in_degree[j] += 1

    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)

    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    if len(topo_order) < n:
        for i in range(n):
            if i not in set(topo_order):
                topo_order.append(i)

    # --- Forward pass ---
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        for v in succs[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
        EF[u] = ES[u] + durations[u]

    makespan_forward = float(np.max(EF)) if n > 0 else 0.0

    # --- Backward pass ---
    LS = np.zeros(n)
    LF = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) == 0:
            LF[u] = makespan_forward
        else:
            LF[u] = float(np.min([LS[v] for v in succs[u]]))
        LS[u] = LF[u] - durations[u]
        for v in succs[u]:
            if LS[u] < LF[v]:
                LF[v] = LS[u]

    slack = LS - ES

    # --- Feature computation ---
    # Resource tightness
    resource_tightness = np.zeros(R)
    for r in range(R):
        total_demand = float(np.sum(demands[:, r]))
        if total_demand > 0:
            resource_tightness[r] = capacities[r] / total_demand
        else:
            resource_tightness[r] = 1.0

    max_demand_per_res = np.max(demands, axis=0)
    max_demand_per_res[max_demand_per_res == 0] = 1.0
    norm_demands = demands / max_demand_per_res
    inv_tightness = 1.0 / (resource_tightness + 1e-10)
    resource_criticality = np.sum(norm_demands * inv_tightness, axis=1)

    # Predecessor count (depth in network)
    pred_count = np.zeros(n)
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            pred_count[j] = len(p)

    # Successor count
    succ_count = np.zeros(n)
    for j in range(n):
        succ_count[j] = len(succs[j])

    # Duration
    dur = durations.copy()

    # Normalize features to [0,1]
    def normalize(x):
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx > mn:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)

    feat_slack = normalize(slack)  # high slack = low urgency
    feat_rc = normalize(resource_criticality)
    feat_dur = normalize(dur)
    feat_pred = normalize(pred_count)
    feat_succ = normalize(succ_count)
    feat_es = normalize(ES)  # earlier ES = more urgent

    # Feature matrix: each column is a feature, rows are activities
    # We want HIGH priority = should be scheduled early
    # So: low slack -> high priority, high RC -> high priority, high dur -> high priority
    features = np.column_stack([
        1.0 - feat_slack,  # inverse slack: low slack = high value
        feat_rc,
        feat_dur,
        feat_pred,
        feat_succ,
        1.0 - feat_es,     # earlier ES = higher urgency
    ])

    # --- Serial SGS simulator (optimized) ---
    def simulate_sgs(prio_scores):
        """
        Serial SGS: repeatedly pick the highest-priority eligible activity
        and schedule it at the earliest feasible time.
        Returns (start_times, makespan) or (None, None) on timeout.
        """
        if time.time() - start_time > DEADLINE:
            return None, None

        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)

        # Precompute predecessor sets
        pred_set = [set() for _ in range(n)]
        for j in range(n):
            p = preds[j]
            if isinstance(p, (list, tuple, set, np.ndarray)):
                for i in p:
                    idx = int(i)
                    if 0 <= idx < n:
                        pred_set[j].add(idx)

        remaining = set(range(n))
        # Track resource usage over time using a list of (start, end, activity)
        scheduled = []

        while remaining:
            if time.time() - start_time > DEADLINE:
                return None, None

            # Find eligible: all predecessors scheduled
            eligible = []
            for j in remaining:
                if all(start_t[p] < np.inf for p in pred_set[j]):
                    eligible.append(j)

            if not eligible:
                # Fallback
                for j in remaining:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break

            # Pick highest priority
            best = max(eligible, key=lambda x: prio_scores[x])

            # Earliest start from precedence
            es_best = 0.0
            for p in pred_set[best]:
                cand = start_t[p] + durations[p]
                if cand > es_best:
                    es_best = cand

            # Find earliest feasible start considering resources
            # Candidate times: es_best and finish times of scheduled activities
            cand_times = [es_best]
            for k in range(n):
                if finish_t[k] < np.inf and finish_t[k] >= es_best:
                    cand_times.append(finish_t[k])
            cand_times = sorted(set(cand_times))

            best_start = es_best
            found = False
            for t in cand_times:
                if t < es_best:
                    continue
                end_t = t + durations[best]
                
                # Check resource feasibility in [t, end_t)
                # Collect all critical time points in [t, end_t)
                check_pts = set([t, end_t])
                for k in range(n):
                    if start_t[k] < np.inf:
                        if t <= start_t[k] < end_t:
                            check_pts.add(start_t[k])
                        if t < finish_t[k] <= end_t:
                            check_pts.add(finish_t[k])
                check_pts = sorted(check_pts)

                feasible = True
                for i in range(len(check_pts) - 1):
                    tp = check_pts[i]
                    tp_next = check_pts[i + 1]
                    if tp >= end_t or tp_next <= t:
                        continue
                    # Sum demands of activities active in [tp, tp_next)
                    active = np.zeros(R)
                    for k in range(n):
                        if start_t[k] < np.inf:
                            if start_t[k] <= tp and finish_t[k] >= tp_next:
                                active += demands[k]
                    active += demands[best]
                    if np.any(active > capacities + 1e-9):
                        feasible = False
                        break
                if feasible:
                    best_start = t
                    found = True
                    break

            if not found:
                best_start = es_best

            start_t[best] = best_start
            finish_t[best] = best_start + durations[best]
            remaining.discard(best)
            scheduled.append(best)

        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return start_t, makespan

    # --- Local search over weight vectors ---
    n_features = features.shape[1]

    def evaluate_weights(w):
        """Compute priority scores from weights, simulate, return makespan."""
        scores = features @ w  # shape (n,)
        # Add small random tie-breaker to avoid ties
        scores = scores + np.random.uniform(0, 1e-6, n)
        _, ms = simulate_sgs(scores)
        return ms if ms is not None else float('inf')

    # Initial candidate: equal weights
    best_w = np.ones(n_features) / n_features
    best_ms = evaluate_weights(best_w)
    best_scores = features @ best_w + np.random.uniform(0, 1e-6, n)

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_scores

    # Grid search over a small set of weight combinations
    # Focus on the most impactful features: inverse_slack, resource_criticality, duration
    # Try various weight combinations
    candidate_weights = []

    # Equal weights
    candidate_weights.append(np.ones(n_features) / n_features)

    # Emphasize slack
    candidate_weights.append(np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.1]))
    # Emphasize resource criticality
    candidate_weights.append(np.array([0.2, 0.5, 0.1, 0.05, 0.05, 0.1]))
    # Emphasize duration
    candidate_weights.append(np.array([0.2, 0.2, 0.4, 0.05, 0.05, 0.1]))
    # Balanced
    candidate_weights.append(np.array([0.3, 0.3, 0.2, 0.05, 0.05, 0.1]))
    # Slack + RC only
    candidate_weights.append(np.array([0.5, 0.5, 0, 0, 0, 0]))
    # Slack + duration
    candidate_weights.append(np.array([0.5, 0, 0.5, 0, 0, 0]))
    # All features moderate
    candidate_weights.append(np.array([0.25, 0.25, 0.25, 0.05, 0.05, 0.15]))

    # Evaluate each candidate
    for w in candidate_weights:
        if time.time() - start_time > DEADLINE:
            break
        ms = evaluate_weights(w)
        if ms < best_ms:
            best_ms = ms
            best_w = w.copy()
            best_scores = features @ best_w + np.random.uniform(0, 1e-6, n)

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_scores

    # Local perturbation around best
    n_perturbations = 20
    for _ in range(n_perturbations):
        if time.time() - start_time > DEADLINE:
            break
        # Perturb weights
        w_pert = best_w + np.random.normal(0, 0.05, n_features)
        w_pert = np.clip(w_pert, 0, 1)
        s = float(np.sum(w_pert))
        if s > 0:
            w_pert = w_pert / s
        ms = evaluate_weights(w_pert)
        if ms < best_ms:
            best_ms = ms
            best_w = w_pert.copy()
            best_scores = features @ best_w + np.random.uniform(0, 1e-6, n)

    return best_scores
# EVOLVE-BLOCK-END
