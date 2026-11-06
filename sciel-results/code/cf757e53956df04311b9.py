"""Improved program for RCPSP using combined heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time


def priority(instance):
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    if n == 0:
        return np.array([], dtype=float)

    # Get predecessors with defensive key handling
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred", "precedence"]:
        if key in instance:
            preds = instance[key]
            break

    if preds is None:
        # No precedence constraints: return duration-based priority (longer first)
        return np.asarray(durations, dtype=float)

    # Normalize preds to list of lists of ints
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()

    # Build successor list
    succs = [set() for _ in range(n)]
    for j in range(n):
        if j < len(preds):
            p_list = preds[j]
            if p_list is None:
                continue
            for i in p_list:
                if i is not None:
                    i = int(i)
                    if 0 <= i < n and i != j:
                        succs[i].add(j)

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return np.asarray(durations, dtype=float)

    # Topological sort
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in succs[j]:
            in_degree[i] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)

    # If not all nodes in topo_order, there's a cycle; add remaining
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    # Forward pass: ES
    ES = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succs[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]

    project_duration = np.max(ES + durations) if n > 0 else 0

    # Backward pass: LS
    LS = np.full(n, project_duration, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if LS[v] - durations[u] < LS[u]:
                LS[u] = LS[v] - durations[u]

    total_float = LS - ES

    # Resource scarcity
    demands = None
    capacities = None
    for key in ["demands", "resource_demands", "demand", "resource_demand"]:
        if key in instance:
            demands = instance[key]
            break
    for key in ["capacities", "resource_capacities", "capacity", "resource_capacity"]:
        if key in instance:
            capacities = instance[key]
            break

    scarcity = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
        if capacities.ndim == 0:
            capacities = capacities.reshape(1)
        if demands.shape[0] == n and capacities.shape[0] > 0:
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            scarcity = np.sum(demands / safe_cap, axis=1)

    # Successor count
    succ_count = np.array([len(s) for s in succs], dtype=float)

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return np.asarray(durations, dtype=float)

    # Normalize helper
    def normalize(x):
        if len(x) == 0:
            return x
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    tf_norm = normalize(-total_float)
    sc_norm = normalize(scarcity)
    succ_norm = normalize(succ_count)

    # Also compute a duration-based component for robustness
    dur_norm = normalize(durations)

    # Weighted combination - emphasize total float (critical path) heavily,
    # then resource scarcity, then successor count, with small duration tiebreaker
    w_tf = 0.60
    w_sc = 0.20
    w_succ = 0.10
    w_dur = 0.10

    priority_scores = (w_tf * tf_norm + w_sc * sc_norm +
                       w_succ * succ_norm + w_dur * dur_norm)

    return priority_scores
# EVOLVE-BLOCK-END
