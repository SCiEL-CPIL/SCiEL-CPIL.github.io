"""Priority function for resource-constrained project scheduling (serial SGS)."""
import time
import numpy as np

_DEADLINE = 140.0  # safety margin before 150s wall-clock kill
_T0 = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _T0)


def priority(instance):
    """Return one score per activity; higher score = scheduled earlier in serial SGS."""
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance["demands"]
    capacities = instance["capacities"]
    preds = instance["predecessors"]

    # Build successor lists and in/out degree info
    succ = [list() for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            indeg[i] += 1

    # --- Backward pass (ignoring resources) to get critical-path slack ---
    # Latest finish/finish from end: LF of sink = max duration-based end
    # Use longest path from each node to any sink.
    # Topological order via Kahn (reverse for backward pass)
    outdeg = np.array([len(succ[i]) for i in range(n)], dtype=int)
    # Forward topological order
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    topo = []
    indeg_work = indeg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                q.append(v)
    # In case of cycles, append remaining
    if len(topo) < n:
        for i in range(n):
            if i not in set(topo):
                topo.append(i)

    # Longest path from node i to any sink (forward durations)
    longest_to_sink = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if outdeg[u] == 0:
            longest_to_sink[u] = durations[u]
        else:
            longest_to_sink[u] = durations[u] + max(longest_to_sink[v] for v in succ[u])

    # Critical path length
    if n > 0:
        cp_len = float(max(longest_to_sink)) if n else 1.0
    else:
        cp_len = 1.0

    # Slack of node i = cp_len - (earliest_start + longest_to_sink)
    # earliest start ignoring resources
    est = np.zeros(n, dtype=float)
    for u in topo:
        if indeg[u] == 0:
            est[u] = 0.0
        else:
            est[u] = max(est[p] + durations[p] for p in preds[u])
    slack = cp_len - (est + longest_to_sink)
    slack = np.maximum(slack, 1e-9)

    # --- Resource contention metric ---
    # For each activity, measure how much it blocks resources that many successors need.
    # contention[i] = sum over resources r used by i of (demand[i,r]/capacity[r]) * (num successors of i that also use r)
    n_res = len(capacities)
    cap = np.asarray(capacities, dtype=float)
    cap_safe = np.where(cap > 0, cap, 1.0)

    # Precompute which resources each activity uses (demand > 0)
    # demands shape: (n, n_res)
    dem = np.asarray(demands, dtype=float)
    uses = dem > 0  # (n, n_res) bool

    # For each resource r, count how many successors of i use r -> we want per-node
    # Build for each resource a set of nodes that use it
    contention = np.zeros(n, dtype=float)
    # For efficiency: for each node i, sum over r where uses[i,r]
    # of (dem[i,r]/cap_safe[r]) * (count of j in succ[i] with uses[j,r])
    # Precompute per-resource usage indicator matrix is uses already.
    # For each node i, successors succ[i]; we need for each r in uses[i], count of succ using r.
    # Use matrix: succ_mask[i] is bool vector over nodes. Then for resource r, count = sum over succ of uses[succ, r].
    # Do it node by node but vectorize over resources.
    for i in range(n):
        s = succ[i]
        if not s:
            continue
        # for resources i uses:
        r_idx = np.where(uses[i])[0]
        if r_idx.size == 0:
            continue
        # count of successors using each of these resources
        # uses[s, r_idx] is (|s|, |r_idx|); sum over axis 0
        counts = uses[np.ix_(s, r_idx)].sum(axis=0)  # (|r_idx|,)
        weights = dem[i, r_idx] / cap_safe[r_idx]
        contention[i] = float(np.sum(weights * counts))

    # --- Number of successors (unlocking power) ---
    n_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)

    # --- Normalize components ---
    def _norm(x):
        x = np.asarray(x, dtype=float)
        rng = x.max() - x.min()
        if rng <= 1e-12:
            return np.zeros_like(x)
        return (x - x.min()) / rng

    # Critical path component: low slack = high priority. Use 1/ (slack) style.
    # Use inverse slack normalized.
    inv_slack = 1.0 / slack
    cp_score = _norm(inv_slack)

    # Duration component
    dur_score = _norm(durations)

    # Contention component
    cont_score = _norm(contention)

    # Successor count component
    succ_score = _norm(n_succ)

    # Longest-to-sink (tail length) component
    tail_score = _norm(longest_to_sink)

    # Combine with weights tuned to emphasize critical path + contention
    w_cp = 0.30
    w_cont = 0.30
    w_dur = 0.15
    w_succ = 0.10
    w_tail = 0.15

    score = (w_cp * cp_score +
             w_cont * cont_score +
             w_dur * dur_score +
             w_succ * succ_score +
             w_tail * tail_score)

    # Add a small tie-breaker: prefer higher tail (longer remaining path)
    # Already included. Return as float array.
    return np.asarray(score, dtype=float)
