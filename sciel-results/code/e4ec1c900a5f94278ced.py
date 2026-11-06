"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

_DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 150s wall-clock kill
_START_TIME = time.perf_counter()


def _elapsed():
    return time.perf_counter() - _START_TIME


def _check_deadline():
    """Return True if we are too close to the deadline and should stop early."""
    return _elapsed() > (150.0 - _DEADLINE_MARGIN)


def _topo_order(n, preds):
    """Return a topological order (list of task indices) given predecessors as list of lists."""
    in_deg = np.zeros(n, dtype=int)
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            in_deg[i] += 1
    q = deque([i for i in range(n) if in_deg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)
    if len(order) < n:
        seen = set(order)
        for i in range(n):
            if i not in seen:
                order.append(i)
    return order, succs


def _longest_path_from_sources(n, preds, times, topo, succs):
    """Compute longest path from any source to each node (earliest start time)."""
    dist = np.zeros(n, dtype=float)
    for u in topo:
        du = dist[u] + times[u]
        for v in succs[u]:
            if du > dist[v]:
                dist[v] = du
    return dist


def _longest_path_to_sinks(n, preds, times, topo, succs):
    """Compute longest path from each node to any sink (latest finish constraint)."""
    dist = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in succs[u]:
            if dist[v] > best:
                best = dist[v]
        dist[u] = times[u] + best
    return dist


def _transitive_successor_info(n, preds, times, succs):
    """Count transitive successors and compute weighted downstream time for each task."""
    succ_counts = np.zeros(n, dtype=int)
    downstream_time = np.zeros(n, dtype=float)
    for i in range(n):
        visited = [False] * n
        stack = list(succs[i])
        total = 0.0
        count = 0
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                total += times[u]
                count += 1
                stack.extend(succs[u])
        succ_counts[i] = count
        downstream_time[i] = total
    return succ_counts, downstream_time


def _transitive_predecessor_count(n, preds, succs):
    """Count number of transitive predecessors for each task."""
    pred_list = [list(preds[i]) for i in range(n)]
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        visited = [False] * n
        stack = list(pred_list[i])
        count = 0
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                count += 1
                stack.extend(pred_list[u])
        counts[i] = count
    return counts


def priority(instance):
    global _START_TIME
    _START_TIME = time.perf_counter()

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = float(instance["cycle_time"])
    n = len(times)

    if n == 0:
        return np.array([], dtype=float)
    if _check_deadline():
        return times.copy()

    # Build topological order and successor lists
    topo, succs = _topo_order(n, preds)

    # Compute positional metrics
    est = _longest_path_from_sources(n, preds, times, topo, succs)
    lft = _longest_path_to_sinks(n, preds, times, topo, succs)

    if _check_deadline():
        return times.copy()

    # Transitive successor count and weighted downstream time
    succ_counts, downstream_time = _transitive_successor_info(n, preds, times, succs)

    if _check_deadline():
        return times.copy()

    # Transitive predecessor count
    pred_counts = _transitive_predecessor_count(n, preds, succs)

    if _check_deadline():
        return times.copy()

    # Normalize each component to [0, 1]
    def normalize(arr):
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-12:
            return np.zeros_like(arr, dtype=float)
        return (arr - mn) / (mx - mn)

    norm_est = normalize(est)
    norm_lft = normalize(lft)
    norm_succ = normalize(succ_counts.astype(float))
    norm_pred = normalize(pred_counts.astype(float))
    norm_time = normalize(times)
    norm_downstream = normalize(downstream_time)

    # Positional criticality: ratio of EST to cycle time
    est_ratio = est / cycle_time if cycle_time > 0 else est
    norm_est_ratio = normalize(est_ratio)

    # Composite priority:
    # - High downstream time -> unlocks large subgraph, place early
    # - High successor count -> unlocks fan-out
    # - High own time -> big task (LPT)
    # - High EST ratio -> constrains start, place early
    # - Low LFT -> more urgent (use 1 - norm_lft)
    # - Low predecessor count -> fewer dependencies, more likely to be eligible early
    w_downstream = 0.28
    w_succ = 0.18
    w_time = 0.22
    w_est_ratio = 0.12
    w_lft = 0.08
    w_pred = 0.12  # Inverse: fewer predecessors = more flexible/eligible early

    score = (w_downstream * norm_downstream +
             w_succ * norm_succ +
             w_time * norm_time +
             w_est_ratio * norm_est_ratio +
             w_lft * (1.0 - norm_lft) +
             w_pred * (1.0 - norm_pred))

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    if _check_deadline():
        return score

    return score
# EVOLVE-BLOCK-END
