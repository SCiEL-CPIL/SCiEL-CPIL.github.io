"""Assembly Line Balancing - priority scoring for greedy station filling."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math
from collections import defaultdict

_START_TIME = None
_DEADLINE_OFFSET = 145.0  # safety margin before 150s kill


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _deadline_reached(safety=5.0):
    return _elapsed() >= _DEADLINE_OFFSET - safety


def _build_graph(instance):
    """
    Precompute graph properties for the instance.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds_raw = instance["predecessors"]
    n = len(times)

    # Normalize predecessors to dict
    if isinstance(preds_raw, dict):
        preds = {}
        for k in range(n):
            v = preds_raw.get(k, preds_raw.get(str(k), []))
            if v:
                preds[k] = list(v)
    elif isinstance(preds_raw, (list, tuple, np.ndarray)):
        preds = {}
        for i in range(n):
            if i < len(preds_raw) and preds_raw[i]:
                preds[i] = list(preds_raw[i])
    else:
        preds = {}

    # Build successor map and in-degrees
    succ = defaultdict(list)
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                if 0 <= i < n:
                    succ[i].append(j)
                    in_deg[j] += 1

    # Topological sort (Kahn's)
    in_deg_copy = in_deg.copy()
    queue = [i for i in range(n) if in_deg_copy[i] == 0]
    topo = []
    while queue:
        u = queue.pop(0)
        topo.append(u)
        for v in succ[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)

    if len(topo) < n:
        # Cycle fallback
        for i in range(n):
            if i not in topo:
                topo.append(i)

    # Rank to end (longest path from u to sink, including u's time)
    rank_to_end = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ = 0.0
        for v in succ[u]:
            if rank_to_end[v] > max_succ:
                max_succ = rank_to_end[v]
        rank_to_end[u] = times[u] + max_succ

    # Rank from start (longest path from source to u, including u's time)
    rank_from_start = np.zeros(n, dtype=float)
    for u in topo:
        max_pred = 0.0
        for v in preds.get(u, []):
            if rank_from_start[v] > max_pred:
                max_pred = rank_from_start[v]
        rank_from_start[u] = times[u] + max_pred

    # Transitive successor count
    succ_count = np.zeros(n, dtype=int)
    for i in range(n):
        visited = set()
        stack = list(succ[i])
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(succ[v])
        succ_count[i] = len(visited)

    return {
        'n': n,
        'times': times,
        'preds': preds,
        'succ': succ,
        'in_deg': in_deg,
        'topo': topo,
        'rank_to_end': rank_to_end,
        'rank_from_start': rank_from_start,
        'succ_count': succ_count
    }


def _simulate(instance, graph, priority):
    """
    Greedy simulation: fill stations with highest-priority eligible task that fits.
    Returns number of stations used.
    """
    n = graph['n']
    times = graph['times']
    succ = graph['succ']
    in_deg = graph['in_deg']
    cycle_time = instance['cycle_time']

    assigned = np.zeros(n, dtype=bool)
    rem_preds = in_deg.copy()

    num_stations = 1
    current_time = 0.0

    while assigned.sum() < n:
        # Find eligible (unassigned, all preds satisfied)
        eligible = [i for i in range(n) if not assigned[i] and rem_preds[i] == 0]

        if not eligible:
            break

        # Find those that fit
        remaining = cycle_time - current_time
        fit = [i for i in eligible if times[i] <= remaining + 1e-9]

        if fit:
            best = max(fit, key=lambda i: priority[i])
        else:
            num_stations += 1
            current_time = 0.0
            remaining = cycle_time
            fit = [i for i in eligible if times[i] <= remaining + 1e-9]
            if fit:
                best = max(fit, key=lambda i: priority[i])
            else:
                # Task exceeds cycle time; still assign to avoid infinite loop
                best = max(eligible, key=lambda i: priority[i])

        assigned[best] = True
        current_time += times[best]

        for s in succ[best]:
            rem_preds[s] -= 1

    return num_stations


def _norm(arr):
    """Min-max normalize array to [0, 1]."""
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _make_priority(graph, w_rank=0.5, w_time=0.3, w_succ=0.2, noise=0.0):
    """Build a priority vector from weighted components."""
    n = graph['n']
    times = graph['times']
    rank_to_end = graph['rank_to_end']
    succ_count = graph['succ_count']

    p = (w_rank * _norm(rank_to_end) +
         w_time * _norm(times) +
         w_succ * _norm(succ_count.astype(float)))

    if noise > 0:
        p = p + np.random.default_rng(42).normal(0, noise, n)

    return p


def _local_search(instance, graph, priority, max_iters=5000, time_limit=12.0):
    """
    Hill-climbing local search on the priority vector.
    Uses swap and perturb moves.
    """
    n = graph['n']
    best_priority = priority.copy()
    best_score = _simulate(instance, graph, best_priority)
    rng = np.random.default_rng(123)

    start = time.time()
    iters = 0
    while iters < max_iters and (time.time() - start) < time_limit:
        iters += 1

        # Handle small n gracefully
        if n < 2:
            # Only one task, no swaps possible, just add noise if needed
            new_priority = best_priority + rng.normal(0, 0.01, n)
        else:
            # Choose move type
            r = rng.random()
            new_priority = best_priority.copy()

            if r < 0.6:
                # Swap two positions
                i, j = rng.choice(n, size=2, replace=False)
                new_priority[i], new_priority[j] = new_priority[j], new_priority[i]
            elif r < 0.85:
                # Add small noise to a few elements
                # Ensure k is at least 1 and at most n
                max_k = max(1, min(n, n // 5))
                if max_k < 1:
                    max_k = 1
                k = rng.integers(1, max_k + 1) # integers is [low, high)
                idx = rng.choice(n, size=k, replace=False)
                new_priority[idx] += rng.normal(0, 0.05, k)
            else:
                # Perturb a random contiguous block
                # Ensure start_idx and length are valid
                if n >= 2:
                    start_idx = rng.integers(0, n)
                    max_len = min(6, n - start_idx)
                    if max_len < 2:
                        # If block can't be length 2, just perturb one
                        length = 1
                    else:
                        length = rng.integers(2, max_len + 1)
                    block = rng.normal(0, 0.05, length)
                    new_priority[start_idx:start_idx + length] += block
                else:
                    # n==1, just add noise
                    new_priority[0] += rng.normal(0, 0.05)

        score = _simulate(instance, graph, new_priority)

        if score < best_score:
            best_priority = new_priority
            best_score = score

    return best_priority, best_score


def _restart_search(instance, graph, max_restarts=3, per_restart_time=10.0):
    """
    Multiple random restarts with local search from different initial heuristics.
    """
    best_priority = None
    best_score = float('inf')

    for restart in range(max_restarts):
        if _deadline_reached():
            break

        # Different initial weightings per restart
        w1 = np.random.default_rng(42 + restart * 7).uniform(0.3, 0.6)
        w2 = np.random.default_rng(43 + restart * 11).uniform(0.1, 1.0 - w1)
        w3 = 1.0 - w1 - w2
        w3 = max(0.0, w3)

        init_priority = _make_priority(graph, w_rank=w1, w_time=w2, w_succ=w3,
                                       noise=0.05)

        # Local search from this starting point
        p, s = _local_search(instance, graph, init_priority,
                             max_iters=3000, time_limit=per_restart_time)

        if s < best_score:
            best_score = s
            best_priority = p.copy()

    return best_priority, best_score


def priority(instance):
    """
    Score each assembly task. Returns a priority vector (higher = picked first).
    Uses multiple heuristics + local search + random restarts.
    """
    global _START_TIME
    _START_TIME = time.time()

    graph = _build_graph(instance)
    n = graph['n']

    # --- Phase 1: Evaluate a set of heuristic priority vectors ---
    heuristics = [
        # (name, priority_vector)
        ("rank_only", _make_priority(graph, w_rank=1.0, w_time=0.0, w_succ=0.0)),
        ("time_only", _make_priority(graph, w_rank=0.0, w_time=1.0, w_succ=0.0)),
        ("rank_time", _make_priority(graph, w_rank=0.6, w_time=0.4, w_succ=0.0)),
        ("rank_time_succ", _make_priority(graph, w_rank=0.5, w_time=0.3, w_succ=0.2)),
        ("rank_succ", _make_priority(graph, w_rank=0.7, w_time=0.1, w_succ=0.2)),
        ("balanced", _make_priority(graph, w_rank=0.4, w_time=0.35, w_succ=0.25)),
        ("critical_heavy", _make_priority(graph, w_rank=0.7, w_time=0.2, w_succ=0.1)),
    ]

    best_priority = None
    best_score = float('inf')

    for name, p in heuristics:
        if _deadline_reached():
            break
        score = _simulate(instance, graph, p)
        if score < best_score:
            best_score = score
            best_priority = p.copy()

    # --- Phase 2: Local search from the best heuristic ---
    if not _deadline_reached():
        p, s = _local_search(instance, graph, best_priority,
                             max_iters=4000, time_limit=15.0)
        if s < best_score:
            best_score = s
            best_priority = p

    # --- Phase 3: Random restarts ---
    if not _deadline_reached():
        p, s = _restart_search(instance, graph, max_restarts=3, per_restart_time=10.0)
        if s < best_score:
            best_score = s
            best_priority = p

    # --- Phase 4: Final polish (tighter local search) ---
    if not _deadline_reached(safety=8.0):
        p, s = _local_search(instance, graph, best_priority,
                             max_iters=2000, time_limit=8.0)
        if s < best_score:
            best_score = s
            best_priority = p

    return best_priority

# EVOLVE-BLOCK-END
