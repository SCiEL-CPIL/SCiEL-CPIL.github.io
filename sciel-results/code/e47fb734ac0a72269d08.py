"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _build_graph(n, predecessors):
    """Build successor list and in-degree from predecessors (list of lists)."""
    succ = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i, preds in enumerate(predecessors):
        for p in preds:
            succ[p].append(i)
            in_deg[i] += 1
    return succ, in_deg


def _downstream_weight(n, succ, times):
    """Total time and count in each task's successor subtree (including itself)."""
    count_w = np.ones(n, dtype=float)
    time_w = times.copy().astype(float)
    # Get topological order via Kahn's algorithm
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for s in succ[i]:
            in_deg[s] += 1
    queue = [i for i in range(n) if in_deg[i] == 0]
    topo_order = []
    temp_indeg = in_deg.copy()
    while queue:
        i = queue.pop()
        topo_order.append(i)
        for s in succ[i]:
            temp_indeg[s] -= 1
            if temp_indeg[s] == 0:
                queue.append(s)
    # Fallback for cycles
    if len(topo_order) < n:
        for i in range(n):
            if i not in set(topo_order):
                topo_order.append(i)
    # Process in reverse topological order (successors/leaves first)
    for i in reversed(topo_order):
        for s in succ[i]:
            count_w[i] += count_w[s]
            time_w[i] += time_w[s]
    return count_w, time_w


def _greedy_stations(n, times, succ, in_deg, priority_scores, cycle_time):
    """Simulate the greedy station filling. Returns station index per task and total stations."""
    assigned = np.full(n, -1, dtype=int)
    station_load = 0.0
    station_idx = 0
    # eligible: in-degree 0
    eligible = [i for i in range(n) if in_deg[i] == 0]
    # We'll maintain a priority-sorted eligible list
    # For speed, just sort each time we pick
    remaining_indeg = in_deg.copy()
    placed = 0
    while placed < n:
        # Start a new station if no eligible task fits
        if not eligible:
            station_idx += 1
            station_load = 0.0
            continue
        # Pick the highest priority task that fits
        best = None
        best_p = -np.inf
        for i in eligible:
            if times[i] <= cycle_time - station_load:
                if priority_scores[i] > best_p:
                    best_p = priority_scores[i]
                    best = i
        if best is None:
            # None fits; start new station
            station_idx += 1
            station_load = 0.0
            continue
        # Place best
        assigned[best] = station_idx
        station_load += times[best]
        placed += 1
        eligible.remove(best)
        # Unlock successors
        for s in succ[best]:
            remaining_indeg[s] -= 1
            if remaining_indeg[s] == 0:
                eligible.append(s)
    return station_idx + 1, assigned, station_load


def priority(instance):
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = float(instance["cycle_time"])
    n = len(times)

    succ, in_deg = _build_graph(n, predecessors)

    # Downstream weight (count and time)
    down_count, down_time = _downstream_weight(n, succ, times)

    # Immediate successor count
    imm_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    imm_succ = np.maximum(imm_succ, 1.0)

    # Bottleneck slack: (cycle_time - time_i) / imm_succ ; smaller => higher priority
    eps = 1e-6
    slack = (cycle_time - times) / imm_succ
    slack_score = 1.0 / (slack + eps)

    # Downstream time score (log to avoid scale issues) - more meaningful than count
    down_time_score = np.log1p(down_time)
    down_count_score = np.log1p(down_count)

    # Predecessor count penalty (mild): tasks with many preds are harder to place early
    pred_count = in_deg.astype(float)
    pred_penalty = 1.0 / (1.0 + 0.3 * pred_count)

    # Normalize helper
    def _norm(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.ones_like(x)
        return (x - mn) / (mx - mn)

    s_down_time = _norm(down_time_score)
    s_down_count = _norm(down_count_score)
    s_slack = _norm(slack_score)
    s_pred = _norm(pred_penalty)
    s_time = _norm(times)

    # Combine: downstream time is the strongest signal, then slack, then count
    base = 1.2 * s_down_time + 0.8 * s_slack + 0.4 * s_down_count + 0.2 * s_pred + 0.25 * s_time

    # Local refinement: simulate greedy, then boost tasks based on actual station utilization
    if time.time() < deadline - 5.0:
        stations, assigned, last_load = _greedy_stations(n, times, succ, in_deg, base, cycle_time)
        # Compute actual station loads
        station_loads = {}
        for i in range(n):
            st = assigned[i]
            station_loads[st] = station_loads.get(st, 0.0) + times[i]
        # Utilization = load / cycle_time; high utilization stations are tight
        # Boost tasks in high-utilization stations that also have high downstream time
        util = np.zeros(n, dtype=float)
        for i in range(n):
            st = assigned[i]
            util[i] = station_loads[st] / cycle_time
        util = _norm(util)
        # Blend: add boost for tasks in tight stations with high downstream
        base = base + 0.2 * util * s_down_time

    # Ensure finite
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    return base
# EVOLVE-BLOCK-END
