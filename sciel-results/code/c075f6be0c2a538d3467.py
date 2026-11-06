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


def _downstream_weight(n, succ):
    """Total number of tasks in each task's successor subtree (including itself)."""
    weight = np.ones(n, dtype=float)
    # Process in reverse topological order (successors before predecessors)
    # Use a simple Kahn-like reverse: repeatedly pick nodes with no unprocessed successors
    processed = np.zeros(n, dtype=bool)
    remaining = n
    # To be safe with cycles, limit iterations
    order = []
    indeg_succ = np.zeros(n, dtype=int)  # number of unprocessed successors
    for i in range(n):
        indeg_succ[i] = len(succ[i])
    queue = [i for i in range(n) if indeg_succ[i] == 0]
    while queue and remaining > 0:
        # pick arbitrary
        i = queue.pop()
        if processed[i]:
            continue
        processed[i] = True
        remaining -= 1
        order.append(i)
        for p in range(n):
            if i in succ[p]:
                indeg_succ[p] -= 1
                if indeg_succ[p] == 0:
                    queue.append(p)
    # Fallback for any unprocessed (cycles): just use identity
    if remaining > 0:
        for i in range(n):
            if not processed[i]:
                order.append(i)
    # Now compute weights in reverse topological order (leaves first)
    for i in reversed(order):
        for s in succ[i]:
            weight[i] += weight[s]
    return weight


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

    # Downstream weight
    down_w = _downstream_weight(n, succ)

    # Immediate successor count
    imm_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    imm_succ = np.maximum(imm_succ, 1.0)

    # Bottleneck slack: (cycle_time - time_i) / imm_succ ; smaller => higher priority
    slack = (cycle_time - times) / imm_succ
    # Invert: 1 / (slack + eps) so smaller slack => larger score
    eps = 1e-6
    slack_score = 1.0 / (slack + eps)

    # Downstream weight score (log to avoid scale issues)
    down_score = np.log1p(down_w)

    # Predecessor count penalty (mild): tasks with many preds are harder to place early
    pred_count = in_deg.astype(float)
    pred_penalty = 1.0 / (1.0 + 0.3 * pred_count)

    # Combine: weight downstream heavily (measured), then slack, then pred penalty
    # Normalize each to [0,1] roughly
    def _norm(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.ones_like(x)
        return (x - mn) / (mx - mn)

    s_down = _norm(down_score)
    s_slack = _norm(slack_score)
    s_pred = _norm(pred_penalty)

    base = 1.0 * s_down + 0.7 * s_slack + 0.3 * s_pred

    # Add time as a tie-breaker (longer tasks slightly preferred)
    s_time = _norm(times)
    base = base + 0.2 * s_time

    # Local refinement: simulate greedy, then boost tasks that are "tight" in their station
    if time.time() < deadline - 5.0:
        stations, assigned, last_load = _greedy_stations(n, times, succ, in_deg, base, cycle_time)
        # For each station, compute slack left after the last task placed
        # Boost tasks whose station has little remaining capacity (they are "tight")
        # and that have high downstream weight
        station_remaining = {}
        for i in range(n):
            st = assigned[i]
            station_remaining[st] = cycle_time - times[i]
        # Use the minimum remaining in the station as a proxy for tightness
        tight = np.zeros(n, dtype=float)
        min_rem = {}
        for i in range(n):
            st = assigned[i]
            rem = cycle_time - times[i]
            if st not in min_rem or rem < min_rem[st]:
                min_rem[st] = rem
        for i in range(n):
            st = assigned[i]
            tight[i] = 1.0 / (min_rem[st] + eps)
        tight = _norm(tight)
        # Blend: keep base, add a small boost for tight tasks with high downstream
        base = base + 0.15 * tight * s_down

    # Ensure finite
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    return base
# EVOLVE-BLOCK-END
