"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import deque

_DEADLINE_SECONDS = 140.0  # safety margin before 150s kill


def _topo_order_and_successors(times, predecessors):
    """Return (topological_order, successors_list)."""
    n = len(times)
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1

    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)

    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)

    return topo, successors


def _compute_longest_downstream_times(times, predecessors):
    """For each task, compute the sum of times along the longest downstream path.
    This is the 'critical path weight' – the total work that depends on this task."""
    n = len(times)
    topo, successors = _topo_order_and_successors(times, predecessors)

    longest_down = [0.0] * n
    for u in reversed(topo):
        max_path = 0.0
        for v in successors[u]:
            candidate = times[v] + longest_down[v]
            if candidate > max_path:
                max_path = candidate
        longest_down[u] = max_path

    return np.array(longest_down, dtype=float)


def _compute_count_descendants(times, predecessors):
    """Count transitive descendants for each task."""
    n = len(times)
    topo, successors = _topo_order_and_successors(times, predecessors)

    count_desc = [0] * n
    for u in reversed(topo):
        total = 0
        for v in successors[u]:
            total += 1 + count_desc[v]
        count_desc[u] = total

    return np.array(count_desc, dtype=float)


def _compute_successor_count_direct(times, predecessors):
    """Count direct successors for each task."""
    n = len(times)
    succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        for p in predecessors[i]:
            succ_count[p] += 1
    return succ_count


def _assign_stations_greedy(times, predecessors, cycle_time, priority_scores):
    """Assign tasks to stations using greedy rule: highest priority eligible task that fits.
    Uses a max-heap for efficient selection."""
    n = len(times)

    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)

    # Remaining count: how many predecessors are not yet assigned
    remaining_preds = [len(predecessors[i]) for i in range(n)]

    unassigned = set(range(n))
    station_count = 0
    station_load = 0.0

    while unassigned:
        # Find eligible tasks: all predecessors assigned and fits in remaining capacity
        remaining_cap = cycle_time - station_load
        eligible = []
        for t in unassigned:
            if remaining_preds[t] == 0 and times[t] <= remaining_cap:
                eligible.append(t)

        if not eligible:
            # Close current station and start a new one
            if station_load > 0:
                station_count += 1
            station_load = 0.0
            remaining_cap = cycle_time
            eligible = []
            for t in unassigned:
                if remaining_preds[t] == 0 and times[t] <= cycle_time:
                    eligible.append(t)

        if not eligible:
            # Shouldn't happen if cycle_time >= max(time), but handle gracefully
            break

        # Pick the highest priority task among eligible
        best_task = max(eligible, key=lambda t: priority_scores[t])

        # Assign it
        unassigned.discard(best_task)
        station_load += times[best_task]

        # Update remaining_preds for successors
        for s in successors[best_task]:
            remaining_preds[s] -= 1

    # Don't forget the last station
    if station_load > 0 or unassigned:
        station_count += 1
    elif station_count == 0:
        station_count = 1

    return station_count


def _compute_priority(instance):
    """Compute a comprehensive priority score combining multiple heuristics."""
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)

    # 1. Task time (longer is better)
    max_time = np.max(times) + 1e-9
    time_score = times / max_time

    # 2. Longest downstream path in time units (critical path weight)
    longest_down = _compute_longest_downstream_times(times, predecessors)
    max_ld = np.max(longest_down) + 1e-9
    down_score = longest_down / max_ld

    # 3. Number of transitive successors (more dependent tasks = more urgent)
    succ_count = _compute_count_descendants(times, predecessors)
    max_sc = np.max(succ_count) + 1e-9
    succ_score = succ_count / max_sc

    # 4. Direct successor count
    direct_succ = _compute_successor_count_direct(times, predecessors)
    max_ds = np.max(direct_succ) + 1e-9
    direct_score = direct_succ / max_ds

    # Weighted combination – emphasize critical path weight and time
    composite = 0.35 * time_score + 0.30 * down_score + 0.25 * succ_score + 0.10 * direct_score

    return composite


def priority(instance):
    """Score each assembly task. Returns one score per task."""
    start_time = time.time()

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Compute base priority using comprehensive heuristic
    base_priority = _compute_priority(instance)

    # Deadline guard: if we're running out of time, return immediately
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE_SECONDS:
        return base_priority

    return base_priority
# EVOLVE-BLOCK-END
