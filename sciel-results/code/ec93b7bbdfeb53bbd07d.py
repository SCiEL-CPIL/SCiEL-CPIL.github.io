"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _build_graph(times, predecessors, n):
    """Build successors list, in-degrees, and a topological order."""
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i, preds in enumerate(predecessors):
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1

    topo_order = []
    in_deg = in_degree.copy()
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)

    if len(topo_order) < n:
        # Cycle: append remaining nodes so downstream sums stay defined
        seen = set(topo_order)
        for i in range(n):
            if i not in seen:
                topo_order.append(i)
    return successors, topo_order


def _downstream_features(times, predecessors, n, cycle_time):
    """
    Compute the structural features used by the base score:
      - longest_downstream: critical-path length from each node
      - total_downstream_work: sum of times over all descendants (incl. self)
      - topo_index: position in topological order
    Returns (longest_downstream, total_downstream_work, topo_index, ok).
    """
    successors, topo_order = _build_graph(times, predecessors, n)

    longest_downstream = np.zeros(n, dtype=float)
    total_downstream_work = np.zeros(n, dtype=float)

    for node in reversed(topo_order):
        max_succ_ld = 0.0
        sum_succ_work = 0.0
        for succ in successors[node]:
            if longest_downstream[succ] > max_succ_ld:
                max_succ_ld = longest_downstream[succ]
            sum_succ_work += total_downstream_work[succ]
        longest_downstream[node] = times[node] + max_succ_ld
        total_downstream_work[node] = times[node] + sum_succ_work

    topo_index = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        topo_index[node] = idx

    return longest_downstream, total_downstream_work, topo_index


def _base_score(times, predecessors, n, cycle_time):
    """
    Static structural score (the current program's design):
      total downstream work (primary) + inverse slack (secondary) + topo depth (tie-break).
    """
    longest_downstream, total_downstream_work, topo_index = _downstream_features(
        times, predecessors, n, cycle_time
    )

    max_ld = np.max(longest_downstream) if n > 0 else 1.0
    max_td = np.max(total_downstream_work) if n > 0 else 1.0
    max_topo = n if n > 0 else 1.0
    if max_ld == 0:
        max_ld = 1.0
    if max_td == 0:
        max_td = 1.0
    if max_topo == 0:
        max_topo = 1.0

    # Inverse slack: higher when the task's downstream path is tight vs cycle time
    slack = cycle_time - longest_downstream
    slack_clipped = np.maximum(slack, 0.0)
    inverse_slack = 1.0 / (1.0 + slack_clipped)
    max_is = np.max(inverse_slack) if n > 0 else 1.0
    if max_is == 0:
        max_is = 1.0
    norm_is = inverse_slack / max_is

    norm_ld = longest_downstream / max_ld
    norm_td = total_downstream_work / max_td
    norm_topo = 1.0 - (topo_index / max_topo)

    score = 1.0 * norm_td + 0.5 * norm_is + 0.1 * norm_topo
    return score


def _simulate_greedy(priority_scores, times, predecessors, cycle_time):
    """
    Simulate the evaluator's greedy assignment:
      stations are filled with the highest-priority eligible task that fits the cycle time.
    Returns number of stations used.
    """
    n = len(times)
    assigned = [False] * n
    remaining = set(range(n))
    station_load = 0.0
    num_stations = 1

    while remaining:
        best_task_idx = -1
        best_priority = -np.inf
        for i in remaining:
            # eligibility: all predecessors assigned
            eligible = True
            for p in predecessors[i]:
                if not assigned[p]:
                    eligible = False
                    break
            if not eligible:
                continue
            if station_load + times[i] <= cycle_time:
                if priority_scores[i] > best_priority:
                    best_priority = priority_scores[i]
                    best_task_idx = i

        if best_task_idx != -1:
            assigned[best_task_idx] = True
            station_load += times[best_task_idx]
            remaining.discard(best_task_idx)
        else:
            # No eligible task fits in current station -> open a new one
            num_stations += 1
            station_load = 0.0
            best_task_idx = -1
            best_priority = -np.inf
            for i in remaining:
                eligible = True
                for p in predecessors[i]:
                    if not assigned[p]:
                        eligible = False
                        break
                if not eligible:
                    continue
                if times[i] <= cycle_time:
                    if priority_scores[i] > best_priority:
                        best_priority = priority_scores[i]
                        best_task_idx = i
            if best_task_idx != -1:
                assigned[best_task_idx] = True
                station_load += times[best_task_idx]
                remaining.discard(best_task_idx)
            else:
                # Infeasible (a single task exceeds cycle time) - bail out gracefully
                break

    return num_stations


def priority(instance):
    """
    Base score = rich structural features (total downstream work + inverse slack + topo depth).
    Refinement = simulate the greedy packing and perturb the score vector to reduce stations.
    """
    start_time = time.time()
    DEADLINE = 150.0

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)
    predecessors = instance.get("predecessors", [])

    if n == 0:
        return np.array([], dtype=float)

    # Base structural score (current program's design)
    base = _base_score(times, predecessors, n, cycle_time)

    # Evaluate the base score directly
    best_scores = base.copy()
    best_stations = _simulate_greedy(best_scores, times, predecessors, cycle_time)

    total_time = float(np.sum(times))
    lower_bound = int(np.ceil(total_time / cycle_time)) if cycle_time > 0 else best_stations

    # If we already hit the theoretical lower bound, stop early
    if best_stations <= lower_bound:
        return best_scores

    # Refinement: local search over the score vector via random perturbations.
    # The greedy objective is nonsmooth in the ordering, so we treat it as a black box
    # and explore the neighborhood of the strong structural base score.
    rng = np.random.default_rng(12345)

    # A few deterministic tie-break noise scales, then random exploration
    noise_scales = [0.0, 0.005, 0.02, 0.08]
    for scale in noise_scales:
        if time.time() > DEADLINE - 5.0:
            break
        if scale == 0.0:
            candidate = base.copy()
        else:
            noise = rng.uniform(-scale, scale, n)
            candidate = base + noise
        stations = _simulate_greedy(candidate, times, predecessors, cycle_time)
        if stations < best_stations:
            best_stations = stations
            best_scores = candidate.copy()
            if best_stations <= lower_bound:
                return best_scores

    # Random exploration with adaptive step size while time remains
    step = 0.1
    while time.time() < DEADLINE - 2.0:
        # Occasionally re-center toward base to avoid drifting too far
        if rng.random() < 0.1:
            center = base
        else:
            center = best_scores
        noise = rng.uniform(-step, step, n)
        candidate = center + noise
        stations = _simulate_greedy(candidate, times, predecessors, cycle_time)
        if stations < best_stations:
            best_stations = stations
            best_scores = candidate.copy()
            # Accepting: shrink step slightly (local polish)
            step = max(0.02, step * 0.8)
            if best_stations <= lower_bound:
                break
        else:
            # Reject: occasionally grow step to escape local plateaus
            if rng.random() < 0.2:
                step = min(0.5, step * 1.3)

    return best_scores
# EVOLVE-BLOCK-END
