"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import deque


def _simulate_greedy(times, preds, cycle_time, priority_scores):
    """
    Simulate the greedy station filling with given priority scores.
    Returns the number of stations used.
    """
    n = len(times)
    if n == 0:
        return 0

    # Precompute successors
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)

    remaining_preds = [len(preds[i]) for i in range(n)]
    eligible_list = [i for i in range(n) if remaining_preds[i] == 0]

    stations = 0
    remaining = n

    while remaining > 0:
        stations += 1
        load = 0.0
        # Fill current station
        while True:
            # Find best eligible task that fits
            best_task = -1
            best_prio = -np.inf
            for i in eligible_list:
                if times[i] <= cycle_time - load + 1e-9:
                    if priority_scores[i] > best_prio:
                        best_prio = priority_scores[i]
                        best_task = i
            if best_task == -1:
                break
            # Assign best_task
            load += times[best_task]
            remaining -= 1
            # Remove from eligible_list
            eligible_list.remove(best_task)
            # Update successors
            for s in succ[best_task]:
                remaining_preds[s] -= 1
                if remaining_preds[s] == 0:
                    eligible_list.append(s)

    return stations


def priority(instance):
    """
    Score each assembly task using a refined Critical Path Density approach
    with local search over weight parameters.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)
    predecessors = instance.get("predecessors", [])

    if n == 0:
        return np.array([])

    # Build adjacency: successors[i] = list of tasks that depend on i
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)

    for i, preds in enumerate(predecessors):
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological sort
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

    # If not all nodes are in topo_order, there's a cycle; fall back to simple ordering
    if len(topo_order) < n:
        return times.copy()

    # Compute longest downstream path for each node (reverse topological order)
    longest_downstream = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in successors[node]:
            if longest_downstream[succ] > max_succ:
                max_succ = longest_downstream[succ]
        longest_downstream[node] = times[node] + max_succ

    # Compute number of immediate successors (fan-out)
    fan_out = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Compute number of immediate predecessors (fan-in)
    fan_in = np.array([len(predecessors[i]) if i < len(predecessors) else 0 for i in range(n)], dtype=float)

    # Compute "rank" or depth from start (longest path from any source to node)
    longest_upstream = np.zeros(n, dtype=float)
    for node in topo_order:
        max_pred = 0.0
        for p in predecessors[node]:
            if longest_upstream[p] + times[p] > max_pred:
                max_pred = longest_upstream[p] + times[p]
        longest_upstream[node] = max_pred

    # Normalize components
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    norm_ld = normalize(longest_downstream)
    norm_time = normalize(times)
    norm_fanout = normalize(fan_out)
    norm_fanin = normalize(fan_in)
    norm_upstream = normalize(longest_upstream)

    # Base priority score
    # We want to prioritize tasks that are on the critical path AND have high fan-out
    # Initial weights
    w_ld, w_time, w_fanout, w_fanin, w_upstream = 1.0, 0.5, 0.3, 0.2, 0.0
    base_score = w_ld * norm_ld + w_time * norm_time + w_fanout * norm_fanout + w_fanin * norm_fanin + w_upstream * norm_upstream

    # Evaluate base score
    best_score = base_score.copy()
    best_stations = _simulate_greedy(times, predecessors, cycle_time, base_score)

    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return best_score

    # Local search over weight parameters
    # Define a set of weight combinations to try
    weight_combos = [
        (1.0, 0.5, 0.3, 0.2, 0.0),
        (1.0, 0.3, 0.5, 0.2, 0.0),
        (1.0, 0.5, 0.5, 0.0, 0.0),
        (1.0, 0.7, 0.3, 0.0, 0.0),
        (1.0, 0.5, 0.3, 0.0, 0.3),
        (1.0, 0.5, 0.3, 0.0, 0.5),
        (1.2, 0.5, 0.3, 0.2, 0.0),
        (0.8, 0.5, 0.5, 0.2, 0.0),
        (1.0, 1.0, 0.5, 0.0, 0.0),
        (1.0, 0.5, 0.3, 0.5, 0.0),
    ]

    # Also try some random perturbations
    rng = np.random.RandomState(42)
    for _ in range(10):
        w_ld_r = rng.uniform(0.5, 1.5)
        w_time_r = rng.uniform(0.0, 1.0)
        w_fanout_r = rng.uniform(0.0, 1.0)
        w_fanin_r = rng.uniform(0.0, 0.5)
        w_upstream_r = rng.uniform(0.0, 0.5)
        weight_combos.append((w_ld_r, w_time_r, w_fanout_r, w_fanin_r, w_upstream_r))

    for w_ld, w_time, w_fanout, w_fanin, w_upstream in weight_combos:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break

        score = w_ld * norm_ld + w_time * norm_time + w_fanout * norm_fanout + w_fanin * norm_fanin + w_upstream * norm_upstream
        stations = _simulate_greedy(times, predecessors, cycle_time, score)

        if stations < best_stations:
            best_stations = stations
            best_score = score.copy()

    return best_score
# EVOLVE-BLOCK-END
