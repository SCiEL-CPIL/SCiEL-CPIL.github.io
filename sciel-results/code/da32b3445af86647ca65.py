"""Assembly Line Balancing - Structural decomposition with iterative local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import deque

def priority(instance):
    """
    Score each assembly task for the priority rule.
    Uses a multi-phase approach:
    1. Compute structural features (critical path, successors, etc.)
    2. Generate candidate priority vectors from different heuristics
    3. Evaluate each with fast greedy simulation
    4. Perform local search (swap-based) on the best candidate to refine
    """
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Build graph structures
    adj = [[] for _ in range(n)]  # adj[i] = successors of i
    preds_list = []
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        preds_list.append(preds)
        for p in preds:
            adj[p].append(i)

    # Topological sort
    in_deg = np.array([len(p) for p in preds_list], dtype=int)
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    in_deg_tmp = in_deg.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in adj[node]:
            in_deg_tmp[succ] -= 1
            if in_deg_tmp[succ] == 0:
                queue.append(succ)

    if len(topo_order) < n:
        # Cycle detected; fall back to task times
        return times.copy()

    # Compute longest path to end (LPE) for each task
    lpe = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in adj[node]:
            val = times[succ] + lpe[succ]
            if val > max_succ:
                max_succ = val
        lpe[node] = times[node] + max_succ

    # Compute longest path from start (LPS) for each task
    lps = np.zeros(n, dtype=float)
    for node in topo_order:
        for succ in adj[node]:
            val = times[node] + lps[node]
            if val > lps[succ]:
                lps[succ] = val

    # Number of successors (transitive)
    num_succ = np.zeros(n, dtype=int)
    for node in reversed(topo_order):
        for succ in adj[node]:
            num_succ[node] += num_succ[succ] + 1

    # Number of predecessors (transitive)
    num_pred = np.zeros(n, dtype=int)
    for node in topo_order:
        for succ in adj[node]:
            num_pred[succ] += num_pred[node] + 1

    # Position in topological order
    pos_in_topo = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        pos_in_topo[node] = idx

    if n > 1:
        norm_pos = pos_in_topo / (n - 1)
    else:
        norm_pos = np.zeros(n, dtype=float)

    # Slack: how much "free time" a task has relative to cycle time
    eff_slack = np.maximum(0.0, cycle_time - lpe)

    # Candidate priority scores
    candidates = {}
    candidates["lpe"] = lpe.copy()
    candidates["lpe_succ"] = lpe + 0.01 * num_succ.astype(float)
    candidates["lpe_times_succ"] = lpe * (1.0 + 0.05 * np.log1p(num_succ.astype(float)))
    candidates["lps"] = lps.copy()
    candidates["time"] = times.copy()
    candidates["time_lpe"] = times + 0.1 * lpe
    candidates["early_pos"] = 1.0 - norm_pos
    candidates["combo1"] = lpe * (1.0 + 0.02 * num_succ.astype(float))
    candidates["combo2"] = 0.7 * lpe + 0.3 * lps
    candidates["combo3"] = lpe + 0.05 * num_succ.astype(float) - 0.01 * norm_pos
    candidates["combo4"] = (lpe / (lpe.max() + 1e-9)) * (1.0 + 0.1 * num_succ.astype(float) / (num_succ.max() + 1e-9))
    candidates["combo5"] = lpe * (1.0 + 0.03 * np.sqrt(num_succ.astype(float)))
    candidates["combo6"] = lpe + 0.1 * num_succ.astype(float)
    candidates["combo7"] = 0.8 * lpe + 0.2 * times * (1.0 + 0.05 * num_succ.astype(float))
    candidates["combo8"] = lpe * np.exp(0.01 * num_succ.astype(float))
    candidates["neg_slack"] = -eff_slack.copy()
    candidates["lpe_neg_slack"] = lpe - 0.1 * eff_slack

    # Pre-compute for speed
    pred_sets = [set(p) for p in preds_list]

    def simulate(priority_scores):
        """Fast greedy simulation using a heap of eligible tasks sorted by priority."""
        completed = [False] * n
        stations = 0
        current_load = 0.0
        remaining_count = n

        pending = np.array([len(p) for p in preds_list], dtype=int)
        heap = []

        for t in range(n):
            if pending[t] == 0 and times[t] <= cycle_time + 1e-9:
                heapq.heappush(heap, (-priority_scores[t], t))

        while remaining_count > 0:
            placed = False
            while heap:
                neg_pri, t = heapq.heappop(heap)
                if completed[t]:
                    continue
                if times[t] <= cycle_time - current_load + 1e-9:
                    completed[t] = True
                    current_load += times[t]
                    remaining_count -= 1
                    placed = True
                    for s in adj[t]:
                        pending[s] -= 1
                        if pending[s] == 0 and times[s] <= cycle_time + 1e-9:
                            heapq.heappush(heap, (-priority_scores[s], s))
                    break

            if not placed:
                stations += 1
                current_load = 0.0
                for t in range(n):
                    if not completed[t] and pending[t] == 0 and times[t] <= cycle_time + 1e-9:
                        heapq.heappush(heap, (-priority_scores[t], t))

            if not heap:
                stations += 1
                break

        return stations

    # Evaluate candidates and pick the best
    best_priority = None
    best_stations = float('inf')

    for name, p in candidates.items():
        if time.time() > DEADLINE:
            break
        stations = simulate(p)
        if stations < best_stations:
            best_stations = stations
            best_priority = p.copy()

    # Parameter sweeps
    if best_priority is not None and time.time() < DEADLINE - 40:
        for alpha in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
            if time.time() > DEADLINE:
                break
            p = lpe + alpha * num_succ.astype(float)
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()

        for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            if time.time() > DEADLINE:
                break
            b = 1.0 - a
            p = a * lpe + b * lps
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()

        for beta in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
            if time.time() > DEADLINE:
                break
            p = lpe * (1.0 + beta * np.sqrt(num_succ.astype(float)))
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()

        for gamma in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
            if time.time() > DEADLINE:
                break
            p = lpe - gamma * eff_slack
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()

    # Local search polish: swap-based local search on priority ordering
    if best_priority is not None and time.time() < DEADLINE - 5:
        current_priority = best_priority.copy()
        current_stations = best_stations
        improvement_found = False

        order = np.argsort(-current_priority)

        max_swaps = 500
        swaps_done = 0
        while swaps_done < max_swaps and time.time() < DEADLINE - 2:
            i, j = np.random.choice(n, size=2, replace=False)
            if i == j:
                continue

            old_i, old_j = current_priority[i], current_priority[j]
            current_priority[i], current_priority[j] = old_j, old_i
            swaps_done += 1

            stations = simulate(current_priority)
            if stations < current_stations:
                current_stations = stations
                improvement_found = True
            else:
                current_priority[i], current_priority[j] = old_i, old_j

        if improvement_found:
            best_priority = current_priority
            best_stations = current_stations

    if best_priority is None:
        best_priority = lpe.copy()

    return np.asarray(best_priority, dtype=float)

# EVOLVE-BLOCK-END
