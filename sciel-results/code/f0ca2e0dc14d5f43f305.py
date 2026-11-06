"""Assembly Line Balancing - Combined approach with fast local search and polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque


def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling. Within each station, repeatedly place the
    highest-priority eligible task that fits until none fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0

    successors = defaultdict(list)
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)

    if np.any(times > cycle_time + 1e-9):
        return 10**9

    eligible = [i for i in range(n) if unassigned_preds[i] == 0]
    in_eligible = [False] * n
    for i in eligible:
        in_eligible[i] = True

    order_key = np.arange(n, dtype=int)

    station_count = 1
    current_load = 0.0

    while eligible:
        candidates = sorted(eligible, key=lambda i: (-priority_scores[i], order_key[i]))
        placed_any = False
        for task in candidates:
            if times[task] <= cycle_time - current_load + 1e-9:
                current_load += times[task]
                eligible.remove(task)
                in_eligible[task] = False
                placed_any = True
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.append(succ)
                        in_eligible[succ] = True
                if len(eligible) > len(candidates):
                    candidates = sorted(eligible, key=lambda i: (-priority_scores[i], order_key[i]))
                continue

        if not placed_any:
            station_count += 1
            current_load = 0.0

    return station_count


def _topo_and_features(times, predecessors, successors, n):
    """Topological order + critical path + downstream load + transitive counts."""
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)

    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    if len(topo_order) != n:
        return None, None, None, None

    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            val = times[u] + cp[v]
            if val > cp[u]:
                cp[u] = val

    dl = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        dl[u] = times[u]
        for v in successors[u]:
            dl[u] += dl[v]

    trans_succ = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        s = 0.0
        for v in successors[u]:
            s += 1.0 + trans_succ[v]
        trans_succ[u] = s

    return cp, dl, trans_succ, topo_order


def _build_heuristics(times, predecessors, successors, n, cycle_time):
    """Build a diverse set of candidate priority vectors."""
    cp, dl, trans_succ, topo_order = _topo_and_features(times, predecessors, successors, n)
    if cp is None:
        return [times.astype(float).copy()]

    num_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    num_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)

    cp_max = np.max(cp) + 1e-6
    dl_max = np.max(dl) + 1e-6
    ts_max = np.max(trans_succ) + 1e-6
    t_max = np.max(times) + 1e-6

    heuristics = []
    heuristics.append(cp.copy())
    heuristics.append(times + cp)
    heuristics.append(dl.copy())
    heuristics.append(cp * (1.0 + 0.2 * trans_succ / ts_max))
    heuristics.append(trans_succ.copy())
    slack = (cycle_time - times) / (1.0 + trans_succ)
    heuristics.append(-slack.copy())
    slack_imm = (cycle_time - times) / (1.0 + num_succ)
    heuristics.append(-slack_imm.copy())
    heuristics.append(cp / (1.0 + trans_succ / ts_max))
    heuristics.append(dl * (cp / cp_max))
    heuristics.append(times.copy())
    heuristics.append(dl + cp)
    heuristics.append(times * (cp / cp_max))
    heuristics.append(cp - 0.05 * num_pred)
    heuristics.append(0.5 * (cp / cp_max) + 0.3 * (dl / dl_max) + 0.2 * (trans_succ / ts_max))
    rank_cp = np.argsort(np.argsort(cp)).astype(float)
    heuristics.append(rank_cp)
    rank_dl = np.argsort(np.argsort(dl)).astype(float)
    heuristics.append(rank_dl)
    heuristics.append(dl / (1.0 + trans_succ / ts_max))
    heuristics.append(times + 0.1 * trans_succ)

    return heuristics


def _fast_local_search(priority, times, predecessors, cycle_time, n, deadline, max_evals=5000):
    """
    Fast randomized local search with early termination.
    Tries random swaps, keeps improvements, stops if no improvement in a batch.
    """
    current_priority = priority.copy()
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    best_priority = current_priority.copy()
    best_score = current_score

    no_improve_count = 0
    batch_size = 20
    eval_count = 0

    while time.time() < deadline and eval_count < max_evals:
        batch_improved = False
        for _ in range(batch_size):
            if time.time() > deadline or eval_count >= max_evals:
                break

            i, j = random.sample(range(n), 2)
            old_i, old_j = current_priority[i], current_priority[j]
            current_priority[i], current_priority[j] = old_j, old_i
            eval_count += 1

            new_score = _greedy_balance(times, predecessors, cycle_time, current_priority)

            if new_score < current_score:
                current_score = new_score
                batch_improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_priority = current_priority.copy()
                    if best_score <= 1:
                        return best_priority, best_score
            else:
                current_priority[i], current_priority[j] = old_i, old_j

        if not batch_improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
        else:
            no_improve_count = 0

    return best_priority, best_score


def _polish(priority, times, predecessors, cycle_time, n, deadline):
    """
    Final polish pass: more aggressive local search with smaller perturbations.
    Tries adjacent swaps in priority ordering.
    """
    current_priority = priority.copy()
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    best_priority = current_priority.copy()
    best_score = current_score

    # Sort tasks by current priority to get an ordering
    order = np.argsort(-current_priority)

    # Try swapping adjacent tasks in the priority ordering
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(order) - 1):
            if time.time() > deadline:
                break
            i, j = order[idx], order[idx + 1]
            old_i, old_j = current_priority[i], current_priority[j]
            current_priority[i], current_priority[j] = old_j, old_i

            new_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_priority = current_priority.copy()
                    if best_score <= 1:
                        return best_priority, best_score
            else:
                current_priority[i], current_priority[j] = old_i, old_j

    return best_priority, best_score


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]

    n = len(times)
    if n == 0:
        return np.array([])

    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s kill

    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)

    # Build diverse heuristics
    heuristics = _build_heuristics(times, predecessors, successors, n, cycle_time)

    # Evaluate all heuristics quickly
    best_priority = None
    best_score = float('inf')
    scored = []
    for h in heuristics:
        if time.time() > deadline - 40:
            break
        score = _greedy_balance(times, predecessors, cycle_time, h)
        scored.append((score, h))
        if score < best_score:
            best_score = score
            best_priority = h.copy()
        if best_score <= 1:
            break

    if best_priority is None:
        best_priority = heuristics[0]
        best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)

    # Fast local search from the best heuristic
    if time.time() < deadline - 15:
        best_priority, best_score = _fast_local_search(
            best_priority, times, predecessors, cycle_time, n, deadline - 10, max_evals=8000
        )

    # Polish pass
    if time.time() < deadline - 5 and best_score > 1:
        best_priority, best_score = _polish(
            best_priority, times, predecessors, cycle_time, n, deadline - 3
        )

    # If time remains and best > 1, try local search from 2nd best heuristic
    if time.time() < deadline - 8 and best_score > 1 and len(scored) >= 2:
        scored_sorted = sorted(scored, key=lambda x: x[0])
        for sc, h in scored_sorted[1:3]:
            if time.time() > deadline - 5:
                break
            if sc >= best_score:
                break
            pri, sc2 = _fast_local_search(
                h, times, predecessors, cycle_time, n, deadline - 4, max_evals=3000
            )
            if sc2 < best_score:
                best_score = sc2
                best_priority = pri
            if time.time() < deadline - 4 and best_score > 1:
                best_priority, best_score = _polish(
                    best_priority, times, predecessors, cycle_time, n, deadline - 3
                )

    return best_priority
# EVOLVE-BLOCK-END
