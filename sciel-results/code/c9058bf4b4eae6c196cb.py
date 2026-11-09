"""Optimize vertex ordering for the fixed greedy MIS decoder via annealing."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _greedy_size(adjacency, order):
    """Size of independent set produced by greedy decode in the given order."""
    n = len(adjacency)
    picked = 0
    used = np.zeros(n, dtype=bool)
    for v in order:
        if used[v]:
            continue
        picked += 1
        used[v] = True
        used[adjacency[v]] = True
    return picked


def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0.0])

    deadline = time.time() + 140.0  # 10s safety margin before 150s kill

    # Build an adjacency boolean matrix view
    adj = adjacency
    # Precompute neighbor masks as boolean rows for fast marking
    # (adjacency is already a bool matrix per contract)

    # Initial ordering: lowest degree first (good heuristic seed)
    deg = adj.sum(axis=1).astype(int)
    order = np.argsort(deg)  # ascending degree = highest priority first
    best_size = _greedy_size(adj, order)
    best_order = order.copy()

    # Random restarts / annealing on the score vector (permutation)
    # We operate on an ordering (permutation). score[v] = rank position.
    # Convert ordering -> score: score[v] = index in order (lower = picked first)
    def order_to_score(order):
        score = np.empty(n, dtype=float)
        score[order] = np.arange(n, dtype=float)
        return score

    current_order = best_order.copy()
    current_size = best_size

    T = 2.0
    T_min = 0.05
    cool = 0.9995

    while time.time() < deadline - 2.0:  # leave 2s for polish + return
        # Propose a swap of two positions
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        current_order[i], current_order[j] = current_order[j], current_order[i]
        new_size = _greedy_size(adj, current_order)
        delta = new_size - current_size
        if delta > 0 or random.random() < np.exp(delta / T):
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_order = current_order.copy()
        else:
            current_order[i], current_order[j] = current_order[j], current_order[i]

        T = max(T * cool, T_min)
        # Occasionally reheat slightly to escape local optima
        if T == T_min and random.random() < 0.01:
            T = 0.5

    # Final polish: deterministic hill-climbing on single swaps (tighter)
    if time.time() < deadline - 0.5:
        cur = best_order.copy()
        cur_size = best_size
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 0.5:
                    break
                for j in range(i + 1, n):
                    cur[i], cur[j] = cur[j], cur[i]
                    s = _greedy_size(adj, cur)
                    if s > cur_size:
                        cur_size = s
                        improved = True
                    else:
                        cur[i], cur[j] = cur[j], cur[i]
        if cur_size > best_size:
            best_size = cur_size
            best_order = cur

    return order_to_score(best_order)
# EVOLVE-BLOCK-END
