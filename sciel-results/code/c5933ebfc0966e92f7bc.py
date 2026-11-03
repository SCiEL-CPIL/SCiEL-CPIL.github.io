import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    # 1. Initialization: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    def get_cost(p_arr):
        b_perm = b[np.ix_(p_arr, p_arr)]
        return np.sum(a * b_perm)

    current_cost = get_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    start_time = time.time()
    deadline = start_time + 118.0

    def best_improvement_2opt(p_arr, current_cost_val, deadline_t):
        """Run best-improvement 2-opt passes until no improvement or time up."""
        while time.time() < deadline_t:
            best_delta = 0
            best_i = -1
            best_j = -1
            for i in range(n):
                if time.time() > deadline_t:
                    break
                for j in range(i + 1, n):
                    u = p_arr[i]
                    v = p_arr[j]
                    diff_a = a[i, :] - a[j, :]
                    d = b[v, p_arr] - b[u, p_arr]
                    dot_val = np.dot(diff_a, d)
                    interaction = a[i, j] * (b[v, u] - b[u, v])
                    delta = dot_val - diff_a[i] * d[i] - diff_a[j] * d[j] + interaction
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_i < 0:
                return p_arr, current_cost_val, True
            p_arr[best_i], p_arr[best_j] = p_arr[best_j], p_arr[best_i]
            current_cost_val += best_delta
        return p_arr, current_cost_val, False

    # Phase 1: Best-improvement 2-opt from rank-matching init
    p, current_cost, converged = best_improvement_2opt(p, current_cost, deadline)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # Phase 2: Randomized restarts if time remains
    remaining = deadline - time.time()
    if remaining > 5.0:
        rng = random.Random(42)
        max_restarts = max(1, min(int(remaining / 8.0), 5))
        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
            p_r = best_p.copy()
            k_swaps = n // 5
            for _ in range(k_swaps):
                i1 = rng.randint(0, n - 1)
                i2 = rng.randint(0, n - 1)
                if i1 != i2:
                    p_r[i1], p_r[i2] = p_r[i2], p_r[i1]
            cost_r = get_cost(p_r)
            p_r, cost_r, _ = best_improvement_2opt(p_r, cost_r, deadline - 2.0)
            if cost_r < best_cost:
                best_cost = cost_r
                best_p = p_r.copy()

    return best_p.tolist()
