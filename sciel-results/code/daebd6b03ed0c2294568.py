import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0

    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))

    # Fast delta computation without mask allocation
    def fast_delta(p, i, j):
        pi, pj = p[i], p[j]
        d1 = b[pj] - b[pi]
        d2 = b[:, pj] - b[:, pi]
        s1 = (a[i] - a[j]) @ d1
        s2 = (a[:, i] - a[:, j]) @ d2
        # Remove k=i,j from s1+s2 and add correct boundary terms
        s1_ik = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
        s1_jk = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
        s2_ik = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
        s2_jk = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
        correct = (a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
                   + a[i, i] * (b[pj, pj] - b[pi, pi]) + a[j, j] * (b[pi, pi] - b[pj, pj]))
        return s1 + s2 - s1_ik - s1_jk - s2_ik - s2_jk + correct

    # Precompute
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    is_sym = np.allclose(b, b.T)
    A_sym = (a + a.T) if is_sym else None

    # Rank-based initialization
    def rank_init():
        heavy_order = np.argsort(-a_sums, kind='stable')
        central_order = np.argsort(b_sums, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p

    # Greedy initialization
    def greedy_init():
        p = np.full(n, -1, dtype=np.int64)
        order = np.argsort(-a_sums, kind='stable')
        for idx in order:
            scores = np.zeros(n, dtype=np.float64)
            for k in range(n):
                if p[k] >= 0:
                    scores -= a[idx, k] * b[p[k]] + a[k, idx] * b[:, p[k]]
            scores += 0.01 * b_sums  # tie-break toward central
            loc = int(np.argmin(scores))
            p[idx] = loc
        return p

    # Steepest descent 2-opt local search
    def local_search(p, current_cost, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    return p, current_cost
                best_j = -1
                best_d = 0
                for j in range(i + 1, n):
                    d = fast_delta(p, i, j)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0 and best_d < 0:
                    p[i], p[best_j] = p[best_j], p[i]
                    current_cost += best_d
                    improved = True
        return p, current_cost

    # Simulated annealing
    def simulated_annealing(p, current_cost, time_limit, T_init=200.0, T_end=0.5):
        remaining = time_limit - time.time()
        if remaining <= 0.1:
            return p, current_cost
        est_steps = max(int(remaining * 15000), 100)
        alpha = (T_end / T_init) ** (1.0 / est_steps)
        T = T_init
        best_p = p.copy()
        best_cost = current_cost
        while time.time() < time_limit:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            d = fast_delta(p, i, j)
            if d < 0 or (d > 0 and np.random.rand() < np.exp(-d / T)):
                p[i], p[j] = p[j], p[i]
                current_cost += d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            T = max(T * alpha, T_end)
        return best_p, best_cost

    # Multi-start: try both initializations
    best_p = None
    best_cost = np.inf

    # Start 1: rank init
    p1 = rank_init()
    cost1 = calc_cost(p1)
    best_p = p1.copy()
    best_cost = cost1

    # Start 2: greedy init
    p2 = greedy_init()
    cost2 = calc_cost(p2)
    if cost2 < best_cost:
        best_cost = cost2
        best_p = p2.copy()

    # Hungarian refinement on best
    hung_deadline = start_time + 30.0
    p = best_p.copy()
    current_cost = best_cost
    hung_stalls = 0
    while time.time() < hung_deadline:
        B_p = b[p]
        if is_sym:
            C = A_sym @ B_p
        else:
            C = a @ B_p.T + a.T @ B_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        new_cost = calc_cost(p_new)
        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            hung_stalls = 0
        else:
            hung_stalls += 1
            if hung_stalls > 4:
                break
            # Perturb: try random improving swaps
            found = False
            for _ in range(50):
                i, j = np.random.choice(n, 2, replace=False)
                d = fast_delta(p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += d
                    found = True
                    break
            if not found:
                # Random subset shuffle
                subset = np.random.choice(n, min(8, n), replace=False)
                p_perm = p.copy()
                p_perm[subset] = np.random.permutation(p_perm[subset])
                nc = calc_cost(p_perm)
                if nc < current_cost:
                    p = p_perm
                    current_cost = nc

    # Local search on best
    ls_deadline = start_time + 55.0
    p_ls, cost_ls = local_search(best_p.copy(), best_cost, ls_deadline)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls

    # SA
    sa_deadline = start_time + 100.0
    p_sa, cost_sa = simulated_annealing(best_p.copy(), best_cost, sa_deadline)
    if cost_sa < best_cost:
        best_cost = cost_sa
        best_p = p_sa

    # Final polish
    p_final, cost_final = local_search(best_p.copy(), best_cost, deadline - 2.0)
    if cost_final < best_cost:
        best_cost = cost_final
        best_p = p_final

    return best_p.tolist()
