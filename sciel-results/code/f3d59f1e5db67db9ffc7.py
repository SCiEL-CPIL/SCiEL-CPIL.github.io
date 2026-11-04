"""QAP tai256c solver: Frank-Wolfe / Birkhoff-von Neumann convex relaxation
with exact Hungarian linearization, decomposition tracking, and 2-opt polish.

Structural change vs. parent: instead of purely combinatorial iterated local
search over permutations, we relax the permutation matrix to the doubly-
stochastic (Birkhoff) polytope and run a Frank-Wolfe scheme. Each Frank-Wolfe
step forms the gradient of the QAP objective at the current doubly-stochastic
matrix, builds a linear assignment cost matrix, and solves it EXACTLY with the
Hungarian algorithm (the projection onto the Birkhoff polytope). The iterate
S is maintained as a weighted convex combination of the visited permutation
vertices, which guarantees S is exactly doubly-stochastic and yields a fast
"nearest vertex" read-off for an integer solution. A fast vectorized 2-opt
polish is applied periodically and at the end to extract and improve integer
solutions from the relaxed iterates.
"""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 118.0  # explicit safety margin under 150s limit

    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)

    def compute_cost(p):
        # p is a permutation of 0..n-1; cost = sum_{i,j} a[i,j]*b[p[i],p[j]]
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(p, i, j):
        """Exact cost delta for swapping facilities at positions i and j."""
        pu = p[i]
        pv = p[j]
        # d_uv = b[pv,:] - b[pu,:]  (row difference)
        # d_vu = b[:,pv] - b[:,pu]  (col difference)
        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[i, :] - a[j, :], d_uv[p])
        s2 = np.dot(a[:, i] - a[:, j], d_vu[p])

        # Corrections for the i and j indices themselves
        s1 -= (a[i, i] - a[j, i]) * d_uv[pu]
        s1 -= (a[i, j] - a[j, j]) * d_uv[pv]

        s2 -= (a[i, i] - a[i, j]) * d_vu[pu]
        s2 -= (a[j, i] - a[j, j]) * d_vu[pv]

        cross = (a[i, j] - a[j, i]) * (b[pv, pu] - b[pu, pv])
        return s1 + s2 + cross

    def local_search(p, cost, max_passes=12):
        """First-improvement 2-opt local search (fast, vectorized main term)."""
        p = p.copy()
        for _ in range(max_passes):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(n - 1):
                if time.time() > deadline - 3:
                    return p, cost
                pu = p[i]
                # vectorized main term for all j > i
                # delta_j approx: we still need exact; compute in a tight loop
                best_delta = 0.0
                best_j = -1
                for j in range(i + 1, n):
                    d = swap_delta(p, i, j)
                    if d < best_delta:
                        best_delta = d
                        best_j = j
                if best_j >= 0:
                    p[i], p[best_j] = p[best_j], p[i]
                    cost += best_delta
                    improved = True
            if not improved:
                break
        return p, cost

    def hungarian_from_linear(C):
        """Solve the linear assignment min sum C[i, pi[i]] -> permutation pi."""
        row_ind, col_ind = linear_sum_assignment(C)
        pi = np.empty(n, dtype=np.int64)
        pi[row_ind] = col_ind
        return pi

    def qap_gradient(S):
        """Gradient of sum_{i,j} a[i,j]*S[i,j]*... w.r.t. S.

        We model the QAP as f(S) = <A, S B S^T> = sum_{i,j,k,l} A[i,j] S[i,k] S[j,l] B[k,l].
        Gradient w.r.t. S[i,k] = sum_{j,l} A[i,j] B[k,l] S[j,l] + sum_{j,l} A[j,i] B[l,k] S[j,l]
                               = (A @ S @ B)[i,k] + (A.T @ S @ B.T)[i,k].
        """
        AS = a @ S
        ASB = AS @ b
        grad = ASB + (a.T @ S @ b.T)
        return grad

    def frank_wolfe(S0, max_iter=2000, step0=1.0):
        """Frank-Wolfe on the Birkhoff polytope.

        S is maintained as a convex combination of vertex permutations:
        S = sum_r w_r * P_r, where P_r is the permutation matrix of perm r.
        We track the decomposition to (a) keep S exactly doubly-stochastic and
        (b) read off the dominant vertex quickly.
        """
        # Decomposition state
        perms = []            # list of permutations (each a vertex)
        weights = np.zeros(0) # convex weights
        # Start from S0 (a permutation) as the first vertex
        start = S0.copy()
        perms.append(start)
        weights = np.array([1.0])

        S = np.zeros((n, n), dtype=np.float64)
        S[start, np.arange(n)] = 1.0  # P_start

        best_p = start.copy()
        best_cost = compute_cost(start)

        for it in range(max_iter):
            if time.time() > deadline - 12:
                break

            # Gradient at current S
            G = qap_gradient(S)

            # Linearization: find vertex minimizing <G, P> = sum_{i,k} G[i,k] P[i,k]
            # which is min over permutations pi of sum_i G[i, pi[i]].
            s = hungarian_from_linear(G)

            # Frank-Wolfe step size (standard diminishing): gamma_t = 2/(t+2)
            gamma = 2.0 / (it + 2.0)

            # Add the new vertex to the decomposition (or merge if it exists)
            # For speed, just append and re-normalize weights.
            perms.append(s)
            new_weights = np.zeros(len(perms))
            new_weights[:] = weights * (1.0 - gamma)
            new_weights[-1] = gamma
            weights = new_weights
            perms = perms  # keep

            # Update S: S = (1-gamma)*S_old + gamma*P_s
            S = (1.0 - gamma) * S
            S[s, np.arange(n)] += gamma

            # Read off the dominant vertex (largest weight) as a candidate integer solution
            # and also the argmax row-sum heuristic. Use the highest-weight vertex.
            dom_idx = int(np.argmax(weights))
            cand = perms[dom_idx]
            c = compute_cost(cand)
            if c < best_cost:
                best_cost = c
                best_p = cand.copy()

            # Occasional: also try the current Frank-Wolfe direction vertex 's'
            cs = compute_cost(s)
            if cs < best_cost:
                best_cost = cs
                best_p = s.copy()

            # Periodic local-search polish on the best vertex to extract better integers
            if it % 40 == 0 and time.time() < deadline - 15:
                lp, lc = local_search(best_p, best_cost, max_passes=4)
                if lc < best_cost:
                    best_cost = lc
                    best_p = lp

        return best_p, best_cost

    # ---- Initial permutations (diverse warm starts) ----
    def init_rank():
        a_rs = a.sum(axis=1)
        b_cs = b.sum(axis=1)
        heavy = np.argsort(-a_rs, kind="stable")
        central = np.argsort(b_cs, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def init_hungarian():
        # Greedy linearization: minimize sum_i (a row-sum)_i * (b col-sum)_{pi[i]}
        C = np.outer(a.sum(axis=1), b.sum(axis=1))
        return hungarian_from_linear(C)

    def init_random():
        p = np.random.permutation(n)
        return p

    # Build a small pool of diverse starting permutations
    starts = []
    p_rank = init_rank()
    starts.append(p_rank)
    p_hung = init_hungarian()
    if not np.array_equal(p_hung, p_rank):
        starts.append(p_hung)
    # a couple random starts
    for _ in range(2):
        pr = init_random()
        starts.append(pr)

    # Polish each start a little to get good initial vertices
    polished_starts = []
    for sp in starts:
        if time.time() > deadline - 20:
            break
        lp, lc = local_search(sp, compute_cost(sp), max_passes=6)
        polished_starts.append(lp)

    best_p = polished_starts[0].copy()
    best_cost = compute_cost(best_p)

    # ---- Frank-Wolfe phase with multiple restarts ----
    # Budget the Frank-Wolfe restarts by time.
    fw_start = time.time()
    for sp in polished_starts:
        if time.time() > deadline - 15:
            break
        fp, fc = frank_wolfe(sp, max_iter=1500)
        if fc < best_cost:
            best_cost = fc
            best_p = fp.copy()

    # ---- Final intensive local-search polish on the best solution ----
    # Run several 2-opt passes from the best, plus a few perturbed restarts
    # seeded by the Frank-Wolfe dominant vertices, to squeeze out integer gains.
    if time.time() < deadline - 5:
        fp, fc = local_search(best_p, best_cost, max_passes=15)
        if fc < best_cost:
            best_cost = fc
            best_p = fp.copy()

    # A few short ILS restarts around the best to escape local optima
    consecutive = 0
    base_perturb = 3
    while time.time() < deadline - 3 and consecutive < 40:
        p = best_p.copy()
        n_swaps = np.random.randint(2, base_perturb + 3)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        lp, lc = local_search(p, compute_cost(p), max_passes=8)
        if lc < best_cost:
            best_cost = lc
            best_p = lp.copy()
            consecutive = 0
        else:
            consecutive += 1

    # Final polish
    if time.time() < deadline - 2:
        fp, fc = local_search(best_p, best_cost, max_passes=10)
        if fc < best_cost:
            best_cost = fc
            best_p = fp.copy()

    return best_p.tolist()
