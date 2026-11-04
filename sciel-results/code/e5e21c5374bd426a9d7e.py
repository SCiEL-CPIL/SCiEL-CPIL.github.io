"""QAP tai256c solver: fast ILS with vectorized swap deltas, adaptive perturbation, and final polish."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 118.0  # safety margin under 150s limit

    # Precompute constant arrays for fast delta computation
    a_row = a.copy()
    a_col = a.T.copy()

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def fast_swap_delta(p, u, v):
        """O(n) swap delta: cost change from swapping positions u and v."""
        pu = p[u]
        pv = p[v]

        # d_uv[x] = b[pv, x] - b[pu, x]  for all x
        # d_vu[x] = b[x, pv] - b[x, pu]  for all x
        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        # s1: contribution from rows u,v of a to columns that changed
        # a[u,:]*d_uv[p] + a[v,:]*(-d_uv[p]) but adjust for self-terms
        s1 = np.dot(a_row[u, :] - a_row[v, :], d_uv[p])
        # s2: contribution from columns u,v of a to rows that changed
        s2 = np.dot(a_col[u, :] - a_col[v, :], d_vu[p])

        # Correct for the fact that positions u and v are swapped with themselves
        # The dot products above include terms where x=u or x=v, but those positions
        # now hold the other facility's location. We need to subtract the self-interactions.
        # Term at position u: was a[u,u]*b[pu,pu] + a[u,v]*b[pu,pv] + a[v,u]*b[pv,pu] + a[v,v]*b[pv,pv]
        # After swap: a[u,v]*b[pv,pv] + a[u,u]*b[pv,pu] + a[v,v]*b[pu,pu] + a[v,u]*b[pu,pv]
        # The s1+s2 formula double-counts or misses these. Let me re-derive.

        # Actually the standard O(n) formula for QAP swap delta:
        # delta = sum_k != u,v of a[u,k]*b[pv,k] + a[v,k]*b[pk,pu] + a[k,u]*b[k,pv] + a[k,v]*b[pu,k]
        #        - sum_k != u,v of a[u,k]*b[pu,k] - a[v,k]*b[pk,pv] - a[k,u]*b[k,pu] - a[k,v]*b[pv,k]
        #        + terms involving u,v cross terms

        # Let's use the well-known formula:
        # For swapping p[u]=pu and p[v]=pv:
        # delta = (a[u,v]-a[v,u])*(b[pv,pu]-b[pu,pv])
        #        + sum_{k != u,v} (a[u,k]-a[v,k])*(b[pv,k]-b[pu,k])
        #        + sum_{k != u,v} (a[k,u]-a[k,v])*(b[k,pv]-b[k,pu])

        # The first two sums can be computed as:
        # sum_k (a[u,k]-a[v,k])*(b[pv,k]-b[pu,k]) - (a[u,u]-a[v,u])*(b[pv,pu]-b[pu,pu]) - (a[u,v]-a[v,v])*(b[pv,pv]-b[pu,pv])
        # Similarly for the second.

        # Recompute more carefully:
        s1_full = np.dot(a_row[u, :] - a_row[v, :], d_uv)
        s2_full = np.dot(a_col[u, :] - a_col[v, :], d_vu)

        # Subtract the k=u and k=v terms that shouldn't be in the sum over k != u,v
        # For s1: k=u term is (a[u,u]-a[v,u])*(b[pv,pu]-b[pu,pu])
        #          k=v term is (a[u,v]-a[v,v])*(b[pv,pv]-b[pu,pv])
        s1_excl = s1_full - (a_row[u,u] - a_row[v,u]) * (b[pv,pu] - b[pu,pu]) \
                         - (a_row[u,v] - a_row[v,v]) * (b[pv,pv] - b[pu,pv])

        # For s2: k=u term is (a[u,u]-a[u,v])*(b[pu,pv]-b[pu,pu])  -> wait, d_vu[k] = b[k,pv]-b[k,pu]
        # k=u: (a[u,u]-a[u,v])*(b[u,pv]-b[u,pu])  but b is not necessarily symmetric
        # Actually a_col[u,:] = a[:,u], so a_col[u,k] = a[k,u]
        # s2 = sum_k (a[k,u]-a[k,v])*(b[k,pv]-b[k,pu])
        # k=u: (a[u,u]-a[u,v])*(b[u,pv]-b[u,pu])
        # k=v: (a[v,u]-a[v,v])*(b[v,pv]-b[v,pu])
        s2_excl = s2_full - (a_col[u,u] - a_col[u,v]) * (b[u,pv] - b[u,pu]) \
                         - (a_col[v,u] - a_col[v,v]) * (b[v,pv] - b[v,pu])

        # Cross term for the u,v interaction
        cross = (a_row[u,v] - a_row[v,u]) * (b[pv,pu] - b[pu,pv])

        return s1_excl + s2_excl + cross

    def local_search(p, cost, max_passes=50):
        """First-improvement local search with O(n) swap deltas."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1

            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += int(delta)
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def hungarian_reopt(p):
        """Linearized re-optimization via Hungarian algorithm."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2
        C = M1 + M2
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def rank_matching_init():
        """Rank-matching initialization using row/column sums."""
        a_row_sums = a.sum(axis=1)
        b_col_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_col_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def spectral_init():
        """Spectral bisection-based initialization."""
        # Use eigenvectors of a and b to create initial assignment
        try:
            w_a, v_a = np.linalg.eigh(a)
            w_b, v_b = np.linalg.eigh(b)
            # Use the top eigenvector for ordering
            order_a = np.argsort(v_a[:, -1])
            order_b = np.argsort(v_b[:, -1])
            p = np.empty(n, dtype=np.int64)
            p[order_a] = order_b
            return p
        except Exception:
            return rank_matching_init()

    # Phase 1: Multiple initial solutions
    best_p = None
    best_cost = float('inf')

    # Init 1: rank matching
    p1 = rank_matching_init()
    p1, c1 = local_search(p1.copy(), compute_cost(p1))
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()

    # Init 2: spectral
    p2 = spectral_init()
    p2, c2 = local_search(p2.copy(), compute_cost(p2))
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()

    # Init 3: Hungarian from random
    p3 = np.random.permutation(n).astype(np.int64)
    h3 = hungarian_reopt(p3)
    p3, c3 = local_search(h3, compute_cost(h3))
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()

    # Phase 2: Iterated Local Search
    current_p = best_p.copy()
    current_cost = best_cost

    # Adaptive perturbation strength
    perturbation_strength = 3
    no_improve_count = 0
    max_no_improve = 15

    iteration = 0
    while time.time() < deadline - 20:
        iteration += 1

        # Perturb: random block reversal with adaptive length
        p = current_p.copy()

        # Occasionally do a larger perturbation
        if no_improve_count > max_no_improve:
            block_len = np.random.randint(10, 25)
            no_improve_count = 0
        else:
            block_len = np.random.randint(perturbation_strength, perturbation_strength + 5)

        start = np.random.randint(0, n - block_len + 1)
        p[start:start + block_len] = p[start:start + block_len][::-1]

        # Local search from perturbed solution
        p, cost = local_search(p, compute_cost(p))

        # Periodic Hungarian re-optimization
        if iteration % 5 == 0 and time.time() < deadline - 35:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        # Accept if improved or with simulated annealing probability
        if cost < current_cost:
            current_p = p.copy()
            current_cost = cost
            no_improve_count = 0

            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                # Reduce perturbation strength on improvement
                perturbation_strength = max(2, perturbation_strength - 1)
        else:
            no_improve_count += 1
            # Increase perturbation strength when stuck
            if no_improve_count % 5 == 0:
                perturbation_strength = min(20, perturbation_strength + 1)

            # Simulated annealing acceptance
            # Simple: accept with small probability to escape
            if random.random() < 0.05:
                current_p = p.copy()
                current_cost = cost

    # Phase 3: Final polish - intensive local search
    if time.time() < deadline - 3:
        final_p = best_p.copy()
        final_cost = best_cost

        for _ in range(5):
            if time.time() > deadline - 2:
                break
            final_p, final_cost = local_search(final_p, final_cost, max_passes=80)
            if final_cost < best_cost:
                best_cost = final_cost
                best_p = final_p.copy()

        # One more Hungarian polish at the end
        if time.time() < deadline - 2:
            h_final = hungarian_reopt(best_p)
            h_final, h_final_cost = local_search(h_final, compute_cost(h_final), max_passes=80)
            if h_final_cost < best_cost:
                best_cost = h_final_cost
                best_p = h_final.copy()

    return best_p.tolist()
