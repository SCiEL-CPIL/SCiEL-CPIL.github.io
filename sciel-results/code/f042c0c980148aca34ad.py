"""QAP tai256c solver: iterated 2-opt local search with Hungarian re-optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 120s

    def compute_cost(p):
        # p[i] = location of facility i
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        return int(np.sum(a * b[np.ix_(p, p)]))

    # Fast O(n) delta for swapping p[u] and p[v]
    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]
        
        # d_uv[k] = b[pv, k] - b[pu, k]
        # d_vu[k] = b[k, pv] - b[k, pu]
        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]
        
        # s1 = sum_k (a[u,k] - a[v,k]) * d_uv[p[k]]
        # s2 = sum_k (a[k,u] - a[k,v]) * d_vu[p[k]]
        
        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])
        
        # Subtract k=u and k=v terms
        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]
        
        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]
        
        # Cross terms
        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])
        
        return s1 + s2 + cross

    # Initial solution: rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # Local search using fast delta
    def local_search(p, cost):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def hungarian_reopt(p, cost):
        """
        Reassign all facilities using Hungarian algorithm on a linearized cost matrix.
        The cost of assigning facility i to location l is approximated by:
        c[i, l] = sum_j a[i, j] * b[l, p[j]]
        This is the marginal cost of placing facility i at location l given current assignments of others.
        """
        # C[i, l] = sum_j a[i, j] * b[l, p[j]]
        # This can be computed as: C = a @ b[:, p]
        # But b[:, p] is b with columns permuted by p.
        # b[:, p] has shape (n, n), where column j is b[:, p[j]].
        # So C = a @ b[:, p]
        
        # Let's verify:
        # C[i, l] = sum_j a[i, j] * b[l, p[j]]
        # (a @ b[:, p])[i, l] = sum_j a[i, j] * (b[:, p])[j, l] = sum_j a[i, j] * b[j, p[l]]
        # This is not quite right. We want b[l, p[j]], not b[j, p[l]].
        
        # Let's re-derive.
        # We want a matrix C such that C[i, l] = sum_j a[i, j] * b[l, p[j]].
        # Let M = b[p, :]. Then M[j, l] = b[p[j], l].
        # We want b[l, p[j]]. Note that b is not necessarily symmetric.
        # So M[j, l] is b[p[j], l]. We want b[l, p[j]].
        # If b is symmetric, b[l, p[j]] = b[p[j], l] = M[j, l].
        # In general, we need to handle the asymmetry.
        
        # Let's compute C directly for clarity, though it's O(n^3).
        # C[i, l] = sum_j a[i, j] * b[l, p[j]]
        
        # Actually, we can compute this as:
        # C = a @ b[p, :].T ? No.
        # Let's look at the indices.
        # (a @ X)[i, l] = sum_j a[i, j] * X[j, l].
        # We want sum_j a[i, j] * b[l, p[j]].
        # So X[j, l] should be b[l, p[j]].
        # This means X = b.T[p, :].T ?
        # b.T[j, l] = b[l, j].
        # b.T[p, :].T [j, l] = b.T[p[j], l] = b[l, p[j]].
        # Yes! So X = b.T[p, :].T
        # Wait, b.T[p, :] selects rows p from b.T.
        # b.T[p, :] has shape (n, n). Element (j, l) is b.T[p[j], l] = b[l, p[j]].
        # So X = b.T[p, :].
        # Then C = a @ X = a @ b.T[p, :].
        
        C = a @ b.T[p, :]
        
        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        # Check if new_p is a permutation
        if not np.array_equal(np.sort(new_p), np.arange(n)):
            return p, cost
            
        new_cost = compute_cost(new_p)
        
        # Only accept if better (or equal, but strictly better to avoid loops)
        if new_cost < cost:
            return new_p, new_cost
        else:
            return p, cost

    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Iterated Local Search with Hungarian Re-optimization
    while time.time() < deadline:
        # Perturb: random swaps
        p = best_p.copy()
        n_swaps = np.random.randint(2, 8)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        
        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))
        
        # Hungarian Re-optimization
        p, cost = hungarian_reopt(p, cost)
        
        # Polish again after Hungarian
        p, cost = local_search(p, cost)
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    return best_p.tolist()
# EVOLVE-BLOCK-END
