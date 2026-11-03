"""QAP tai256c solver: iterated 2-opt local search with fast cost evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin under 120s

    # Precompute: a and b are int64 arrays of shape (n, n)
    # We'll work with them directly for the fast swap delta.

    def compute_cost(p):
        # p[i] = location of facility i
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Use np.ix_ for efficient indexing
        return int(np.sum(a * b[np.ix_(p, p)]))

    # Fast O(n) delta for swapping p[u] and p[v]
    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]
        
        # d_uv[k] = b[pv, k] - b[pu, k]  (row difference)
        # d_vu[k] = b[k, pv] - b[k, pu]  (col difference)
        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]
        
        # s1 = sum_k (a[u,k] - a[v,k]) * d_uv[p[k]]
        # s2 = sum_k (a[k,u] - a[k,v]) * d_vu[p[k]]
        
        # We need to be careful with the terms where k=u and k=v because p[k] changes.
        # The standard derivation gives:
        # delta = sum_{k != u, v} (a[u,k] - a[v,k]) * (b[pv, p[k]] - b[pu, p[k]])
        #       + sum_{k != u, v} (a[k,u] - a[k,v]) * (b[p[k], pv] - b[p[k], pu])
        #       + (a[u,v] - a[v,u]) * (b[pv, pu] - b[pu, pv])
        #       + a[u,u] * (b[pv, pv] - b[pu, pu])
        #       + a[v,v] * (b[pu, pu] - b[pv, pv])
        #       + ... actually let's use the dot product form and subtract the u,v terms.
        
        # Let's use the formula:
        # delta = np.dot(a[u, :] - a[v, :], d_uv[p]) + np.dot(a[:, u] - a[:, v], d_vu[p])
        #         - correction terms for k=u and k=v in both sums
        #         + cross term for (u,v) and (v,u)
        
        # Actually, a cleaner way:
        # The change in cost when swapping locations pu and pv for facilities u and v is:
        # For all k not in {u,v}:
        #   a[u,k] changes from b[pu, p[k]] to b[pv, p[k]]: delta += a[u,k] * (b[pv, p[k]] - b[pu, p[k]])
        #   a[k,u] changes from b[p[k], pu] to b[p[k], pv]: delta += a[k,u] * (b[p[k], pv] - b[p[k], pu])
        #   a[v,k] changes from b[pv, p[k]] to b[pu, p[k]]: delta += a[v,k] * (b[pu, p[k]] - b[pv, p[k]])
        #   a[k,v] changes from b[p[k], pv] to b[p[k], pu]: delta += a[k,v] * (b[p[k], pu] - b[p[k], pv])
        # For k=u:
        #   a[u,u] stays at b[pu,pu] -> becomes b[pv,pv]: delta += a[u,u] * (b[pv,pv] - b[pu,pu])
        #   a[u,v] stays at b[pu,pv] -> becomes b[pv,pu]: delta += a[u,v] * (b[pv,pu] - b[pu,pv])
        # For k=v:
        #   a[v,u] stays at b[pv,pu] -> becomes b[pu,pv]: delta += a[v,u] * (b[pu,pv] - b[pv,pu])
        #   a[v,v] stays at b[pv,pv] -> becomes b[pu,pu]: delta += a[v,v] * (b[pu,pu] - b[pv,pv])
        
        # Combining:
        # delta = sum_{k != u,v} (a[u,k] - a[v,k]) * (b[pv, p[k]] - b[pu, p[k]])
        #       + sum_{k != u,v} (a[k,u] - a[k,v]) * (b[p[k], pv] - b[p[k], pu])
        #       + a[u,u] * (b[pv,pv] - b[pu,pu]) + a[v,v] * (b[pu,pu] - b[pv,pv])
        #       + a[u,v] * (b[pv,pu] - b[pu,pv]) + a[v,u] * (b[pu,pv] - b[pv,pu])
        
        # The last two terms: (a[u,v] - a[v,u]) * (b[pv,pu] - b[pu,pv])
        # The a[u,u] and a[v,v] terms cancel: a[u,u]*(b[pv,pv]-b[pu,pu]) + a[v,v]*(b[pu,pu]-b[pv,pv])
        # = (a[u,u] - a[v,v]) * (b[pv,pv] - b[pu,pu])
        
        # So:
        # delta = sum_{k != u,v} (a[u,k] - a[v,k]) * d_uv[p[k]]
        #       + sum_{k != u,v} (a[k,u] - a[k,v]) * d_vu[p[k]]
        #       + (a[u,u] - a[v,v]) * (b[pv,pv] - b[pu,pu])
        #       + (a[u,v] - a[v,u]) * (b[pv,pu] - b[pu,pv])
        
        # Using dot products and subtracting k=u, v terms:
        # sum_{k != u,v} (a[u,k] - a[v,k]) * d_uv[p[k]]
        # = np.dot(a[u, :] - a[v, :], d_uv[p]) - (a[u,u] - a[v,u]) * d_uv[pu] - (a[u,v] - a[v,v]) * d_uv[pv]
        
        # sum_{k != u,v} (a[k,u] - a[k,v]) * d_vu[p[k]]
        # = np.dot(a[:, u] - a[:, v], d_vu[p]) - (a[u,u] - a[u,v]) * d_vu[pu] - (a[v,u] - a[v,v]) * d_vu[pv]
        
        s1_full = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2_full = np.dot(a[:, u] - a[:, v], d_vu[p])
        
        s1 = s1_full - (a[u, u] - a[v, u]) * d_uv[pu] - (a[u, v] - a[v, v]) * d_uv[pv]
        s2 = s2_full - (a[u, u] - a[u, v]) * d_vu[pu] - (a[v, u] - a[v, v]) * d_vu[pv]
        
        diag_term = (a[u, u] - a[v, v]) * (b[pv, pv] - b[pu, pu])
        cross_term = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])
        
        return int(s1 + s2 + diag_term + cross_term)

    # Initial solution: greedy construction
    # Sort facilities by sum of a (row sums) and locations by sum of b (col sums)
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # Local search using fast delta
    def local_search(p, cost, max_passes=50):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
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
                        # Restart from beginning to ensure convergence
                        break
                if improved:
                    break
        return p, cost

    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Iterated Local Search with adaptive perturbation
    rng = np.random.default_rng(42)
    no_improve_count = 0
    max_no_improve = 20

    while time.time() < deadline:
        # Perturb: random swaps
        p = best_p.copy()
        
        # Adaptive perturbation strength
        if no_improve_count < 5:
            n_swaps = rng.integers(2, 6)
        elif no_improve_count < 10:
            n_swaps = rng.integers(4, 10)
        elif no_improve_count < 15:
            n_swaps = rng.integers(6, 14)
        else:
            n_swaps = rng.integers(8, 20)
        
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        
        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            # If stuck for a long time, try a larger perturbation or restart
            if no_improve_count >= max_no_improve:
                # Try a completely different initial solution
                heavy2 = np.argsort(-a_row_sums + rng.normal(0, a_row_sums.std(), n), kind="stable")
                central2 = np.argsort(b_col_sums + rng.normal(0, b_col_sums.std(), n), kind="stable")
                p_new = np.empty(n, dtype=np.int64)
                p_new[heavy2] = central2
                p_new, cost_new = local_search(p_new, compute_cost(p_new))
                if cost_new < best_cost:
                    best_p = p_new.copy()
                    best_cost = cost_new
                    no_improve_count = 0
                else:
                    no_improve_count = max_no_improve // 2

    return best_p.tolist()
# EVOLVE-BLOCK-END
