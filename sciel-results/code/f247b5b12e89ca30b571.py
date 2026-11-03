"""QAP tai256c solver: iterated 2-opt local search with fast cost evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110.0  # safety margin under 120s

    def compute_cost(p):
        # p[i] = location of facility i
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        return np.sum(a * b[np.ix_(p, p)])

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
        
        # We need to be careful with the terms where k=u and k=v because p[k] changes.
        # However, the formula derived in the inspiration seems to handle it by subtracting those specific terms.
        
        # Let's verify the delta formula from inspiration 1:
        # delta = s1 + s2 + cross
        # where:
        # s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        # s2 = np.dot(a[:, u] - a[:, v], d_vu[p])
        # Then subtract k=u and k=v terms from s1 and s2.
        
        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])
        
        # Subtract k=u terms
        # For s1: k=u -> (a[u,u] - a[v,u]) * d_uv[p[u]] = (a[u,u] - a[v,u]) * d_uv[pu]
        # For s2: k=u -> (a[u,u] - a[u,v]) * d_vu[p[u]] = (a[u,u] - a[u,v]) * d_vu[pu]
        
        # Subtract k=v terms
        # For s1: k=v -> (a[u,v] - a[v,v]) * d_uv[p[v]] = (a[u,v] - a[v,v]) * d_uv[pv]
        # For s2: k=v -> (a[v,u] - a[v,v]) * d_vu[p[v]] = (a[v,u] - a[v,v]) * d_vu[pv]
        
        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]
        
        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]
        
        # Cross terms:
        # The terms involving both u and v in the "before" and "after" states.
        # Before: a[u,v]*b[pu,pv] + a[v,u]*b[pv,pu]
        # After:  a[u,v]*b[pv,pu] + a[v,u]*b[pu,pv]
        # Change: a[u,v]*(b[pv,pu] - b[pu,pv]) + a[v,u]*(b[pu,pv] - b[pv,pu])
        #        = (a[u,v] - a[v,u]) * (b[pv,pu] - b[pu,pv])
        
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

    # Iterated Local Search
    while time.time() < deadline:
        # Perturb: random swaps
        p = best_p.copy()
        n_swaps = np.random.randint(2, 8)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        
        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    return best_p.tolist()
# EVOLVE-BLOCK-END
