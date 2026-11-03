"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 140.0
    
    # --- Spectral initialization ---
    try:
        U, s, Vt = np.linalg.svd(b, full_matrices=False)
        b_eig = Vt[0]
    except:
        b_eig = b.sum(axis=1)
    
    loc_order = np.argsort(-b_eig, kind="stable")
    a_row_sum = a.sum(axis=1)
    fac_order = np.argsort(-a_row_sum, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    for i, f in enumerate(fac_order):
        p[f] = loc_order[i]
    
    def compute_cost(p):
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)
    
    best_cost = compute_cost(p)
    best_p = p.copy()
    
    # --- Local search with time-limited passes ---
    def local_search(p, current_cost, max_time, max_passes=50):
        for _ in range(max_passes):
            if time.time() > max_time:
                break
            improved = False
            for u in range(n - 1):
                if time.time() > max_time:
                    break
                lu = p[u]
                for v in range(u + 1, n):
                    if time.time() > max_time:
                        break
                    lv = p[v]
                    if lu == lv:
                        continue
                    
                    diff_b = b[lv, :] - b[lu, :]
                    diff_bT = b[:, lv] - b[:, lu]
                    
                    S1 = np.dot(a[u, :], diff_b[p])
                    S2 = np.dot(a[v, :], diff_b[p])
                    S3 = np.dot(a[:, u], diff_bT[p])
                    S4 = np.dot(a[:, v], diff_bT[p])
                    
                    cross_term = a[u, v] * (b[lv, lu] - b[lu, lv]) + a[v, u] * (b[lu, lv] - b[lv, lu])
                    delta = (S1 - S2) + (S3 - S4) + cross_term
                    
                    if delta < -1e-9:
                        p[u], p[v] = p[v], p[u]
                        current_cost += delta
                        improved = True
            
            if not improved:
                break
        
        return p, current_cost
    
    p, current_cost = local_search(p, best_cost, deadline - 5)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # --- Iterated local search ---
    while time.time() < deadline - 10:
        p = best_p.copy()
        
        n_perturb = max(2, n // 15)
        for _ in range(n_perturb):
            u = np.random.randint(n)
            v = np.random.randint(n)
            if u != v:
                p[u], p[v] = p[v], p[u]
        
        pert_cost = compute_cost(p)
        p, pert_cost = local_search(p, pert_cost, deadline - 5)
        
        if pert_cost < best_cost:
            best_cost = pert_cost
            best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
