"""Rewritten QAP solver: spectral init + optimized local search with first-improvement full scan."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization for a strong starting point.
    2. Hungarian linearization for initial refinement.
    3. Fast local search with first-improvement full scan (all pairs).
    4. Multiple Hungarian refinement rounds interleaved with local search.
    5. Final high-precision polish pass.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    # --- Compute cost efficiently ---
    def compute_cost(p):
        B_p = b[np.ix_(p, p)]
        return int(np.sum(a * B_p))
    
    # --- Build precomputed arrays for fast swap delta ---
    def build_precomp(p):
        """Build precomputed arrays for fast swap delta."""
        b_col_p = b[:, p]  # shape (n, n): [l, k] = b[l, p[k]]
        row_cost = a @ b_col_p  # shape (n, n): [i, l] = sum_k a[i,k]*b[l,p[k]]
        
        b_row_p = b[p, :]  # shape (n, n): [k, l] = b[p[k], l]
        col_cost = a.T @ b_row_p  # shape (n, n): [i, l] = sum_k a[k,i]*b[p[k],l]
        
        return row_cost, col_cost
    
    # --- Optimized swap delta: O(n) but with minimal overhead ---
    def swap_delta(i, j, p, row_cost, a, b):
        """Compute delta for swapping positions i and j."""
        u = p[i]
        v = p[j]
        if u == v:
            return 0.0
        
        # Direct formula:
        # delta = row_cost[i,v] - row_cost[i,u] + row_cost[j,u] - row_cost[j,v]
        #       + (a[i,i]-a[i,j])*(b[v,v]-b[v,u]) + (a[j,i]-a[j,j])*(b[u,v]-b[u,u])
        #       + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        
        d = (row_cost[i, v] - row_cost[i, u] + row_cost[j, u] - row_cost[j, v]
             + (a[i, i] - a[i, j]) * (b[v, v] - b[v, u])
             + (a[j, i] - a[j, j]) * (b[u, v] - b[u, u]))
        
        # S = sum_k (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u]) minus k=i,j terms
        diff_a = a[:, i] - a[:, j]
        diff_b = b[p, v] - b[p, u]
        S = float(np.dot(diff_a, diff_b))
        S -= (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
        S -= (a[j, i] - a[j, j]) * (b[v, u] - b[v, v])
        d += S
        
        return d
    
    # --- Spectral initialization ---
    def spectral_init(a, b):
        """Use spectral decomposition to get initial permutation."""
        # Center the matrices
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        
        # Get top eigenvectors
        try:
            _, vecs_a = np.linalg.eigh(a_centered)
            _, vecs_b = np.linalg.eigh(b_centered)
            
            # Use top k eigenvectors (k=10 for n=150)
            k = min(10, n)
            A_top = vecs_a[:, -k:]  # top k eigenvectors
            B_top = vecs_b[:, -k:]  # top k eigenvectors
            
            # Match by maximizing trace
            # C[i,j] = sum_l A_top[i,l] * B_top[j,l]
            C = A_top @ B_top.T
            
            row_ind, col_ind = linear_sum_assignment(-C)
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            return p
        except:
            # Fallback to rank matching
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    # --- Hungarian refinement ---
    def hungarian_refine(p_cur, cost_cur):
        """Apply one round of Hungarian linearization refinement."""
        b_col_p = b[:, p_cur]
        b_row_p = b[p_cur, :]
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        cost_new = compute_cost(p_new)
        if cost_new < cost_cur:
            return p_new, cost_new
        return p_cur, cost_cur
    
    # --- Local search with first-improvement full scan ---
    def local_search_full_scan(p_cur, cost_cur, time_limit):
        """First-improvement local search: scan all pairs, accept first improvement."""
        row_cost, col_cost = build_precomp(p_cur)
        
        while time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() > time_limit:
                        break
                    d = swap_delta(i, j, p_cur, row_cost, a, b)
                    if d < -1e-9:
                        u = p_cur[i]
                        v = p_cur[j]
                        p_cur[i], p_cur[j] = v, u
                        cost_cur += d
                        improved = True
                        
                        # Update precomputed arrays
                        da = a[:, i] - a[:, j]
                        db = b[:, v] - b[:, u]
                        row_cost += np.outer(da, db)
                        
                        dc = a[:, i] - a[:, j]
                        dl = b[v, :] - b[u, :]
                        col_cost += np.outer(dc, dl)
                        
                        # Restart scan from beginning after improvement
                        break
                if improved:
                    break
            if not improved:
                break
        
        return p_cur, cost_cur
    
    # --- Local search with random sampling (for later stages) ---
    def local_search_random(p_cur, cost_cur, time_limit):
        """Random-sampling local search for polishing."""
        row_cost, col_cost = build_precomp(p_cur)
        
        while time.time() < time_limit:
            improved = False
            for _ in range(20000):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta(i, j, p_cur, row_cost, a, b)
                if d < -1e-9:
                    u = p_cur[i]
                    v = p_cur[j]
                    p_cur[i], p_cur[j] = v, u
                    cost_cur += d
                    improved = True
                    
                    da = a[:, i] - a[:, j]
                    db = b[:, v] - b[:, u]
                    row_cost += np.outer(da, db)
                    
                    dc = a[:, i] - a[:, j]
                    dl = b[v, :] - b[u, :]
                    col_cost += np.outer(dc, dl)
            
            if not improved:
                break
        
        return p_cur, cost_cur
    
    # --- Initial solution: spectral ---
    best_p = spectral_init(a, b)
    best_cost = compute_cost(best_p)
    
    # --- Hungarian refinement ---
    if time.time() < deadline - 5.0:
        best_p, best_cost = hungarian_refine(best_p, best_cost)
    
    # --- Phase 1: Full-scan local search ---
    if time.time() < deadline - 40.0:
        best_p, best_cost = local_search_full_scan(best_p, best_cost, deadline - 40.0)
    
    # --- Phase 2: Hungarian + random local search iterations ---
    for round_num in range(8):
        if time.time() > deadline - 15.0:
            break
        best_p, best_cost = hungarian_refine(best_p, best_cost)
        if time.time() > deadline - 10.0:
            break
        best_p, best_cost = local_search_random(best_p, best_cost, deadline - 10.0)
    
    # --- Phase 3: Final full-scan polish ---
    if time.time() < deadline - 5.0:
        best_p, best_cost = local_search_full_scan(best_p, best_cost, deadline - 5.0)
    
    # --- Final Hungarian ---
    if time.time() < deadline - 1.0:
        best_p, best_cost = hungarian_refine(best_p, best_cost)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
