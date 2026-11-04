import numpy as np
import time
import math

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 113.0  # Safety margin before 120s limit

    # Convert to float64 for faster arithmetic in local search
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        """Compute the exact integer cost of permutation p."""
        return int(np.sum(a * b[np.ix_(p, p)]))

    def mds_embedding(D, dim=2):
        """Classical MDS embedding."""
        n = D.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals = vals[:dim]
        vecs = vecs[:, :dim]
        vals = np.maximum(vals, 0)
        coords = vecs * np.sqrt(vals)
        return coords

    def procrustes_alignment(X, Y):
        """Align X to Y and solve assignment problem."""
        from scipy.optimize import linear_sum_assignment
        
        U, S, Vt = np.linalg.svd(Y.T @ X)
        Q = U @ Vt
        X_aligned = X @ Q
        
        X_norms = np.sum(X_aligned ** 2, axis=1)
        Y_norms = np.sum(Y ** 2, axis=1)
        Cross = X_aligned @ Y.T
        C = X_norms[:, None] + Y_norms[None, :] - 2 * Cross
        
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def local_search(p, current_cost, time_limit):
        """
        Deterministic 2-opt local search with incremental updates.
        Uses the simplified delta formula:
        delta = (a[i, :] - a[k, :]) @ (b[l, p] - b[j, p]) + (a[:, i] - a[:, k]) @ (b[p, l] - b[p, j])
        where j=p[i], l=p[k].
        """
        p = p.copy()
        cost = current_cost
        improved = True
        passes = 0
        max_passes = 200

        # Precompute pairs for consistent ordering (optional, but let's just iterate)
        # For speed, we iterate i < k
        
        while improved and time.time() < time_limit and passes < max_passes:
            improved = False
            passes += 1
            
            for i in range(n):
                if time.time() >= time_limit:
                    return p, cost
                for k in range(i + 1, n):
                    if time.time() >= time_limit:
                        return p, cost
                    
                    j = p[i]
                    l = p[k]
                    
                    # delta = sum_v (a[i,v] - a[k,v]) * (b[l, p[v]] - b[j, p[v]]) 
                    #       + sum_v (a[v,i] - a[v,k]) * (b[p[v], l] - b[p[v], j])
                    
                    # Vectorized computation
                    # b[l, p] is a row of b indexed by p
                    # b[j, p] is a row of b indexed by p
                    diff_b_row = b_f[l, p] - b_f[j, p]
                    diff_a_row = a_f[i, :] - a_f[k, :]
                    term1 = np.dot(diff_a_row, diff_b_row)
                    
                    diff_b_col = b_f[p, l] - b_f[p, j]
                    diff_a_col = a_f[:, i] - a_f[:, k]
                    term2 = np.dot(diff_a_col, diff_b_col)
                    
                    delta = term1 + term2
                    
                    if delta < -1e-9:
                        # Accept swap
                        p[i], p[k] = l, j
                        cost += delta
                        improved = True
                        
        return p, cost

    def perturbation(p, strength):
        """
        Apply random 2-opt swaps to perturb the solution.
        """
        p = p.copy()
        for _ in range(strength):
            i = np.random.randint(0, n)
            k = np.random.randint(0, n)
            if i != k:
                p[i], p[k] = p[k], p[i]
        return p

    # --- Phase 1: Constructive Heuristic (Rank Matching) ---
    heavy = np.argsort(-a_f.sum(axis=1), kind="stable")
    central = np.argsort(b_f.sum(axis=1), kind="stable")
    
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)
    
    best_p = p_rank.copy()
    best_cost = cost_rank

    # --- Phase 2: MDS Initialization ---
    if time.time() < deadline - 40.0:
        try:
            a_sym = (a_f + a_f.T) / 2.0
            b_sym = (b_f + b_f.T) / 2.0
            
            # Try different dimensions
            for dim in [2, 3, 5]:
                if time.time() >= deadline - 40.0:
                    break
                X = mds_embedding(a_sym, dim)
                Y = mds_embedding(b_sym, dim)
                p_mds = procrustes_alignment(X, Y)
                cost_mds = compute_cost(p_mds)
                
                if cost_mds < best_cost:
                    best_cost = cost_mds
                    best_p = p_mds.copy()
        except Exception:
            pass

    # --- Phase 3: Spectral Initialization (Hungarian on A@B.T) ---
    if time.time() < deadline - 30.0:
        try:
            from scipy.optimize import linear_sum_assignment
            M = a_f @ b_f.T
            row_ind, col_ind = linear_sum_assignment(-M)
            p_spec = np.empty(n, dtype=np.int64)
            p_spec[row_ind] = col_ind
            cost_spec = compute_cost(p_spec)
            
            if cost_spec < best_cost:
                best_cost = cost_spec
                best_p = p_spec.copy()
        except Exception:
            pass

    # --- Phase 4: Local Search from Best Initial ---
    if time.time() < deadline - 10.0:
        p_ls, cost_ls = local_search(best_p, best_cost, deadline - 10.0)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

    # --- Phase 5: Iterated Local Search (ILS) ---
    p_ils = best_p.copy()
    cost_ils = best_cost
    
    # SA parameters for ILS acceptance
    T = 1000.0
    cooling = 0.95
    min_T = 1.0
    
    if time.time() < deadline - 5.0:
        no_improve_iter = 0
        max_no_improve = 100
        rng = np.random.RandomState(42)
        
        while time.time() < deadline - 5.0 and no_improve_iter < max_no_improve:
            # Perturb
            strength = 5 + no_improve_iter // 10
            p_pert = perturbation(p_ils, strength)
            
            # Local Search
            # Use a tight time limit for the inner LS to allow multiple iterations
            # Or just let it run until the global deadline?
            # Better to let it run until global deadline minus safety, but if we do many iterations,
            # we should limit each LS.
            # Let's give it a chunk of time, e.g., 5s or until global deadline.
            # Since LS is fast, let's just use the global deadline.
            p_ls, cost_ls = local_search(p_pert, compute_cost(p_pert), deadline - 5.0)
            
            if cost_ls < best_cost:
                best_p = p_ls.copy()
                best_cost = cost_ls
                p_ils = p_ls.copy()
                no_improve_iter = 0
                T = 1000.0
            else:
                # Simulated Annealing acceptance
                delta_cost = cost_ls - best_cost
                if T > min_T:
                    # Use log space for stability if delta is large, but exp is fine for small
                    # delta_cost can be large, so exp(-large) -> 0
                    prob = math.exp(-delta_cost / T)
                    if rng.random_sample() < prob:
                        p_ils = p_ls.copy()
                        # Move to this basin
                        no_improve_iter += 1
                    else:
                        # Reject
                        no_improve_iter += 1
                        if no_improve_iter % 10 == 0:
                            p_ils = best_p.copy()
                else:
                    no_improve_iter += 1
                    if no_improve_iter % 10 == 0:
                        p_ils = best_p.copy()
                
                T *= cooling

    return best_p.tolist()
