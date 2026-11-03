"""Rewritten QAP solver: spectral init + fast local search + Hungarian refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Multiple spectral/rank initializations.
    2. Fast local search with correct O(n) delta computation.
    3. Hungarian-based linearization refinement.
    4. Final polish pass.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    rng = np.random.default_rng(42)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        return int(np.einsum('ij,jk,ik->', a, b[:, p], b[p, :]))
    
    # --- Spectral initialization ---
    def spectral_init(method='flow'):
        """Generate initial permutation using spectral methods."""
        if method == 'flow':
            a_centered = a - a.mean()
            try:
                eigvals, eigvecs = np.linalg.eigh(a_centered)
                idx = np.argsort(-np.abs(eigvals))
                spec_coords = eigvecs[:, idx[0]]
                fac_rank = np.argsort(spec_coords)
                loc_rank = np.argsort(b.sum(axis=1))
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'distance':
            b_centered = b - b.mean()
            try:
                eigvals, eigvecs = np.linalg.eigh(b_centered)
                idx = np.argsort(-np.abs(eigvals))
                spec_coords = eigvecs[:, idx[0]]
                loc_rank = np.argsort(spec_coords)
                fac_rank = np.argsort(a.sum(axis=1))
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'combined':
            a_centered = a - a.mean()
            b_centered = b - b.mean()
            try:
                eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
                eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
                idx_a = np.argsort(-np.abs(eigvals_a))
                idx_b = np.argsort(-np.abs(eigvals_b))
                spec_a = eigvecs_a[:, idx_a[0]]
                spec_b = eigvecs_b[:, idx_b[0]]
                fac_rank = np.argsort(spec_a)
                loc_rank = np.argsort(spec_b)
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'random':
            p = rng.permutation(n)
            return p
        else:
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    # --- Fast local search with correct O(n) delta ---
    def local_search(p, cost, time_limit):
        """Perform local search with fast incremental delta updates.
        
        Delta for swapping p[i]=u and p[j]=v:
        delta = (a[i,i] - a[j,j])*(b[v,v] - b[u,u])
              + (a[i,j] - a[j,i])*(b[v,u] - b[u,v])
              + (a[i,:] - a[j,:]) @ (b[v,:] - b[u,:])
              + (a[:,i] - a[:,j]) @ (b[:,v] - b[:,u])
        """
        p = p.copy()
        
        # Precompute row/col differences for fast delta
        improved = True
        no_improve_count = 0
        max_no_improve = 5000
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Try a batch of random swaps
            num_trials = 20000
            for _ in range(num_trials):
                if time.time() > time_limit:
                    break
                
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue
                
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                # Fast delta computation - O(n)
                # Term 1: diagonal
                delta = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u])
                # Term 2: off-diagonal between i,j
                delta += (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
                # Term 3: row contribution
                delta += np.dot(a[i, :] - a[j, :], b[v, :] - b[u, :])
                # Term 4: column contribution
                delta += np.dot(a[:, i] - a[:, j], b[:, v] - b[:, u])
                
                if delta < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    cost += int(delta)
                    improved = True
                    no_improve_count = 0
                    
            if not improved:
                no_improve_count += num_trials
                if no_improve_count > max_no_improve:
                    break
        
        return p, cost
    
    # --- Hungarian-based refinement ---
    def hungarian_refine(p, cost):
        """Refine using Hungarian algorithm on linearized cost matrix."""
        b_col_p = b[:, p]  # shape (n, n), [j, k] = b[j, p[k]]
        b_row_p = b[p, :]  # shape (n, n), [k, j] = b[p[k], j]
        
        # C[i,j] = sum_k a[i,k]*b[j, p[k]] + sum_k a[k,i]*b[p[k], j]
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < cost:
            return p_hung, cost_hung
        return p, cost
    
    # --- Main algorithm ---
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Multiple initializations with local search
    methods = ['flow', 'distance', 'combined', 'random', 'random']
    for method in methods:
        if time.time() > deadline - 40.0:
            break
        p_init = spectral_init(method)
        cost_init = compute_cost(p_init)
        
        # Local search with time budget
        time_left = deadline - time.time() - 20.0
        if time_left < 5.0:
            break
        p_ls, cost_ls = local_search(p_init, cost_init, time.time() + time_left)
        
        if cost_ls < best_cost:
            best_p = p_ls.copy()
            best_cost = cost_ls
    
    # If no valid solution found, fall back to simple rank matching
    if best_p is None:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        best_p = np.empty(n, dtype=np.int64)
        best_p[heavy] = central
        best_cost = compute_cost(best_p)
    
    # Phase 2: Hungarian refinement + local search
    if time.time() < deadline - 15.0:
        best_p, best_cost = hungarian_refine(best_p, best_cost)
        if time.time() < deadline - 5.0:
            best_p, best_cost = local_search(best_p, best_cost, deadline - 3.0)
    
    # Phase 3: Another Hungarian + local search if time permits
    if time.time() < deadline - 10.0:
        best_p, best_cost = hungarian_refine(best_p, best_cost)
        if time.time() < deadline - 3.0:
            best_p, best_cost = local_search(best_p, best_cost, deadline - 1.0)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
