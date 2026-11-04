import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit (5s buffer)

    # Convert to float64 for computation, keep int64 for exact cost
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    a_i = a.astype(np.int64)
    b_i = b.astype(np.int64)

    best_p = None
    best_cost = np.iinfo(np.int64).max

    def compute_cost(p):
        """Compute the QAP cost for permutation p using int64 arithmetic."""
        return int(np.sum(a_i * b_i[np.ix_(p, p)]))

    def compute_cost_fast(p):
        """Compute cost using float64 (for comparisons during search)."""
        return float(np.sum(a64 * b64[np.ix_(p, p)]))

    def fast_local_search(p, max_time=None, max_passes=50):
        """
        Fast 2-opt local search using precomputed row/column cost contributions.
        For each facility i at location p[i], precompute:
          row_cost[i] = sum_j a[i,j] * b[p[i], p[j]]
          col_cost[i] = sum_j a[j,i] * b[p[j], p[i]]
        Swap delta can be computed in O(1) after updating these in O(n).
        """
        p = p.copy().astype(np.int64)
        current_cost = compute_cost_fast(p)
        
        # Precompute row and column cost contributions
        # row_cost[i] = sum_j a[i,j] * b[p[i], p[j]]
        # col_cost[i] = sum_j a[j,i] * b[p[j], p[i]]
        row_cost = np.zeros(n, dtype=np.float64)
        col_cost = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            pi = p[i]
            row_cost[i] = np.dot(a64[i, :], b64[pi, p])
            col_cost[i] = np.dot(a64[:, i], b64[p, pi])
        
        # Also track diagonal terms: a[i,i]*b[p[i],p[i]]
        # These are included in both row_cost and col_cost, so subtract once
        diag_cost = np.array([a64[i,i] * b64[p[i], p[i]] for i in range(n)])
        
        # Total cost = sum(row_cost) + sum(col_cost) - sum(diag_cost)
        # Verify: sum(row_cost) = sum_{i,j} a[i,j]*b[p[i],p[j]]
        # sum(col_cost) = sum_{i,j} a[j,i]*b[p[j],p[i]] = sum_{i,j} a[i,j]*b[p[j],p[i]] (renaming)
        # These are different unless a is symmetric and b is symmetric.
        # Actually for QAP: cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # row_cost[i] = sum_j a[i,j] * b[p[i], p[j]]  -- this IS the contribution of row i
        # So total cost = sum_i row_cost[i]
        # col_cost is redundant for the total but useful for swap deltas.
        
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if max_time and time.time() > max_time:
                    return p, current_cost
                for j in range(i+1, n):
                    pi = p[i]
                    pj = p[j]
                    
                    # Delta for swapping i and j
                    # New row_cost[i] = sum_k a[i,k] * b[pj, p[k]]  (p[i] becomes pj)
                    # New row_cost[j] = sum_k a[j,k] * b[pi, p[k]]  (p[j] becomes pi)
                    # Old row_cost[i] = sum_k a[i,k] * b[pi, p[k]]
                    # Old row_cost[j] = sum_k a[j,k] * b[pj, p[k]]
                    
                    # The change in total cost from swapping:
                    # delta = [row_new_i + row_new_j] - [row_old_i + row_old_j]
                    #      = [sum_k a[i,k]*b[pj,p[k]] + sum_k a[j,k]*b[pi,p[k]]]
                    #        - [sum_k a[i,k]*b[pi,p[k]] + sum_k a[j,k]*b[pj,p[k]]]
                    #      = sum_k (a[i,k]-a[j,k]) * (b[pj,p[k]] - b[pi,p[k]])
                    #        + sum_k (a[j,k]-a[i,k]) * (b[pi,p[k]] - b[pj,p[k]])  -- same thing
                    # Wait, let me redo:
                    # delta_row = (row_new_i - row_old_i) + (row_new_j - row_old_j)
                    # row_new_i - row_old_i = sum_k a[i,k]*(b[pj,p[k]] - b[pi,p[k]])
                    # row_new_j - row_old_j = sum_k a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
                    # = sum_k (a[i,k] - a[j,k]) * (b[pj,p[k]] - b[pi,p[k]])
                    
                    diff_a = a64[i, :] - a64[j, :]
                    diff_b = b64[pj, :] - b64[pi, :]
                    delta = np.dot(diff_a, diff_b[p])
                    
                    if delta < -1e-9:
                        # Perform swap
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        # Update row_cost and col_cost
                        pi_new = p[i]  # was pj
                        pj_new = p[j]  # was pi
                        
                        # Recompute row_cost for i and j
                        row_cost[i] = np.dot(a64[i, :], b64[pi_new, p])
                        row_cost[j] = np.dot(a64[j, :], b64[pj_new, p])
                        
                        # Update col_cost: for all k, col_cost[k] involves b[p[k], p[i]] and b[p[k], p[j]]
                        # col_cost[k] = sum_m a[m,k] * b[p[m], p[k]]
                        # Only col_cost[i] and col_cost[j] change because p[i] and p[j] changed
                        col_cost[i] = np.dot(a64[:, i], b64[p, pi_new])
                        col_cost[j] = np.dot(a64[:, j], b64[p, pj_new])
                        
                        # Also update diag_cost for i and j
                        diag_cost[i] = a64[i,i] * b64[pi_new, pi_new]
                        diag_cost[j] = a64[j,j] * b64[pj_new, pj_new]
                        
        return p, current_cost

    def embedding_init(a64, b64, n, k=16):
        """
        Vector alignment initialization using leading eigenvectors.
        Embed facilities in latent space using eigenvectors of flow matrix,
        embed locations using eigenvectors of distance matrix,
        then solve soft assignment and harden.
        """
        # Symmetrize
        A_sym = (a64 + a64.T) / 2.0
        B_sym = (b64 + b64.T) / 2.0
        
        # Get leading eigenvectors
        try:
            eigvals_a, eigvecs_a = np.linalg.eigh(A_sym)
            eigvals_b, eigvecs_b = np.linalg.eigh(B_sym)
        except Exception:
            return None
        
        # Use top k eigenvectors (largest eigenvalues)
        idx_a = np.argsort(-eigvals_a)[:k]
        idx_b = np.argsort(-eigvals_b)[:k]
        
        Fa = eigvecs_a[:, idx_a]  # (n, k) facility embedding
        Fb = eigvecs_b[:, idx_b]  # (n, k) location embedding
        
        # Soft assignment: minimize ||Fa @ X @ Fb.T - A_sym||_F^2
        # where X is a soft permutation matrix (n x n)
        # Gradient descent on log-softmax parameters
        # X = softmax(M) where M is (n, n)
        
        # Initialize M
        # Use alignment: M[i,j] = Fa[i] @ Fb[j]
        M = Fa @ Fb.T  # (n, n)
        
        # Gradient descent on M with entropic regularization
        # Objective: minimize -trace(Fa.T @ M @ Fb) + trace(M @ A_sym)
        # subject to M being a doubly stochastic matrix (Birkhoff polytope)
        # Use Sinkhorn-Knopp iteration to project onto doubly stochastic
        
        lr = 0.1
        for step in range(100):
            if time.time() > start_time + 20:
                break
            # Compute gradient
            # d/dM: -Fa @ Fb.T + A_sym (for the linear part)
            # + entropic regularizer gradient (handled by projection)
            grad = A_sym - Fa @ Fb.T
            M -= lr * grad
            M /= max(np.abs(M).max(), 1e-10) * 0.1  # normalize
            
            # Project to doubly stochastic via Sinkhorn
            for _ in range(20):
                row_sums = M.sum(axis=1, keepdims=True)
                M /= row_sums
                col_sums = M.sum(axis=0, keepdims=True)
                M /= col_sums
        
        # Harden: solve linear assignment on M
        try:
            _, p = linear_sum_assignment(-M)
            return p.astype(np.int64)
        except Exception:
            return None

    def spectral_bisection_init(a64, b64):
        """
        Spectral bisection initialization using graph partitioning.
        """
        A = (a64 + a64.T) / 2.0
        
        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        L = D - A
        
        try:
            L_sparse = csr_matrix(L)
            eigenvalues, eigenvectors = eigsh(L_sparse, k=2, sigma=0, which='LM')
            fiedler_idx = np.argmin(np.abs(eigenvalues - eigenvalues[1]))
            fiedler = eigenvectors[:, fiedler_idx]
        except Exception:
            fiedler = degrees.copy()
        
        median_val = np.median(fiedler)
        left_nodes = np.where(fiedler <= median_val)[0]
        right_nodes = np.where(fiedler > median_val)[0]
        
        if len(left_nodes) < n // 4:
            sorted_by_fiedler = np.argsort(fiedler)
            left_nodes = sorted_by_fiedler[:n//2]
            right_nodes = sorted_by_fiedler[n//2:]
        elif len(right_nodes) < n // 4:
            sorted_by_fiedler = np.argsort(fiedler)
            left_nodes = sorted_by_fiedler[:n//2]
            right_nodes = sorted_by_fiedler[n//2:]
        
        loc_centralities = b64.sum(axis=1)
        loc_order = np.argsort(loc_centralities)
        
        p = np.empty(n, dtype=np.int64)
        
        left_flow = a64.sum(axis=1)[left_nodes]
        left_rank = np.argsort(-left_flow)
        right_flow = a64.sum(axis=1)[right_nodes]
        right_rank = np.argsort(-right_flow)
        
        left_locs = loc_order[:len(left_nodes)]
        right_locs = loc_order[len(left_nodes):]
        
        for idx, node in enumerate(left_nodes[left_rank]):
            p[node] = left_locs[idx]
        for idx, node in enumerate(right_nodes[right_rank]):
            p[node] = right_locs[idx]
        
        return p

    def linex_init(a64, b64):
        """
        Linex initialization: Use linear assignment to create initial permutation.
        """
        fac_weights = a64.sum(axis=1)
        loc_weights = b64.sum(axis=1)
        C = np.outer(fac_weights, loc_weights)
        _, p = linear_sum_assignment(C)
        return p.astype(np.int64)

    def random_restart_init(n, rng):
        """Generate a random permutation."""
        return rng.permutation(n).astype(np.int64)

    # Generate initial solutions and run local search on each
    candidates = []
    
    # 1. Embedding-based initialization
    p_emb = embedding_init(a64, b64, n, k=16)
    if p_emb is not None:
        candidates.append(p_emb)
    
    # 2. Spectral bisection
    p_spectral = spectral_bisection_init(a64, b64)
    candidates.append(p_spectral)
    
    # 3. Linex
    p_linex = linex_init(a64, b64)
    candidates.append(p_linex)
    
    # 4. Rank matching
    heavy = np.argsort(-a64.sum(axis=1), kind="stable")
    central = np.argsort(b64.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    candidates.append(p_rank)
    
    # 5. Reverse rank matching
    p_rank_rev = np.empty(n, dtype=np.int64)
    p_rank_rev[heavy] = central[::-1]
    candidates.append(p_rank_rev)
    
    # Run local search on each candidate
    time_per_candidate = 15.0
    for p_init in candidates:
        if time.time() > deadline - 10:
            break
        p_ls, cost_ls = fast_local_search(p_init, max_time=deadline - 5, max_passes=30)
        exact_cost = compute_cost(p_ls)
        if exact_cost < best_cost:
            best_cost = exact_cost
            best_p = p_ls.copy()
    
    # Try multiple random restarts with remaining time
    np_random = np.random.RandomState(42)
    
    while time.time() < deadline - 3:
        p_rand = random_restart_init(n, np_random)
        p_ls, cost_ls = fast_local_search(p_rand, max_time=deadline - 2, max_passes=10)
        exact_cost = compute_cost(p_ls)
        if exact_cost < best_cost:
            best_cost = exact_cost
            best_p = p_ls.copy()
    
    # Final polish: one more local search pass on best solution
    if best_p is not None and time.time() < deadline - 2:
        p_final, _ = fast_local_search(best_p, max_time=deadline - 1, max_passes=20)
        exact_cost = compute_cost(p_final)
        if exact_cost < best_cost:
            best_cost = exact_cost
            best_p = p_final.copy()
    
    if best_p is None:
        best_p = p_rank
    
    return best_p.tolist()
