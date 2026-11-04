import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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
        row_cost = np.zeros(n, dtype=np.float64)
        col_cost = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            pi = p[i]
            row_cost[i] = np.dot(a64[i, :], b64[pi, p])
            col_cost[i] = np.dot(a64[:, i], b64[p, pi])
        
        diag_cost = np.array([a64[i,i] * b64[p[i], p[i]] for i in range(n)])
        
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
                    
                    diff_a = a64[i, :] - a64[j, :]
                    diff_b = b64[pj, :] - b64[pi, :]
                    delta = np.dot(diff_a, diff_b[p])
                    
                    if delta < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        pi_new = p[i]
                        pj_new = p[j]
                        
                        row_cost[i] = np.dot(a64[i, :], b64[pi_new, p])
                        row_cost[j] = np.dot(a64[j, :], b64[pj_new, p])
                        
                        col_cost[i] = np.dot(a64[:, i], b64[p, pi_new])
                        col_cost[j] = np.dot(a64[:, j], b64[p, pj_new])
                        
                        diag_cost[i] = a64[i,i] * b64[pi_new, pi_new]
                        diag_cost[j] = a64[j,j] * b64[pj_new, pj_new]
                        
        return p, current_cost

    def embedding_init(a64, b64, n, k=16):
        """
        Vector alignment initialization using leading eigenvectors.
        """
        A_sym = (a64 + a64.T) / 2.0
        B_sym = (b64 + b64.T) / 2.0
        
        try:
            eigvals_a, eigvecs_a = np.linalg.eigh(A_sym)
            eigvals_b, eigvecs_b = np.linalg.eigh(B_sym)
        except Exception:
            return None
        
        idx_a = np.argsort(-eigvals_a)[:k]
        idx_b = np.argsort(-eigvals_b)[:k]
        
        Fa = eigvecs_a[:, idx_a]
        Fb = eigvecs_b[:, idx_b]
        
        M = Fa @ Fb.T
        
        lr = 0.1
        for step in range(100):
            if time.time() > start_time + 20:
                break
            grad = A_sym - Fa @ Fb.T
            M -= lr * grad
            M /= max(np.abs(M).max(), 1e-10) * 0.1
            
            for _ in range(20):
                row_sums = M.sum(axis=1, keepdims=True)
                M /= row_sums
                col_sums = M.sum(axis=0, keepdims=True)
                M /= col_sums
        
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
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
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

    def frank_wolfe_polish(p_init, time_budget=20.0, max_iters=100):
        """
        Frank-Wolfe on Birkhoff polytope for polishing.
        Starts from the permutation p_init as a vertex of the Birkhoff polytope.
        """
        X = np.zeros((n, n), dtype=np.float64)
        X[np.arange(n), p_init] = 1.0
        
        best_p = p_init.copy()
        best_cost = compute_cost(p_init)
        
        prev_cost = best_cost
        start = time.time()
        
        for t in range(max_iters):
            if time.time() - start > time_budget:
                break
            
            # Gradient of f(X) = tr(A X B X^T) = sum_{i,j} A[i,j] B[X[i],X[j]]
            # grad = A @ X @ B + A.T @ X @ B.T
            grad = a64 @ X @ b64 + a64.T @ X @ b64.T
            grad = (grad + grad.T) / 2.0
            
            # Solve linear assignment on grad (minimize)
            row_ind, col_ind = linear_sum_assignment(grad)
            Y = np.zeros((n, n), dtype=np.float64)
            Y[row_ind, col_ind] = 1.0
            
            # FW step size
            step = 2.0 / (t + 2)
            
            # Update
            X = (1 - step) * X + step * Y
            
            # Get permutation from Y (hard assignment)
            p_new = col_ind.copy()
            
            cost_new = compute_cost(p_new)
            if cost_new < best_cost:
                best_cost = cost_new
                best_p = p_new.copy()
            
            # Convergence check
            if abs(prev_cost - cost_new) < 1e-10 * max(1, abs(cost_new)):
                break
            prev_cost = cost_new
        
        return best_p, best_cost

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
    
    while time.time() < deadline - 30:
        p_rand = random_restart_init(n, np_random)
        p_ls, cost_ls = fast_local_search(p_rand, max_time=deadline - 25, max_passes=10)
        exact_cost = compute_cost(p_ls)
        if exact_cost < best_cost:
            best_cost = exact_cost
            best_p = p_ls.copy()
    
    # Final polish: Frank-Wolfe on best solution
    if best_p is not None and time.time() < deadline - 10:
        fw_budget = min(30.0, deadline - time.time() - 2)
        if fw_budget > 5:
            p_fw, cost_fw = frank_wolfe_polish(best_p, time_budget=fw_budget)
            exact_cost_fw = compute_cost(p_fw)
            if exact_cost_fw < best_cost:
                best_cost = exact_cost_fw
                best_p = p_fw.copy()
    
    # Final local search polish
    if best_p is not None and time.time() < deadline - 2:
        p_final, _ = fast_local_search(best_p, max_time=deadline - 1, max_passes=20)
        exact_cost = compute_cost(p_final)
        if exact_cost < best_cost:
            best_cost = exact_cost
            best_p = p_final.copy()
    
    if best_p is None:
        best_p = p_rank
    
    return best_p.tolist()
