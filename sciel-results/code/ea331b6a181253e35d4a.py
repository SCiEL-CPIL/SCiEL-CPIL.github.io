import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Indices for the 27 non-zero entries of the tensor T
    eq_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_eqs = len(eq_indices)
    
    # Seed with naive solution (rank 27)
    U_naive = np.zeros((n*m, 27), dtype=int)
    V_naive = np.zeros((m*p, 27), dtype=int)
    W_naive = np.zeros((p*n, 27), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, col] = 1
                V_naive[j*p+k, col] = 1
                W_naive[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_naive, V_naive, W_naive
    best_rank = 27
    
    def build_M(U, V, rank):
        """Vectorized construction of the linear system matrix M"""
        M = np.zeros((num_eqs, p * n * rank))
        for e_idx, (i, j, k) in enumerate(eq_indices):
            w_idx_base = k * n + i
            coeffs = U[i*m+j, :] * V[j*p+k, :]
            for c in range(rank):
                if coeffs[c] != 0:
                    M[e_idx, c * p*n + w_idx_base] = coeffs[c]
        return M
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor"""
        for (i, j, k) in eq_indices:
            w_idx = k * n + i
            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[:, w_idx])
            if abs(val - 1.0) > 1e-9:
                return False
        return True
    
    def try_solve_for_W(U, V, rank, tol=1e-6):
        """Try to solve for W given U and V. Returns W or None"""
        M = build_M(U, V, rank)
        try:
            Y_vec, residuals, rank_m, s = np.linalg.lstsq(M, 2*np.ones(num_eqs), rcond=None)
            Y_rounded = np.round(Y_vec)
            error = np.max(np.abs(Y_vec - Y_rounded))
            
            if error < tol:
                W_solution = Y_rounded / 2.0
                if verify_decomposition(U, V, W_solution, rank):
                    return W_solution
        except Exception:
            pass
        return None
    
    def random_UV_sparse(rank, sparsity):
        """Generate random sparse U and V with ±1 entries"""
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        for c in range(rank):
            u_mask = np.random.random(n*m) < sparsity
            v_mask = np.random.random(m*p) < sparsity
            U[:, c] = np.where(u_mask, np.random.choice([-1, 1], n*m), 0)
            V[:, c] = np.where(v_mask, np.random.choice([-1, 1], m*p), 0)
        return U, V
    
    def random_UV_dense(rank):
        """Generate random dense U and V with ±1 entries"""
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        for c in range(rank):
            U[:, c] = np.random.choice([-1, 1], n*m)
            V[:, c] = np.random.choice([-1, 1], m*p)
        return U, V
    
    def local_search(U, V, W, rank, max_iters=50):
        """Local search: try flipping individual entries in U, V, W"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for iteration in range(max_iters):
            improved = False
            
            # Try flipping entries in U
            for i in range(n*m):
                for c in range(rank):
                    if U[i, c] != 0:
                        U[i, c] *= -1
                        if verify_decomposition(U, V, W, rank):
                            improved = True
                            break
                        else:
                            U[i, c] *= -1
                if improved:
                    break
            
            if improved:
                continue
            
            # Try flipping entries in V
            for j in range(m*p):
                for c in range(rank):
                    if V[j, c] != 0:
                        V[j, c] *= -1
                        if verify_decomposition(U, V, W, rank):
                            improved = True
                            break
                        else:
                            V[j, c] *= -1
                if improved:
                    break
            
            if improved:
                continue
            
            # Try flipping entries in W
            for k in range(p*n):
                for c in range(rank):
                    if W[k, c] != 0:
                        W[k, c] *= -1
                        if verify_decomposition(U, V, W, rank):
                            improved = True
                            break
                        else:
                            W[k, c] *= -1
                if improved:
                    break
            
            if not improved:
                break
        
        return U, V, W
    
    # Search for lower rank
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        
        found = False
        
        # Phase 1: Sparse random search
        for sparsity in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            if time.time() > deadline or found:
                break
            max_trials = 300
            for trial in range(max_trials):
                if time.time() > deadline:
                    break
                U, V = random_UV_sparse(target_rank, sparsity)
                W_sol = try_solve_for_W(U, V, target_rank)
                if W_sol is not None:
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W_sol
                    found = True
                    break
        
        if found:
            break
        
        # Phase 2: Dense random search
        if time.time() < deadline:
            for trial in range(200):
                if time.time() > deadline:
                    break
                U, V = random_UV_dense(target_rank)
                W_sol = try_solve_for_W(U, V, target_rank)
                if W_sol is not None:
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W_sol
                    found = True
                    break
        
        if found:
            break
        
        # Phase 3: Local search from random starts
        if time.time() < deadline:
            for trial in range(100):
                if time.time() > deadline:
                    break
                U, V = random_UV_sparse(target_rank, 0.5)
                W_sol = try_solve_for_W(U, V, target_rank)
                if W_sol is not None:
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W_sol
                    found = True
                    break
                # Try local search even if W not found initially
                if time.time() < deadline:
                    U_ls, V_ls, W_ls = local_search(U, V, np.zeros((p*n, target_rank)), target_rank, max_iters=20)
                    if verify_decomposition(U_ls, V_ls, W_ls, target_rank):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_ls, V_ls, W_ls
                        found = True
                        break
    
    def to_list(mat):
        return mat.tolist()
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
