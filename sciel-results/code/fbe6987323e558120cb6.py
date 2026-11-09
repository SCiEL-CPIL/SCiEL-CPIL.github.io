"""Tensor decomposition search for 2x5 x 5x5 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 5, 5]

def construct():
    """Search for rank-minimal tensor decomposition."""
    deadline = time.time() + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 50
    
    # Build schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if decomposition is exact."""
        nm, mp, pn = n * m, m * p, p * n
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    def als_decompose(target_rank, max_iters=200, tol=1e-10):
        """
        Alternating Least Squares to find U, V, W of given rank
        such that T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        """
        if time.time() > deadline - 10:
            return None
        
        nm, mp, pn = n * m, m * p, p * n
        
        # Initialize randomly
        U = np.random.randn(nm, target_rank) * 0.5
        V = np.random.randn(mp, target_rank) * 0.5
        W = np.random.randn(pn, target_rank) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Update U: for each (i,j), solve linear system
            # For fixed (i,j), the equation is:
            # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
            # This is: U[i*m+j, :] @ M = ones(p) where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    # Build M matrix: (p x target_rank)
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve U[i*m+j, :] @ M = ones(p)
                    # i.e., M.T @ U[i*m+j, :].T = ones(p)
                    try:
                        U[i * m + j, :], residuals, rank_M, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    except Exception:
                        continue
            
            # Update V: for each (j,k), solve
            # sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for all i
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :], residuals, rank_M, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    except Exception:
                        continue
            
            # Update W: for each (k,i), solve
            # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :], residuals, rank_M, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    except Exception:
                        continue
            
            # Check convergence
            if it % 10 == 0:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for r in range(target_rank):
                                total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            max_err = max(max_err, abs(total - 1.0))
                if max_err < tol:
                    return U, V, W
        
        # Final check
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(target_rank):
                        total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(total - 1.0))
        
        if max_err < 1e-8:
            return U, V, W
        return None
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_schoolbook
    
    # Try decreasing ranks
    # Known lower bound for 2x5x5 is around 20-25, try from 49 down
    for target_rank in range(rank_schoolbook - 1, 15, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Multiple random restarts for each rank
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            result = als_decompose(target_rank, max_iters=100, tol=1e-10)
            if result is not None:
                U_cand, V_cand, W_cand = result
                if verify(U_cand, V_cand, W_cand, tol=1e-9):
                    best_U, best_V, best_W = U_cand, V_cand, W_cand
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
    
    # Also try the schoolbook as fallback
    if best_rank == rank_schoolbook:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
