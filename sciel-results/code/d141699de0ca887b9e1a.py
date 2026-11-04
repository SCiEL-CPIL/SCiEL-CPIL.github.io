"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Build the target tensor T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = (U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Lower bound for M_{3,3,5} is max(9, 15, 15) = 15
    # Known results suggest rank might be around 15-20 for these dimensions
    
    # Strategy: For each target rank r, try to find U, V, W with half-integer entries
    # using a combination of:
    # 1. Alternating least squares with random restarts
    # 2. Local search / refinement around found solutions
    # 3. Rounding to half-integers and verification
    
    # Precompute tensor for fast error evaluation
    T = np.ones((n, m, p))
    
    def compute_error(U, V, W, r):
        """Compute the error ||T - sum_c U[:,:,c] V[:,:,c] W[:,:,c]||_F"""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    error += (val - 1.0) ** 2
        return np.sqrt(error)
    
    def try_rank(r, max_restarts=20, max_iters=200):
        """Try to find a rank-r decomposition. Returns (U, V, W) if found, else None."""
        if time.time() > deadline:
            return None
            
        for attempt in range(max_restarts):
            if time.time() > deadline:
                return None
                
            np.random.seed(r * 1000 + attempt * 7 + 42)
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Alternating least squares
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                    
                # Update U
                U_new = U.copy()
                for i in range(n):
                    for j in range(m):
                        # Build A: (p, r) where A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        A_mat = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                A_mat[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        try:
                            U_new[i * m + j, :] = np.linalg.lstsq(A_mat, T[i, j, :], rcond=None)[0]
                        except:
                            pass
                
                # Update V
                V_new = V.copy()
                for j in range(m):
                    for k in range(p):
                        # Build A: (n, r) where A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        A_mat = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A_mat[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            V_new[j * p + k, :] = np.linalg.lstsq(A_mat, T[:, j, k], rcond=None)[0]
                        except:
                            pass
                
                # Update W
                W_new = W.copy()
                for k in range(p):
                    for i in range(n):
                        # Build A: (m, r) where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        A_mat = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A_mat[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            W_new[k * n + i, :] = np.linalg.lstsq(A_mat, T[i, :, k], rcond=None)[0]
                        except:
                            pass
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                error = compute_error(U, V, W, r)
                if error < 1e-8:
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    error_rounded = compute_error(U_rounded, V_rounded, W_rounded, r)
                    if error_rounded < 1e-10:
                        return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                    
                    # If rounding doesn't work, try local refinement around rounded values
                    # Use the rounded values as starting point for more iterations
                    U, V, W = U_rounded, V_rounded, W_rounded
                    for it2 in range(50):
                        if time.time() > deadline - 5:
                            break
                        # Quick ALS step
                        U_new = U.copy()
                        for i in range(n):
                            for j in range(m):
                                A_mat = np.zeros((p, r))
                                for k in range(p):
                                    for c in range(r):
                                        A_mat[k, c] = V[j * p + k, c] * W[k * n + i, c]
                                try:
                                    U_new[i * m + j, :] = np.linalg.lstsq(A_mat, T[i, j, :], rcond=None)[0]
                                except:
                                    pass
                        V_new = V.copy()
                        for j in range(m):
                            for k in range(p):
                                A_mat = np.zeros((n, r))
                                for i in range(n):
                                    for c in range(r):
                                        A_mat[i, c] = U[i * m + j, c] * W[k * n + i, c]
                                try:
                                    V_new[j * p + k, :] = np.linalg.lstsq(A_mat, T[:, j, k], rcond=None)[0]
                                except:
                                    pass
                        W_new = W.copy()
                        for k in range(p):
                            for i in range(n):
                                A_mat = np.zeros((m, r))
                                for j in range(m):
                                    for c in range(r):
                                        A_mat[j, c] = U[i * m + j, c] * V[j * p + k, c]
                                try:
                                    W_new[k * n + i, :] = np.linalg.lstsq(A_mat, T[i, :, k], rcond=None)[0]
                                except:
                                    pass
                        U, V, W = U_new, V_new, W_new
                        error = compute_error(U, V, W, r)
                        if error < 1e-8:
                            break
                    
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    error_rounded = compute_error(U_rounded, V_rounded, W_rounded, r)
                    if error_rounded < 1e-10:
                        return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
        
        return None
    
    # Try ranks from 15 up to schoolbook
    for r in range(15, rank_schoolbook):
        if time.time() > deadline - 10:
            break
            
        result = try_rank(r)
        if result is not None:
            U_out, V_out, W_out = result
            best_result = (U_out, V_out, W_out)
            best_rank = r
            # Found a valid decomposition at this rank, return immediately
            # (since we're searching from low to high, this is the minimum)
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    U_out, V_out, W_out = best_result
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
