"""Rewritten program for 2x5x5 matrix multiplication tensor rank minimization."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    T_reconstructed = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
    T_target = _build_tensor(n, m, p)
    return np.allclose(T_reconstructed, T_target, atol=tol)

def _try_rank(n, m, p, target_rank, max_iter=500, seed=42, time_limit=10):
    """Try to find a rank-`target_rank` decomposition using iterative refinement."""
    np.random.seed(seed)
    start_time = time.time()
    
    # Initialize with random values
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    T_target = _build_tensor(n, m, p)
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Alternating minimization
        # Update U: for each (i,j), minimize ||T[i,j,:] - sum_r U[i*m+j,r] * V[j*p+*,r] * W[*,i,r]||
        for i in range(n):
            for j in range(m):
                # Build linear system for U[i*m+j, :]
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Let A_r = V[j*p+k,r] * W[k*n+i,r] for k in range(p)
                # Then T[i,j,:] = U[i*m+j,:] @ A where A[r,k] = V[j*p+k,r]*W[k*n+i,r]
                A = np.zeros((target_rank, p))
                for r in range(target_rank):
                    for k in range(p):
                        A[r, k] = V[j * p + k, r] * W[k * n + i, r]
                # Solve U[i*m+j,:] @ A = T[i,j,:]
                # This is underdetermined if target_rank > p, so use least squares
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A.T, T_target[i, j, :], rcond=None)
                    U[i * m + j, :] = U_new
                except:
                    pass
        
        # Update V: for each (j,k), similar approach
        for j in range(m):
            for k in range(p):
                A = np.zeros((target_rank, n))
                for r in range(target_rank):
                    for i in range(n):
                        A[r, i] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A.T, T_target[:, j, k], rcond=None)
                    V[j * p + k, :] = V_new
                except:
                    pass
        
        # Update W: for each (k,i), similar approach
        for k in range(p):
            for i in range(n):
                A = np.zeros((target_rank, m))
                for r in range(target_rank):
                    for j in range(m):
                        A[r, j] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A.T, T_target[i, :, k], rcond=None)
                    W[k * n + i, :] = W_new
                except:
                    pass
        
        # Compute error
        T_recon = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        error = np.linalg.norm(T_recon - T_target)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-9:
            break
    
    # Final check
    if best_error < 1e-9:
        # Convert to list format
        U_list = [[float(best_U[i, r]) for r in range(target_rank)] for i in range(n * m)]
        V_list = [[float(best_V[i, r]) for r in range(target_rank)] for i in range(m * p)]
        W_list = [[float(best_W[i, r]) for r in range(target_rank)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 5
    start_time = time.time()
    deadline = 130  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition as fallback
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_fallback
    
    # Try lower ranks
    for target_rank in range(15, rank_fallback):  # Try ranks 15 to 49
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, target_rank, max_iter=200, seed=seed, time_limit=8)
            if result is not None:
                best_rank = target_rank
                best_result = result
                break  # Found a valid decomposition at this rank
        
        if best_rank < rank_fallback:
            break  # Found a better rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
