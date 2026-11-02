"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all valid triples."""
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _reconstruct_from_factors(U, V, W, n, m, p, rank):
    """Reconstruct the tensor from factors: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    return T

def _als_iterate(T, U, V, W, n, m, p, rank, max_iter=30, tol=1e-8):
    """Alternating Least Squares to fit T = sum_r u_r ⊗ v_r ⊗ w_r."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Update U: fix V and W
        for i in range(n):
            for j in range(m):
                # For fixed (i,j), we want: T[i,j,k] ≈ sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then T[i,j,:] ≈ A @ U[i*m+j, :]
                A = np.zeros((p, rank), dtype=np.float64)
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                b = T[i, j, :]
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    x = np.zeros(rank)
                U[i * m + j, :] = x
        
        # Update V: fix U and W
        for j in range(m):
            for k in range(p):
                # For fixed (j,k), we want: T[i,j,k] ≈ sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Let A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # Then T[:,j,k] ≈ A @ V[j*p+k, :]
                A = np.zeros((n, rank), dtype=np.float64)
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                b = T[:, j, k]
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    x = np.zeros(rank)
                V[j * p + k, :] = x
        
        # Update W: fix U and V
        for k in range(p):
            for i in range(n):
                # For fixed (k,i), we want: T[i,j,k] ≈ sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Let A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Then T[i,:,k] ≈ A @ W[k*n+i, :]
                A = np.zeros((m, rank), dtype=np.float64)
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                b = T[i, :, k]
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    x = np.zeros(rank)
                W[k * n + i, :] = x
        
        # Check convergence
        T_recon = _reconstruct_from_factors(U, V, W, n, m, p, rank)
        err = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
        if err < tol:
            break
    
    return U, V, W, err

def _round_to_half_integers(U, V, W):
    """Round entries to nearest half-integer."""
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0
    return U_r, V_r, W_r

def _verify_exact(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify exact decomposition."""
    T_target = _build_tensor(n, m, p)
    T_recon = _reconstruct_from_factors(U, V, W, n, m, p, rank)
    return np.linalg.norm(T_recon - T_target) < tol

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition as a dict."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, target_rank, seed, time_budget):
    """Try to find a valid decomposition at the given rank."""
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    T = _build_tensor(n, m, p)
    
    # Run ALS
    U, V, W, err = _als_iterate(T, U, V, W, n, m, p, target_rank, max_iter=20, tol=1e-8)
    
    # Round to half-integers
    U_r, V_r, W_r = _round_to_half_integers(U, V, W)
    
    # Verify
    if _verify_exact(U_r, V_r, W_r, n, m, p, target_rank, tol=1e-6):
        # Convert to lists
        U_list = [[U_r[i, j] for j in range(target_rank)] for i in range(n * m)]
        V_list = [[V_r[i, j] for j in range(target_rank)] for i in range(m * p)]
        W_list = [[W_r[i, j] for j in range(target_rank)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # Known results for 4x4x7 suggest rank around 55-65
    # Try ranks from low to high
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a range of ranks, starting from reasonable estimates
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple seeds for each rank
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, target_rank, seed, deadline - time.time())
            if result is not None:
                best_result = result
                best_rank = target_rank
                break  # Found valid at this rank, try lower next
    
    return best_result
# EVOLVE-BLOCK-END
