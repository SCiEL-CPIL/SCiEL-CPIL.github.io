"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) valid."""
    # T has shape (n*m, m*p, p*n)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _check_decomposition(T, U, V, W, tol=1e-9):
    """Check if U @ diag(V) @ W^T == T (in tensor form).
    Actually: T[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Compute reconstructed tensor
    recon = np.zeros(T.shape, dtype=np.float64)
    for c in range(rank):
        recon += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
    return np.allclose(recon, T, atol=tol, rtol=0)

def _round_to_halfint(arr, tol=1e-6):
    """Round array to nearest half-integer if close."""
    rounded = np.round(arr * 2) / 2
    if np.allclose(arr, rounded, atol=tol, rtol=0):
        return rounded
    return arr

def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    arr = _round_to_halfint(arr)
    return [[float(x) for x in row] for row in arr]

def _try_decompose(n, m, p, rank, max_iters=200, tol=1e-8, seed=42):
    """Try to find a rank-`rank` decomposition using iterative refinement."""
    np.random.seed(seed)
    
    T = _build_tensor(n, m, p)
    
    # Initialize U, V, W randomly
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Normalize columns to avoid scale issues
    for c in range(rank):
        norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
        if norm > 0:
            U[:, c] /= norm
            V[:, c] /= norm
            W[:, c] /= norm
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Update U: for each u, U[u,c] = sum_{v,w} T[u,v,w] * V[v,c] * W[w,c] / (sum_c' V[v,c']^2 * W[w,c']^2)
        # This is a simplified ALS-like update. Let's do proper ALS.
        
        # Update U
        for u in range(n * m):
            for c in range(rank):
                # numerator = sum_{v,w} T[u,v,w] * V[v,c] * W[w,c]
                num = 0.0
                den = 0.0
                for v in range(m * p):
                    for w in range(p * n):
                        if T[u, v, w] != 0:
                            num += T[u, v, w] * V[v, c] * W[w, c]
                            den += V[v, c] ** 2 * W[w, c] ** 2
                if den > 1e-12:
                    U[u, c] = num / den
                else:
                    U[u, c] = 0
        
        # Update V
        for v in range(m * p):
            for c in range(rank):
                num = 0.0
                den = 0.0
                for u in range(n * m):
                    for w in range(p * n):
                        if T[u, v, w] != 0:
                            num += T[u, v, w] * U[u, c] * W[w, c]
                            den += U[u, c] ** 2 * W[w, c] ** 2
                if den > 1e-12:
                    V[v, c] = num / den
                else:
                    V[v, c] = 0
        
        # Update W
        for w in range(p * n):
            for c in range(rank):
                num = 0.0
                den = 0.0
                for u in range(n * m):
                    for v in range(m * p):
                        if T[u, v, w] != 0:
                            num += T[u, v, w] * U[u, c] * V[v, c]
                            den += U[u, c] ** 2 * V[v, c] ** 2
                if den > 1e-12:
                    W[w, c] = num / den
                else:
                    W[w, c] = 0
        
        # Compute residual
        recon = np.zeros(T.shape, dtype=np.float64)
        for c in range(rank):
            recon += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
        residual = np.sum((recon - T) ** 2)
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if residual < tol:
            break
    
    # Check if we found a valid decomposition
    if best_residual < 1e-6:
        # Try to round to half-integers
        U_r = _round_to_halfint(best_U)
        V_r = _round_to_halfint(best_V)
        W_r = _round_to_halfint(best_W)
        if _check_decomposition(T, U_r, V_r, W_r, tol=1e-6):
            return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 90
    
    # Start with the schoolbook decomposition as fallback
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
    
    best_result = (U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    deadline = time.time() + 140  # 140 seconds, leaving 10s margin
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Also rank >= max(n, m, p) * max(n, m, p) = 6*6 = 36? No, that's not right.
    # Lower bound: rank >= max(n*m, m*p, p*n) = 30
    # Let's try ranks from 30 upward
    
    T = _build_tensor(n, m, p)
    
    for rank in range(30, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            result = _try_decompose(n, m, p, rank, max_iters=50, tol=1e-8, seed=seed)
            if result is not None:
                U_r, V_r, W_r = result
                best_result = (U_r, V_r, W_r)
                best_rank = rank
                break  # Found a valid decomposition at this rank, try lower
    
    # Return the best result found
    U_best, V_best, W_best = best_result
    return {
        "U": _to_list(U_best),
        "V": _to_list(V_best),
        "W": _to_list(W_best)
    }
# EVOLVE-BLOCK-END
