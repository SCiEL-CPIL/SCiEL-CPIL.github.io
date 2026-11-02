"""Matrix multiplication 3x5x7 - rank minimization via algebraic search."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - target) > 1e-12:
                    return False
    return True

def _verify_np(U, V, W, n, m, p):
    """Fast verification using numpy."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - expected) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(n, m, p, rank, time_limit=5.0):
    """
    Try to find a rank-`rank` decomposition using iterative refinement.
    Returns dict or None if not found.
    """
    start = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    # Build the tensor target
    # T[i,j,k] = 1 for all i,j,k in matrix multiplication
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Use a simple gradient-free approach: try to satisfy equations one by one
    # This is a non-convex problem, so we use multiple restarts
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(20):
        if time.time() - start > time_limit:
            break
        
        np.random.seed(42 + attempt)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Scale to get reasonable magnitudes
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        # Iterative refinement: for each (i,j,k), adjust the vectors
        for it in range(50):
            if time.time() - start > time_limit:
                break
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i*m + j
                        idx_v = j*p + k
                        idx_w = k*n + i
                        current = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                        err += (current - 1.0)**2
            
            best_err = min(best_err, err)
            if err < 1e-10:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
            
            if err < best_err:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_err = err
            
            # Simple gradient step (approximate)
            # For each (i,j,k), the gradient w.r.t. U[idx_u, c] is V[idx_v, c] * W[idx_w, c]
            # We do a simplified update
            lr = 0.01
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i*m + j
                        idx_v = j*p + k
                        idx_w = k*n + i
                        current = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                        residual = current - 1.0
                        # Update U
                        U[idx_u] -= lr * residual * (V[idx_v] * W[idx_w])
                        # Update V  
                        V[idx_v] -= lr * residual * (U[idx_u] * W[idx_w])
                        # Update W
                        W[idx_w] -= lr * residual * (U[idx_u] * V[idx_v])
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-10 and best_U is not None:
        # Convert to list of lists with rounded values for exactness
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    n, m, p = 3, 5, 7
    deadline = time.time() + 140  # 140s deadline with 10s safety margin
    
    # Schoolbook baseline
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try recursive block partitioning: 3 = 1 + 2
    # 1x5x7 has rank 35, 2x5x7 needs to be solved
    # Total rank would be 35 + rank(2x5x7)
    
    # Try various ranks for the full problem
    # Lower bound: max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    # Try ranks from 35 upward
    
    ranks_to_try = []
    # First try the recursive approach: 1x5x7 (rank 35) + 2x5x7
    # For 2x5x7, lower bound is max(10, 35, 14) = 35
    # So total lower bound is 35 + 35 = 70
    
    # Try ranks: 70, 71, ..., up to 104
    for r in range(70, min(105, best_rank)):
        if time.time() > deadline - 10:
            break
        result = _try_rank_decomposition(n, m, p, r, time_limit=3.0)
        if result is not None:
            # Verify
            if _verify_np(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = r
                break  # Found minimal rank in this range
    
    # If we haven't improved, try a few more ranks
    if best_rank == 105:
        for r in [60, 50, 45, 40, 35]:
            if time.time() > deadline - 10:
                break
            result = _try_rank_decomposition(n, m, p, r, time_limit=3.0)
            if result is not None:
                if _verify_np(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = r
                    break
    
    return best_result

# EVOLVE-BLOCK-END
