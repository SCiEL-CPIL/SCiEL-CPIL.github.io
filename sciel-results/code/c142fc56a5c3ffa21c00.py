"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor as a 3D array T[i,j,k]=1."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U,V,W exactly decompose the multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Reshape: U is (n*m, rank) -> (n, m, rank)
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    # Compute sum over rank: result[i,j,k] = sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
    # Use einsum
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    target = np.ones((n, m, p))
    return np.allclose(result, target, atol=tol, rtol=0)

def _als_solve(U, V, W, n, m, p, max_iter=200, tol=1e-8, deadline=None):
    """ALS refinement of the decomposition."""
    rank = U.shape[2]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    target = np.ones((n, m, p))
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Update U: for each (i,j), U[i,j,:] should satisfy sum_k W[k,i,:]*V[j,k,:] * U[i,j,:] = 1
        # This is: (W[:,i,:] * V[j,:,:]) dot U[i,j,:] = 1
        # Build matrix M[i,j] of shape (rank, rank): M[i,j] = sum_k outer(W[k,i], V[j,k])
        # Then M[i,j] @ U[i,j,:] = 1
        for i in range(n):
            for j in range(m):
                M = np.zeros((rank, rank))
                for k in range(p):
                    M += np.outer(W[k, i], V[j, k])
                # Solve M @ u = 1
                try:
                    U[i, j] = np.linalg.solve(M, np.ones(rank))
                except np.linalg.LinAlgError:
                    U[i, j] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
        
        # Update V: for each (j,k), V[j,k,:] should satisfy sum_i U[i,j,:]*W[k,i,:] * V[j,k,:] = 1
        for j in range(m):
            for k in range(p):
                M = np.zeros((rank, rank))
                for i in range(n):
                    M += np.outer(U[i, j], W[k, i])
                try:
                    V[j, k] = np.linalg.solve(M, np.ones(rank))
                except np.linalg.LinAlgError:
                    V[j, k] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
        
        # Update W: for each (k,i), W[k,i,:] should satisfy sum_j V[j,k,:]*U[i,j,:] * W[k,i,:] = 1
        for k in range(p):
            for i in range(n):
                M = np.zeros((rank, rank))
                for j in range(m):
                    M += np.outer(V[j, k], U[i, j])
                try:
                    W[k, i] = np.linalg.solve(M, np.ones(rank))
                except np.linalg.LinAlgError:
                    W[k, i] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
        
        # Check convergence
        U3 = U
        V3 = V
        W3 = W
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(result - target))
        if err < tol:
            break
    
    return U, V, W, err

def _try_rank(rank, n, m, p, seed, deadline):
    """Try to find a valid decomposition of given rank."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n, m, rank) * 0.1
    V = np.random.randn(m, p, rank) * 0.1
    W = np.random.randn(p, n, rank) * 0.1
    
    U, V, W, err = _als_solve(U, V, W, n, m, p, max_iter=100, tol=1e-8, deadline=deadline)
    
    if err < 1e-6:
        # Convert to list format
        U_list = U.reshape(n*m, rank).tolist()
        V_list = V.reshape(m*p, rank).tolist()
        W_list = W.reshape(p*n, rank).tolist()
        return U_list, V_list, W_list, err
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Best known: schoolbook rank 48
    best_rank = n * m * p
    best_result = None
    
    # Build schoolbook as fallback
    U_sb = [[0] * best_rank for _ in range(n * m)]
    V_sb = [[0] * best_rank for _ in range(m * p)]
    W_sb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Based on theory, rank for [2,3,8] should be at most 28
    for rank in range(28, best_rank):
        if time.time() > deadline - 10:
            break
        found = False
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = _try_rank(rank, n, m, p, seed, deadline)
            if result is not None:
                U_list, V_list, W_list, err = result
                best_rank = rank
                best_result = (U_list, V_list, W_list)
                found = True
                break
        if found:
            continue
        # If rank not found, try next
        continue
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
