import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 4]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_vector(T, n, m, p):
    """Convert 3D tensor to flat vector."""
    return T.flatten()

def _vector_to_tensor(v, n, m, p):
    """Convert flat vector to 3D tensor."""
    return v.reshape((n, m, p))

def _check_decomposition(U, V, W, T, n, m, p, tol=1e-9):
    """Check if U, V, W decompose tensor T."""
    rank = U.shape[1]
    # Compute sum of outer products
    result = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u = U[:, r]  # shape (n*m,)
        v = V[:, r]  # shape (m*p,)
        w = W[:, r]  # shape (p*n,)
        # u is indexed by (i,j), v by (j,k), w by (k,i)
        # We need to verify: sum_r u[i*m+j] * v[j*p+k] * w[k*n+i] == T[i,j,k]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    return np.allclose(result, T, atol=tol)

def _try_rank_r(n, m, p, r, max_iter=10000, time_budget=10.0):
    """Try to find a rank-r decomposition using randomized local search."""
    T = _build_tensor(n, m, p)
    start_time = time.time()
    
    # Initialize with random values
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    error += (val - T[i, j, k]) ** 2
        error = np.sqrt(error)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-9:
            return best_U, best_V, best_W, True
        
        # Random perturbation
        factor = random.choice([U, V, W])
        row = random.randint(0, factor.shape[0] - 1)
        col = random.randint(0, r - 1)
        factor[row, col] += random.gauss(0, 0.1)
    
    # Check if best solution is valid
    if best_error < 1e-9:
        return best_U, best_V, best_W, True
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # We want to find a lower rank decomposition.
    # We will attempt to find a decomposition of rank r < rank_school.
    # Given the small size, we can try to find a rank 19 or 20 decomposition.
    # However, finding it from scratch is hard.
    # We will use a randomized local search to try to reduce the rank.
    
    # To ensure we return a valid solution, we keep the schoolbook solution as a fallback.
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0 # Leave margin
    
    # Try to find lower rank decompositions
    for target_rank in [23, 22, 21, 20, 19]:
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try multiple random restarts
        found = False
        for restart in range(10):
            if time.time() - start_time > time_limit - 5:
                break
            
            U_try, V_try, W_try, success = _try_rank_r(n, m, p, target_rank, max_iter=5000, time_budget=5.0)
            
            if success and U_try is not None:
                # Verify the solution
                if _check_decomposition(U_try, V_try, W_try, _build_tensor(n, m, p), n, m, p):
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
