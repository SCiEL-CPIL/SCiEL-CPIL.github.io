"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(n, m, p, U, V, W, atol=1e-9):
    """Verify that the tensor decomposition is exact for matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    target = np.ones((n, m, p))
    return np.allclose(result, target, atol=atol)

def _schoolbook_decomposition(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _try_structured_decomposition(n, m, p, rank, seed=42, max_iters=20000):
    """
    Try to find a rank-decomposition using structured initialization and optimization.
    Uses alternating least squares (ALS) approach.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with structured patterns
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Initialize with random values scaled appropriately
    U = rng.randn(n * m, rank) * (1.0 / np.sqrt(rank))
    V = rng.randn(m * p, rank) * (1.0 / np.sqrt(rank))
    W = rng.randn(p * n, rank) * (1.0 / np.sqrt(rank))
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    target = np.ones((n, m, p))
    
    for it in range(max_iters):
        # ALS: update U, V, W in turn
        # Update U: for each (i,j), U[i,j,:] should satisfy sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all k
        # This is a linear system in U[i,j,:]
        for i in range(n):
            for j in range(m):
                # Build matrix A where A[k, c] = sum over... actually it's:
                # sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c]) = 1 for all k
                # So: M @ U[i,j,:] = 1 where M[k,c] = V[j,k,c] * W[k,i,c]
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M @ x = ones(p)
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i * m + j, :] = sol
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j * p + k, :] = sol
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k * n + i, :] = sol
                except:
                    pass
        
        # Compute error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.mean((result - target) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
    
    if best_err < 1e-8:
        # Round to nearest half-integer if possible
        def round_to_half(arr):
            return np.round(arr * 2) / 2
        
        U_r = round_to_half(best_U)
        V_r = round_to_half(best_V)
        W_r = round_to_half(best_W)
        
        # Check if rounded version still works
        if _verify_decomposition(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return U_r, V_r, W_r, best_err
        else:
            return best_U, best_V, best_W, best_err
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    """
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    fallback = _schoolbook_decomposition(n, m, p)
    best_result = fallback
    best_rank = schoolbook_rank
    
    # Lower bound: rank >= max(n*p, m*p, n*m)
    lower_bound = max(n * p, m * p, n * m)
    
    # Try ranks from lower_bound upward, but limit the range
    # Focus on ranks that might have known constructions
    ranks_to_try = list(range(lower_bound, min(lower_bound + 30, schoolbook_rank + 1)))
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple seeds for this rank
        for seed in range(20):
            if time.time() > deadline:
                break
                
            result = _try_structured_decomposition(n, m, p, rank, seed=seed, max_iters=5000)
            if result is not None:
                U, V, W, err = result
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify
                if _verify_decomposition(n, m, p, U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    # Found a valid decomposition at this rank
                    # Continue to see if we can find even lower (but we're going ascending, so this is the first)
                    # Actually, we're trying in ascending order, so this is the minimal
                    return best_result
    
    # If no better decomposition found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
