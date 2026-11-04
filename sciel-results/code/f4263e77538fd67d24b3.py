import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _validate(U, V, W):
    """
    Validate that U, V, W form an exact decomposition of the matmul(3,5,6) tensor.
    Returns True if valid, False otherwise.
    """
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0])
    
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    if r == 0:
        return False
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m + j]
                v_val = V_np[j*p + k]
                w_val = W_np[k*n + i]
                # Element-wise product and sum
                val = np.sum(u_val * v_val * w_val)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the decomposition error."""
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m + j] * V[j*p + k] * W[k*n + i])
                error += (val - 1.0) ** 2
    return np.sqrt(error)

def _try_decomposition(rank, n, m, p, max_iterations=500, tol=1e-10, init_scale=1.0):
    """
    Try to find a decomposition of given rank using alternating minimization.
    Returns (U, V, W) as numpy arrays if found, None otherwise.
    """
    # Initialize with random values scaled appropriately
    U = np.random.randn(n * m, rank) * init_scale
    V = np.random.randn(m * p, rank) * init_scale
    W = np.random.randn(p * n, rank) * init_scale
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iterations):
        # Update U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    u_new = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    u_new = U[i*m + j]
                U[i*m + j] = u_new
        
        # Update V: for each (j,k), V[j*p+k, :] should satisfy
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    v_new = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    v_new = V[j*p + k]
                V[j*p + k] = v_new
        
        # Update W: for each (k,i), W[k*n+i, :] should satisfy
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    w_new = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    w_new = W[k*n + i]
                W[k*n + i] = w_new
        
        # Compute error
        error = _compute_error(U, V, W, n, m, p)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        return best_U, best_V, best_W
    return None

def _polish_decomposition(U, V, W, n, m, p, max_iterations=200, tol=1e-12):
    """
    High-precision polish pass to refine an approximate decomposition.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for iteration in range(max_iterations):
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, U.shape[1]))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    u_new = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    continue
                U[i*m + j] = u_new
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, V.shape[1]))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    v_new = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    continue
                V[j*p + k] = v_new
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, W.shape[1]))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    w_new = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    continue
                W[k*n + i] = w_new
        
        error = _compute_error(U, V, W, n, m, p)
        if error < tol:
            break
    
    if _compute_error(U, V, W, n, m, p) < tol:
        return U, V, W
    return None

def _round_to_half_integers(arr):
    """Round array to nearest half-integers."""
    return np.round(arr * 2) / 2

def _try_rank_with_polish(rank, n, m, p, deadline, max_attempts=8):
    """
    Try to find a valid decomposition of given rank with polish pass.
    Returns (U_list, V_list, W_list) if found, None otherwise.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline - 3:
            break
        
        np.random.seed(attempt * 7919 + rank * 104729)
        
        # Try different initialization scales
        init_scale = 1.0 + 0.5 * (attempt % 3)
        
        result = _try_decomposition(rank, n, m, p, max_iterations=300, tol=1e-8, init_scale=init_scale)
        
        if result is not None:
            U_np, V_np, W_np = result
            
            # Try polishing
            polished = _polish_decomposition(U_np, V_np, W_np, n, m, p, max_iterations=100, tol=1e-12)
            if polished is not None:
                U_np, V_np, W_np = polished
            
            # Round to half-integers
            U_rounded = _round_to_half_integers(U_np)
            V_rounded = _round_to_half_integers(V_np)
            W_rounded = _round_to_half_integers(W_np)
            
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if _validate(U_list, V_list, W_list):
                return U_list, V_list, W_list
    
    return None

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = [row[:] for row in U_sb], [row[:] for row in V_sb], [row[:] for row in W_sb]
    best_rank = schoolbook_rank
    
    # Known lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    lower_bound = max(n * m, m * p, p * n)
    
    # Strategy: Try ranks near the lower bound first, then expand outward
    # This is more efficient than trying from schoolbook downward
    
    # Build a list of ranks to try, prioritizing lower ranks
    ranks_to_try = []
    
    # First, try from lower_bound upward to a reasonable point
    for r in range(lower_bound, min(lower_bound + 15, schoolbook_rank)):
        ranks_to_try.append(r)
    
    # Then try from schoolbook downward (in case lower ranks fail)
    for r in range(schoolbook_rank - 1, lower_bound + 14, -1):
        if r not in ranks_to_try:
            ranks_to_try.append(r)
    
    # Limit total ranks to try based on time
    max_ranks_to_try = 25
    
    for rank in ranks_to_try[:max_ranks_to_try]:
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_with_polish(rank, n, m, p, deadline, max_attempts=6)
        
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
