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

def _try_decomposition(rank, n, m, p, max_iterations=1000, tol=1e-6):
    """
    Try to find a decomposition of given rank using alternating minimization.
    Returns (U, V, W) as numpy arrays if found, None otherwise.
    """
    # Initialize with random values
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iterations):
        # Update U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        # This is a linear system in U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # Build the system: for each k, sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # Let A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise)
                # Then sum_c U[i*m+j, c] * A_k[c] = 1 for all k
                # This is: M @ U[i*m+j, :] = ones(p) where M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                # Solve M @ u = ones(p)
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
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j] * V[j*p + k] * W[k*n + i])
                    error += (val - 1.0) ** 2
        error = np.sqrt(error)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        return best_U, best_V, best_W
    return None

def _round_to_half_integers(arr):
    """Round array to nearest half-integers."""
    return np.round(arr * 2) / 2

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
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
    
    # Try to find lower rank decompositions
    # Start from a reasonable upper bound and go down
    # Known lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But actual rank might be higher
    
    # Try ranks from a bit below schoolbook down to the lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a range of ranks, starting from slightly below schoolbook
    ranks_to_try = []
    for r in range(schoolbook_rank - 1, lower_bound - 1, -1):
        ranks_to_try.append(r)
    
    # Limit the number of ranks to try based on time
    max_ranks_to_try = 20
    
    for rank in ranks_to_try[:max_ranks_to_try]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt * 1000 + rank)
            result = _try_decomposition(rank, n, m, p, max_iterations=200)
            
            if result is not None:
                U_np, V_np, W_np = result
                
                # Round to half-integers
                U_rounded = _round_to_half_integers(U_np)
                V_rounded = _round_to_half_integers(V_np)
                W_rounded = _round_to_half_integers(W_np)
                
                # Check if rounded version is exact
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if _validate(U_list, V_list, W_list):
                    # Found a valid decomposition!
                    best_U, best_V, best_W = U_list, V_list, W_list
                    best_rank = rank
                    break
        
        if best_rank < schoolbook_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
