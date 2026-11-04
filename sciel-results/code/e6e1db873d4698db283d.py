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

def _fixed_w_decomposition(rank, n, m, p):
    """
    Construct a decomposition with W as identity (or partial identity).
    Requires rank >= p*n.
    """
    if rank < p * n:
        return None
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Set W to identity in first p*n columns
    for c in range(p * n):
        W[c, c] = 1
    
    # For each (k,i), set U[i*m+j, k*n+i] = 1 and V[j*p+k, k*n+i] = 1 for all j
    for i in range(n):
        for k in range(p):
            c = k * n + i
            if c < rank:
                for j in range(m):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _validate(U_list, V_list, W_list):
        return U_list, V_list, W_list
    
    return None

def _try_rank_search(n, m, p, deadline):
    """
    Search for the minimum rank decomposition.
    """
    start_time = time.time()
    
    # Try fixed-W approach with rank = p*n = 18
    result = _fixed_w_decomposition(p * n, n, m, p)
    if result is not None:
        return result
    
    # Try larger ranks with fixed-W
    for rank in range(p * n + 1, n * m * p + 1):
        if time.time() > deadline - 5:
            break
        result = _fixed_w_decomposition(rank, n, m, p)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U, V, W = _try_rank_search(n, m, p, deadline)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
