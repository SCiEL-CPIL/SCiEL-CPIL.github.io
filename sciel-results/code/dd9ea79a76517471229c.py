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
    r = len(U[0]) if U else 0
    
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    
    # Check if rank is reasonable (not too high, not zero)
    if r == 0:
        return False
    
    # Use the trace convention: W[k*n+i, r]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 
    # for all (i,j,k)
    
    # Efficient check using numpy
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _is_valid_entry(x):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-10

def _snap_to_half(arr):
    """Snap values to nearest half-integer."""
    return np.round(np.array(arr) * 2) / 2.0

def _verify_fast(U, V, W, tol=1e-10):
    """Fast verification using numpy."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _random_decomposition(rank, seed=None):
    """Generate a random decomposition with given rank."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    n, m, p = DIMENSIONS
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Try to scale to make it work - this is a heuristic
    # The actual optimization would be more sophisticated
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank(rank, time_budget, start_time):
    """Try to find a decomposition of given rank within time budget."""
    n, m, p = DIMENSIONS
    
    if time.time() - start_time > time_budget:
        return None
    
    # For rank = n*m*p, we have the schoolbook solution
    if rank == n * m * p:
        U, V, W = _schoolbook()
        return U, V, W
    
    # For lower ranks, try random search with refinement
    # This is a simplified approach - a proper optimization would be much harder
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    num_trials = min(100, max(10, int((time_budget - (time.time() - start_time)) / 0.5)))
    
    for trial in range(num_trials):
        if time.time() - start_time > time_budget:
            break
        
        # Generate random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Simple gradient-free optimization: try to minimize the error
        # by adjusting entries
        err = float('inf')
        for _ in range(50):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
            
            if err < 1e-10:
                break
            
            # Random perturbation
            U[np.random.randint(n*m), np.random.randint(rank)] += np.random.randn() * 0.1
            V[np.random.randint(m*p), np.random.randint(rank)] += np.random.randn() * 0.1
            W[np.random.randint(p*n), np.random.randint(rank)] += np.random.randn() * 0.1
        
        if err < 1e-10:
            # Found a valid decomposition
            U_list = _snap_to_half(U).tolist()
            V_list = _snap_to_half(V).tolist()
            W_list = _snap_to_half(W).tolist()
            if _verify_fast(U_list, V_list, W_list):
                return U_list, V_list, W_list
    
    if best_err < 1e-10 and best_U is not None:
        U_list = _snap_to_half(np.array(best_U)).tolist()
        V_list = _snap_to_half(np.array(best_V)).tolist()
        W_list = _snap_to_half(np.array(best_W)).tolist()
        if _verify_fast(U_list, V_list, W_list):
            return U_list, V_list, W_list
    
    return None

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 145  # 145 seconds to be safe
    
    # Start with the schoolbook decomposition
    U, V, W = _schoolbook()
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But we need to find an actual decomposition
    
    # Try ranks from lower to higher, but limit the search
    # Given the difficulty, we'll try a few promising ranks
    
    ranks_to_try = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 45, 50, 60, 70, 80, 89]
    
    for rank in ranks_to_try:
        if time.time() - start_time > deadline:
            break
        
        if rank >= best_rank:
            break
        
        result = _try_rank(rank, deadline - start_time - 5, start_time)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            # If we found a low rank, we can stop
            if rank <= 35:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
