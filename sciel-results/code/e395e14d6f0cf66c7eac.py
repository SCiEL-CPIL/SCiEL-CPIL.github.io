import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    r = U.shape[1]
    # Check all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                # The tensor entry T_{i,j,k} is 1.
                # So sum_l U_{ij,l} V_{jk,l} W_{ki,l} must be 1.
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def _random_decomposition(n, m, p, r, max_val=2):
    """
    Generate a random decomposition with integer entries in [-max_val, max_val].
    """
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Initialize with random integers
    for _ in range(r):
        # To increase chance of success, we can try to make rows sparse
        # But let's just use random small integers
        U[:, _] = np.random.randint(-max_val, max_val + 1, size=n * m)
        V[:, _] = np.random.randint(-max_val, max_val + 1, size=m * p)
        W[:, _] = np.random.randint(-max_val, max_val + 1, size=p * n)
        
    return U, V, W

def _try_rank(n, m, p, r, time_limit=10):
    """
    Try to find a decomposition of rank r.
    """
    start_time = time.time()
    attempts = 0
    max_attempts = 1000
    
    while time.time() - start_time < time_limit and attempts < max_attempts:
        U, V, W = _random_decomposition(n, m, p, r)
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
        attempts += 1
        
    return None

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank
    # We know rank 60 works.
    # Let's try rank 59, 58... down to some limit.
    # Given the difficulty, we'll only try a few ranks.
    
    # Set a deadline for the whole construct function
    total_deadline = time.time() + 100  # 100s limit to be safe
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    
    # Generate schoolbook solution first to have a fallback
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
                
    # Try to improve
    # We will try ranks from rank_schoolbook - 1 down to rank_schoolbook - 10
    # If we find one, we return it.
    for r in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 0), -1):
        if time.time() > total_deadline:
            break
            
        # Allocate time for this rank
        time_left = total_deadline - time.time()
        if time_left < 2:
            break
            
        alloc_time = min(5, time_left - 1) # Spend max 5s per rank
        
        result = _try_rank(n, m, p, r, alloc_time)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists for output
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Return schoolbook
    return {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }
