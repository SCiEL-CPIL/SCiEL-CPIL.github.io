import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for 120s limit

    # Build target tensor indices
    # T[i, j, k] = 1 for valid i,j,k
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    
    non_zero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                non_zero_indices.append((alpha, beta, gamma))
                
    num_non_zero = len(non_zero_indices)
    
    # Schoolbook rank-50 solution
    R_school = n * m * p
    U_school = np.zeros((n * m, R_school), dtype=np.float64)
    V_school = np.zeros((m * p, R_school), dtype=np.float64)
    W_school = np.zeros((p * n, R_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_rank = R_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Try to find a lower rank decomposition
    # We search for R from R_school - 1 down to a lower bound
    # Lower bound is max(n*m, m*p, p*n) = 25
    # We try a few ranks, prioritizing R_school - 1
    
    def verify(U, V, W, R):
        """
        Verify if U, V, W of rank R decompose the tensor.
        Only checks non-zero entries of T_target and ensures zeros elsewhere.
        """
        # Check non-zero entries
        for idx in non_zero_indices:
            a, b, c = idx
            val = 0.0
            for r in range(R):
                val += U[a, r] * V[b, r] * W[c, r]
            if abs(val - 1.0) > 1e-9:
                return False
                
        # Check zero entries
        # To speed up, we can sample some zero entries or check all if fast enough
        # Total entries = 10 * 25 * 10 = 2500
        # Non-zero = 50
        # Zero = 2450
        # Checking all is 2450 * R operations.
        # If R=49, 2450 * 49 ~ 120k ops. Very fast.
        
        # Create a set of non-zero indices for fast lookup
        nz_set = set(non_zero_indices)
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in nz_set:
                        continue
                    val = 0.0
                    for r in range(R):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val) > 1e-9:
                        return False
        return True

    # Search strategy:
    # Random search with integer entries from {-1, 0, 1} is too slow for R=49.
    # Instead, we can try to perturb the schoolbook solution.
    # Or use a local search.
    
    # Given the time constraint and difficulty, we will try a limited random search
    # for R = 49, 48, 47.
    
    for R in range(R_school - 1, max(24, R_school - 5), -1):
        if time.time() > deadline:
            break
            
        # Number of attempts
        max_attempts = 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Using small integers
            U = np.random.randint(-1, 2, (n * m, R))
            V = np.random.randint(-1, 2, (m * p, R))
            W = np.random.randint(-1, 2, (p * n, R))
            
            # Quick check: sum of squares of products must match?
            # Or just verify.
            
            if verify(U, V, W, R):
                best_rank = R
                best_U = U.astype(np.float64)
                best_V = V.astype(np.float64)
                best_W = W.astype(np.float64)
                break
                
        if best_rank < R_school:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
