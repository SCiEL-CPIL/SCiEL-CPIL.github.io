"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # The condition for the decomposition is:
    # For all i in [0, n), j in [0, m), k in [0, p):
    #   sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
    
    # First, try the schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
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
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Try ranks from 27 up to schoolbook-1
    
    def check_decomposition(U, V, W, rank, tol=1e-10):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(rank, max_iter=5000):
        """Try to find a valid decomposition of given rank using random search."""
        if time.time() > deadline:
            return None
        
        # Random initialization with small values
        for attempt in range(20):
            if time.time() > deadline:
                return None
            
            # Initialize with random values in [-1, 1]
            U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
            
            # Local search: try to reduce error
            for iteration in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Compute current error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            error += (s - 1.0) ** 2
                
                if error < 1e-20:
                    # Check exactness with higher precision
                    if check_decomposition(U, V, W, rank, tol=1e-15):
                        return {"U": U, "V": V, "W": W}
                
                # Pick a random element to perturb
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, n*m-1)
                    col = random.randint(0, rank-1)
                    old_val = U[row][col]
                    U[row][col] = old_val + random.gauss(0, 0.1)
                elif mat == 'W':
                    row = random.randint(0, p*n-1)
                    col = random.randint(0, rank-1)
                    old_val = W[row][col]
                    W[row][col] = old_val + random.gauss(0, 0.1)
                else:
                    row = random.randint(0, m*p-1)
                    col = random.randint(0, rank-1)
                    old_val = V[row][col]
                    V[row][col] = old_val + random.gauss(0, 0.1)
                
                # Compute new error
                new_error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            new_error += (s - 1.0) ** 2
                
                # Accept if better, or with some probability (simulated annealing)
                if new_error < error:
                    pass  # Keep the change
                else:
                    # Revert
                    if mat == 'U':
                        U[row][col] = old_val
                    elif mat == 'V':
                        V[row][col] = old_val
                    else:
                        W[row][col] = old_val
        
        return None
    
    # Try ranks from lower bound up
    lower_bound = max(n*m, m*p, p*n)
    
    for rank in range(lower_bound, rank_schoolbook):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(rank, max_iter=2000)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found the minimum rank
    
    return best_result

# EVOLVE-BLOCK-END
