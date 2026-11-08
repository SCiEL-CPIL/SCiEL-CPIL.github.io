"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Use a search approach
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition.
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        """
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U[i * m + j, :]
                    v = V[j * p + k, :]
                    w = W[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def error_decomposition(U, V, W):
        """Compute the error of the decomposition."""
        r = U.shape[1]
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U[i * m + j, :]
                    v = V[j * p + k, :]
                    w = W[k * n + i, :]
                    val = np.dot(u * v, w)
                    total_err += (val - 1) ** 2
        return total_err
    
    # Try various ranks from low to high
    # Start with a reasonable guess
    for target_rank in [49, 56, 64, 72, 80, 88, 96]:
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try to find a decomposition at this rank
        found = False
        attempts = 0
        max_attempts = 50  # Limit attempts per rank
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random initialization
            U = np.random.randint(-2, 3, (n * m, target_rank))
            V = np.random.randint(-2, 3, (m * p, target_rank))
            W = np.random.randint(-2, 3, (p * n, target_rank))
            
            # Check if valid
            if check_decomposition(U, V, W):
                found = True
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if found:
            # Try to go lower
            continue
        else:
            # This rank didn't work, try higher
            continue
    
    return best_result
# EVOLVE-BLOCK-END
