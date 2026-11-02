"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 60).
    2. Attempt a limited search for lower ranks using randomized numerical optimization.
    3. If a valid integer/half-integer decomposition is found with rank < 60, return it.
    4. Otherwise, return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p
    
    # Helper to check if a decomposition is valid and exact
    def is_valid(U, V, W, rank):
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    # val should be close to k/2
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        # Check tensor decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Search for lower ranks
    # We try ranks from 59 down to 50. 
    # For each rank, we try a few random initializations and short gradient descents.
    for rank in range(59, 49, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random starts
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(rank * 100 + trial)
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Simple coordinate descent / gradient step
            # Objective: sum_{i,j,k} (sum_c U[i,j,c]V[j,k,c]W[k,i,c] - 1)^2
            for iter in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            grad = 2 * (s - 1) * V[j*p+k] * W[k*n+i]
                            U[i*m+j] -= 0.01 * grad
                # Update V
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            grad = 2 * (s - 1) * U[i*m+j] * W[k*n+i]
                            V[j*p+k] -= 0.01 * grad
                # Update W
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            grad = 2 * (s - 1) * U[i*m+j] * V[j*p+k]
                            W[k*n+i] -= 0.01 * grad
            
            # Check if we found a valid solution
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if is_valid(U_r.tolist(), V_r.tolist(), W_r.tolist(), rank):
                best_rank = rank
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                # We found a lower rank, break out of loops
                return best_result

    return best_result
# EVOLVE-BLOCK-END
