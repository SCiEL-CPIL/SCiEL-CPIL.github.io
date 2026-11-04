"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # We need U @ V^T element-wise * W such that the sum equals T
    # Actually: sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But actual lower bound is higher
    
    # Try ranks from best_rank-1 down to a reasonable lower bound
    lower_bound = max(n * m, m * p, p * n)  # 24
    
    for rank in range(best_rank - 1, max(lower_bound - 1, 20), -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Scale to make entries reasonable
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Check if this decomposition works (it won't for random init)
            # We need a better approach: solve the system
            # For now, just check if we can find a valid one via search
            # This is a placeholder - real search would be more sophisticated
            pass
    
    return best_result
# EVOLVE-BLOCK-END
