"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Build the schoolbook decomposition first as a fallback
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    for r in range(rank_schoolbook):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        
        U_school[idx_u][r] = 1
        V_school[idx_v][r] = 1
        W_school[idx_w][r] = 1
    
    # Try to find a lower-rank decomposition using tensor rank algorithms
    # The matrix multiplication tensor T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # We want to decompose T as sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # For small dimensions, we can try random initialization + iterative refinement
    # But first, let's check known bounds:
    # For 2x3 * 3x10, the rank is at most 60 (schoolbook)
    # Lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    
    # Try a few random attempts to find rank < 60
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Attempt to find a lower rank decomposition
    # Strategy: try ranks from 30 upward, use random initialization and check validity
    # For each rank r, we need U (6 x r), V (30 x r), W (20 x r)
    # such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # This is a hard problem. Let's try a few random attempts with small ranks.
    # If we can't find one quickly, return the schoolbook.
    
    np.random.seed(42)
    
    # Try rank 30, 35, 40, 45, 50 with a few random attempts each
    for target_rank in [30, 35, 40, 45, 50]:
        if time.time() > deadline - 20:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Check if this decomposition is valid
            # For each (i, j, k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This should equal 1 for all i, j, k
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list of lists with integer/half-integer entries if possible
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best_rank < rank_schoolbook:
            break
    
    return best_result
# EVOLVE-BLOCK-END
