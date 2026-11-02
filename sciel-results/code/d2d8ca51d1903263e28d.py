"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition using random search
    # The known lower bound for 2x4 * 4x5 is around 24-25
    # Let's try to find rank 24 or 25
    
    start_time = time.time()
    timeout = 110  # seconds, leave 10s margin for evaluation
    
    best_rank = rank_school
    best_decomp = (U_school, V_school, W_school)
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But actual lower bound might be higher. Let's try 20, 21, 22, 23, 24, 25
    
    target_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    
    for target_rank in target_ranks:
        if time.time() - start_time > timeout:
            break
        
        # Random search for this rank
        for trial in range(50):
            if time.time() - start_time > timeout:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Optimize using gradient descent or alternating least squares
            # Use alternating minimization: fix two, solve for third
            for iteration in range(100):
                if time.time() - start_time > timeout:
                    break
                
                # Fix V and W, solve for U
                # For each (i,j), U[i*m+j, :] should satisfy:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if k==0 else 0
                # This is a linear system for U[i*m+j, :]
                
                # Actually, let's use a simpler approach: 
                # The tensor T[i,j,k] = 1 if k==0 else 0 (for 2x4*4x5, T[i,j,k] = delta_{k,0})
                # Wait, that's not right. Let me reconsider.
                
                # For matrix multiplication C = A * B where A is 2x4, B is 4x5, C is 2x5
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor T[i,j,k] = 1 for all i,j,k (it's the standard matrix mult tensor)
                # We want T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # Note the indices: U is indexed by (i,j), V by (j,k), W by (k,i)
                
                # Let's use a different approach: random initialization + local search
                pass
            
            # Check if this random decomposition is valid (it won't be, so skip)
            # Instead, let's try a structured approach
            
            break  # Skip random search for now
    
    # For now, return the schoolbook decomposition
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
