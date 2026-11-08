import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Try to find a decomposition with rank < 75
    # We will try ranks from 70 down to 50
    # For each rank, we try a few random initializations
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback to schoolbook if time runs out
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
        return U, V, W, rank

    start_time = time.time()
    deadline = 100.0 # Leave 20s for safety
    
    for rank in range(70, 49, -1):
        if time.time() > deadline:
            break
            
        # Try a few random attempts for this rank
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Initialize with small integers
            U = np.random.randint(-2, 3, size=(n * m, rank))
            V = np.random.randint(-2, 3, size=(m * p, rank))
            W = np.random.randint(-2, 3, size=(p * n, rank))
            
            # Check if it satisfies the condition
            # Sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k
            # This is equivalent to checking the tensor product
            # We can vectorize this check
            
            # Reshape U to (n, m, rank)
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            # Compute the sum for each i, j, k
            # This is a tensor contraction
            # Sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            
            # We can loop over i, j, k to check
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute sum
                        # U[i, j, :] * V[j, k, :] * W[k, i, :]
                        s = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break # Found a solution for this rank, try lower rank
        
        if best_rank < n * m * p:
            break # Found a solution, stop searching for lower rank? 
            # Actually, we want the MINIMAL rank. So we should continue if we want to be sure.
            # But finding a lower rank is hard. Let's just return the first one found.
            # Or continue to see if we can go lower.
            # Given the time limit, returning the first found is safer.
            
    if best_rank < n * m * p:
        # Convert to lists
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        U, V, W, _ = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
