import numpy as np
import time
from itertools import product
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Schoolbook rank as baseline
    rank_schoolbook = n * m * p  # 36
    
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def verify_fast(U, V, W):
        """Vectorized verification using tensor contraction."""
        R = U.shape[1]
        if V.shape[1] != R or W.shape[1] != R:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        
        # Check tensor equation: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # T[i,j,k] should be 1 for all valid i, j, k
        # Use einsum for efficient computation
        # Reshape U, V, W to 3D tensors
        U_3d = U.reshape(n, m, R)  # U[i,j,r]
        V_3d = V.reshape(m, p, R)  # V[j,k,r]
        W_3d = W.reshape(p, n, R)  # W[k,i,r]
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a bit tricky with einsum. Let's do it step by step.
        # First compute A[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We can think of this as: for each (i,j,k), sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r]
        
        # One approach: iterate over k since p is small
        for k in range(p):
            # For fixed k, we need T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
            # Let B[j,i,r] = V[j,k,r] * W[k,i,r]
            V_k = V_3d[:, k, :]  # shape (m, R) -> V[j,k,r]
            W_k = W_3d[k, :, :]  # shape (n, R) -> W[k,i,r]
            # B[j,i,r] = V_k[j,r] * W_k[i,r]
            B = np.einsum('jr,ir->jir', V_k, W_k)  # shape (m, n, R)
            # Now T[i,j,k] = sum_r U[i,j,r] * B[j,i,r]
            # = sum_r U[i,j,r] * B[j,i,r]
            # U[i,j,r] has shape (n,m,R), B[j,i,r] has shape (m,n,R)
            # We need to align: U[i,j,r] * B[j,i,r]
            # Let's transpose B to get B'[i,j,r] = B[j,i,r]
            B_t = B.transpose(1, 0, 2)  # shape (n, m, R), B_t[i,j,r] = B[j,i,r] = V_k[j,r]*W_k[i,r]
            # Now T[i,j,k] = sum_r U[i,j,r] * B_t[i,j,r]
            T_k = np.einsum('ijr,ijr->ij', U_3d, B_t)
            # Check if all entries are approximately 1
            if not np.allclose(T_k, 1.0, atol=1e-10):
                return False
        return True
    
    # Try to find a lower rank decomposition
    best_rank = rank_schoolbook
    best_U, best_V, best_W = build_schoolbook()
    
    # Strategy: Use random search with small integer/half-integer entries
    # Try ranks from lower bound up to schoolbook - 1
    
    # Lower bound: max(n*m, m*p, n*p) = max(6, 18, 12) = 18
    lower_bound = max(n * m, m * p, n * p)
    
    # Entry values to try: {-1, -0.5, 0, 0.5, 1}
    entry_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Try each rank from lower_bound to schoolbook - 1
    for R in range(lower_bound, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Number of random trials for this rank
        max_trials = 1000
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Generate random U, V, W with entries from entry_values
            U = np.random.choice(entry_values, size=(n * m, R))
            V = np.random.choice(entry_values, size=(m * p, R))
            W = np.random.choice(entry_values, size=(p * n, R))
            
            # Verify
            if verify_fast(U, V, W):
                best_rank = R
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
    
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
