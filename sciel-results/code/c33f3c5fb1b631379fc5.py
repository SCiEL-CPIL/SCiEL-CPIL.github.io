"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 45
    
    # Return the schoolbook decomposition as baseline
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a better decomposition using Strassen-based approach
    # For 3x3 * 3x5, we can use block decomposition:
    # Split 3x3 into 2x2 + 1, and 3x5 into 2x5 + 1x5
    # Use Strassen (rank 7) for 2x2*2x2, and trivial for rest
    
    # Actually, let's try a direct approach with random search for lower rank
    # The known rank for 3x3*3x5 might be less than 45
    
    # Let's implement a simple gradient-free search to see if we can beat 45
    # But first, let's verify that 45 works (it does, it's schoolbook)
    
    # Try rank 44, 43, ... down to some minimum
    # Use random initialization and check if the tensor equation holds
    
    def check_decomposition(U_mat, V_mat, W_mat, rank_test):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] for (C = AB)[i,k] = sum_j A[i,j] B[j,k]
        # In our indexing: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # W is indexed as W[k*n + i, r]
        # The decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And T[i,j,k] = 1 if we're looking at the standard tensor (each entry is 1 for valid i,j,k)
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Build the left side: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 for all i,j,k
        
        # Efficient check using vectorized operations
        import numpy as np
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_np.reshape(n, m, rank_test)
        V_reshaped = V_np.reshape(m, p, rank_test)
        W_reshaped = W_np.reshape(p, n, rank_test)
        
        # Compute sum_r U[i,m,j,r] * V[j,r,k] * W[k,n,i,r]
        # This is a tensor contraction
        # Result[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,r,k] * W_reshaped[k,n=i,r]
        
        # Let's compute this step by step
        # First: for each i,j,k, compute the sum
        # Use einsum: sum_r U[i,j,r] * V[j,r,k] * W[k,i,r]
        # But W is indexed as W[k*n+i, r], so W_reshaped[k, i, r]
        
        # Compute intermediate: U[i,j,r] * V[j,r,k] summed over r gives M[i,j,k]
        # Then multiply by W[k,i,r] and sum over r... wait, that's not right.
        
        # Actually: T[i,j,k] = sum_r U[i,j,r] * V[j,r,k] * W[k,i,r]
        
        # Let's use numpy einsum
        # U shape: (n, m, rank)
        # V shape: (m, p, rank)  
        # W shape: (p, n, rank)
        
        # We want: result[i,j,k] = sum_r U[i,j,r] * V[j,r,k] * W[k,i,r]
        
        # Step 1: Compute A[i,j,k] = sum_r U[i,j,r] * V[j,r,k]
        A_inter = np.einsum('ijr,jkr->ijk', U_reshaped, V_reshaped)
        
        # Step 2: result[i,j,k] = sum_r A[i,j,k] * W[k,i,r] ... no, that's not right either
        # Actually, we need to be more careful.
        
        # The full expression is: sum_r U[i,j,r] * V[j,r,k] * W[k,i,r]
        # This can't be done in two simple steps because the r index is shared among all three.
        
        # Let's just do it directly:
        # For each (i,j,k), compute sum over r of U[i,j,r] * V[j,r,k] * W[k,i,r]
        
        # Vectorized approach:
        # Create a 4D array and sum
        # U_reshaped: (n, m, rank) -> expand to (n, m, 1, rank)
        # V_reshaped: (m, p, rank) -> expand to (1, m, p, rank)
        # W_reshaped: (p, n, rank) -> expand to (p, n, 1, rank)
        
        # This is getting complicated. Let's just iterate for small sizes.
        
        # Since n=3, m=3, p=5, total entries = 45, and rank <= 45, this is feasible
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank_test):
                        s += U_mat[i*m + j][r] * V_mat[j*p + k][r] * W_mat[k*n + i][r]
                    if s != 1:  # Should be exactly 1
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Start from rank 44 down to some reasonable minimum
    # Use random search with integer/half-integer values
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try a few random searches for lower rank
    for target_rank in range(rank - 1, max(20, rank - 10), -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random attempts for this rank
        attempts = 5
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers and half-integers
            U_test = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(n*m)]
            V_test = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(m*p)]
            W_test = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if check_decomposition(U_test, V_test, W_test, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_test, V_test, W_test
                break
        else:
            # If we didn't find a solution for this rank, continue to lower rank
            continue
        break  # Found a better rank, stop searching
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
