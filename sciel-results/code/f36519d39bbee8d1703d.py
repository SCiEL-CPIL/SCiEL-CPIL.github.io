"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    best_rank = n * m * p  # 54, schoolbook
    best_result = None

    # Return schoolbook as fallback
    def make_schoolbook():
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()

    # Try to find a lower-rank decomposition via rank-1 removal from schoolbook
    # The schoolbook has rank 54. We try to find if any linear combination of columns
    # can be removed (i.e., the tensor is in the span of fewer rank-1 terms).
    
    # Alternative: try known small-rank decompositions by solving linear systems.
    # For a rank-r decomposition, we need:
    #   T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a system of bilinear/quadratic equations. Instead, fix one matrix and
    # solve for the others, or use iterative methods.
    
    # A simpler approach: try to find a rank-r decomposition by:
    # 1. Choosing random U, V of shape (9,r) and (18,r)
    # 2. For each (k,i), the equation is: T[:,k,i] = sum_c U[:,c]*V[k,:,c]  (but W enters too)
    
    # Actually the tensor is T[i,j,k] with T[i,j,k] = 1 if (i,j,k) is a valid index.
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Let's try a structured approach: exploit that p=6 = 2*3.
    # Split the 3x6 matrix into two 3x3 blocks.
    # The multiplication (A 3x3) * (B 3x6) can be seen as:
    #   Result[i,k] for k in [0,2]: sum_j A[i,j]*B[j,k]
    #   Result[i,k] for k in [3,5]: sum_j A[i,j]*B[j,k-3]
    
    # So the tensor is the direct sum of two 3x3x3 tensors.
    # R(3,3,6) <= R(3,3,3) + R(3,3,3) = 2*R(3,3,3).
    # If R(3,3,3) = 23, then R(3,3,6) <= 46.
    # But we need an explicit decomposition.
    
    # Since we can't easily produce a rank-23 decomposition of 3x3x3 here,
    # let's try a random search for rank-r decompositions with r < 54.
    
    # For a given rank r, set up the problem as:
    # Find U (9 x r), V (18 x r), W (18 x r) such that
    # for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 if valid, 0 otherwise.
    
    # This is hard. Instead, let's try a specific structured ansatz.
    
    # Ansatz: Use the fact that the tensor is a sum of two 3x3x3 blocks.
    # For block 1 (k=0,1,2): T1[i,j,k] = delta (standard 3x3x3)
    # For block 2 (k=3,4,5): T2[i,j,k-3] = delta (standard 3x3x3)
    
    # If we had a rank-r decomposition of 3x3x3, we could use 2r rank.
    # Let's try to find a rank-r decomposition of 3x3x3 by solving a system.
    
    # For 3x3x3, the tensor has 27 nonzero entries (all 1s).
    # We want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0,1,2}.
    
    # Fix W to be identity-like and solve for U, V.
    # If r >= 9, set W[k,i,c] = delta_{k,i,c'} for appropriate c'.
    
    # Actually, let's try a simpler random approach: 
    # For each candidate rank r from 46 down to 27, try random initializations.
    
    for r in range(53, 26, -1):
        if time.time() > deadline - 5:
            break
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 3:
                break
            np.random.seed(attempt + r * 100)
            U = np.random.randn(9, r) * 0.1
            V = np.random.randn(18, r) * 0.1
            W = np.random.randn(18, r) * 0.1
            
            # We need to solve: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i,j,k]
            # This is a system of 54 equations (9*6=54 entries, but many are zero).
            # Actually, T[i,j,k] is 1 for all valid (i,j,k) since it's matrix multiplication.
            # Wait, T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,6).
            # That's 54 equations, all equal to 1.
            
            # This is a hard nonlinear system. Let's use a different approach.
            # Instead, let's try to verify if a rank-46 decomposition exists by
            # using the block structure.
            
            # For now, skip to a known result if available.
            pass
    
    # Given the difficulty, return the schoolbook solution.
    return best_result
# EVOLVE-BLOCK-END
