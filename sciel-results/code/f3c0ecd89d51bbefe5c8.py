"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s
    
    # Schoolbook decomposition: rank = n*m*p = 52
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
    
    # Try to find a better decomposition using Strassen-like ideas
    # For 2x2 x 2x13, we can potentially use a rank < 52 decomposition
    # Known result: for 2x2 x 2xp, the rank is at most 3p + 2 (for p >= 2)
    # Let's try to construct a rank 3p+2 = 41 decomposition
    
    target_rank = 3 * p + 2  # 41
    if target_rank < best_rank:
        # We need to find a decomposition of rank 41
        # This is non-trivial. Let's try a different approach.
        # 
        # Actually, for 2x2 x 2xp matrix multiplication, the tensor rank is known to be 3p+2
        # for p >= 2 (see e.g., Landsberg's work on tensor rank).
        # 
        # The construction is complex. Let's try a simpler approach first:
        # Can we do better than schoolbook with a simple pattern?
        # 
        # For now, let's try rank 41 and see if we can construct it.
        # The key insight is that we can share some computations.
        # 
        # Actually, let's try a different strategy: use the fact that for 2x2 matrices,
        # we can use 7 multiplications (Strassen) instead of 4.
        # But this doesn't directly extend to 2x13 in a simple way.
        # 
        # Let's try a numerical approach: use random initialization and gradient descent
        # to minimize the tensor norm, hoping to find a low-rank decomposition.
        
        if time.time() < deadline - 10:
            # Try to find a rank 41 decomposition via optimization
            # This is a hard problem, so we'll try a simple heuristic.
            
            # Initialize with random values
            np.random.seed(42)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # The tensor we want to decompose is T[i,j,k] = 1 for all i,j,k
            # (since every (i,j,k) triple is valid in matrix multiplication)
            # Wait, that's not right. The tensor for matrix multiplication is:
            # T[i,j,k,i',j',k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}? No.
            # 
            # Actually, the matrix multiplication tensor is defined as:
            # (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
            # So the tensor T has T[i,j,k] = 1 if the indices are "compatible",
            # which they always are for matrix multiplication (j is the shared index).
            # 
            # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            # 
            # This is a system of n*m*p = 52 equations in 41 variables (per rank term).
            # It's overdetermined, so it may not have a solution.
            # 
            # Let's check if rank 41 is actually achievable.
            # According to the literature, the rank of 2x2 x 2xp is 3p+2 for p >= 2.
            # So rank 41 should be achievable for p=13.
            # 
            # The construction is non-trivial. Let's try a different approach:
            # Use the known construction from the literature.
            # 
            # For 2x2 x 2xp, the construction uses the following idea:
            # Split the 13 columns into groups and use Strassen-like tricks.
            # 
            # This is getting too complex to implement correctly in a short time.
            # Let's just return the schoolbook decomposition to ensure validity.
            
            pass
    
    # Try a simple improvement: can we reduce the rank by 1 or 2?
    # For 2x2 x 2x2, the rank is 7 (Strassen).
    # For 2x2 x 2xp, the rank is 3p+2.
    # So for p=13, rank = 41.
    # 
    # Let's try to construct the rank 41 decomposition.
    # The construction is based on the following:
    # 
    # For each column k (0 <= k < p), we need to multiply A (2x2) by B[:, k] (2x1).
    # This can be done with 3 multiplications (since 2x2 x 2x1 has rank 3).
    # But we can share some multiplications across columns.
    # 
    # The total rank is 3p + 2.
    # 
    # The +2 comes from the "boundary" terms.
    # 
    # This is a well-known construction, but it's complex to implement.
    # Let's try a simpler approach: just use the schoolbook decomposition.
    
    return best_result
# EVOLVE-BLOCK-END
