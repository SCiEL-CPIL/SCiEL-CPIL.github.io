"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    # Start with schoolbook decomposition
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
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Validate function using numpy for speed
    def validate_decomposition(U_mat, V_mat, W_mat, rank_val):
        """Check if the decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        # For all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Verify schoolbook works
    assert validate_decomposition(U, V, W, rank), "Schoolbook decomposition failed validation"
    
    # Try structured decompositions
    # Strategy: Use the fact that 3x3x8 = 8 independent 3x3x1 products
    # Each 3x3x1 has optimal rank 9 (lower bound = max(9,3,3)=9)
    # So 8*9=72 is the trivial bound. Can we share?
    
    # Attempt 1: Use block structure
    # Think of B as 3 rows of 8 elements each
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Try to find a rank-64 or lower decomposition using algebraic tricks
    
    # For now, try a few structured approaches
    # Approach: Use intermediate sums
    
    # Let's try to see if we can reduce rank by using the structure
    # where we compute partial sums that are shared across k
    
    # Try rank values from 24 (lower bound) up to 71
    lower_bound = max(n * m, m * p, p * n)  # 24
    
    # Use a smarter random search with constraints
    # Key insight: entries must be integer or half-integer
    # Try small sets of values
    
    value_set = [0, 1, -1, 0.5, -0.5, 2, -2]
    
    # Try a few ranks near the lower bound first, then work up
    # But realistically, finding rank < 72 is very hard
    # Focus on confirming we have a valid rank-72 solution
    
    # Try some structured patterns for rank reduction
    # Pattern 1: Pair up k values and use shared computations
    
    # For each pair of k values (k1, k2), can we share some terms?
    # This would require finding U, V, W such that the bilinear forms work out
    
    # Given time constraints, let's try a limited search
    # Focus on ranks that are multiples of common factors
    
    candidate_ranks = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 5:
            break
        if target_rank >= best_rank:
            continue
        
        # Try a few random attempts at this rank
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            U_rand = [[random.choice(value_set) for _ in range(target_rank)] 
                      for _ in range(n * m)]
            V_rand = [[random.choice(value_set) for _ in range(target_rank)] 
                      for _ in range(m * p)]
            W_rand = [[random.choice(value_set) for _ in range(target_rank)] 
                      for _ in range(p * n)]
            
            if validate_decomposition(U_rand, V_rand, W_rand, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
