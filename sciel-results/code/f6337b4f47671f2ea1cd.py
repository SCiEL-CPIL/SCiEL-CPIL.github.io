"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # We search for a rank-minimal exact tensor decomposition for multiplying
    # a 3x3 matrix by a 3x6 matrix.
    # The multiplication tensor T has dimensions (n*m) x (m*p) x (p*n) = 9 x 18 x 18.
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # The schoolbook decomposition has rank n*m*p = 54.
    # We look for a lower rank decomposition.
    #
    # Strategy: We use a structured search. 
    # First, we try to find a decomposition with rank r < 54.
    # We use a randomized search with local optimization.
    #
    # The key insight is that the multiplication tensor for (3,3,6) can be viewed
    # as a block structure. The 3x6 matrix has two 3x3 blocks.
    # Let A be 3x3, B = [B1 B2] where B1, B2 are 3x3.
    # C = A * B = [A*B1, A*B2].
    #
    # If we had a rank r decomposition for 3x3*3x3, we could use it for both blocks.
    # But that would give rank 2r, which is not better than 54 if r > 27.
    #
    # Actually, let's think about the structure more carefully.
    # The tensor T[i*m+j, j*p+k, k*n+i] = delta(i,i') delta(j,j') delta(k,k')
    # where the indices are grouped.
    #
    # For n=m=p=3, the rank is 27 (schoolbook). It is known that the rank of 3x3 matrix
    # multiplication is at most 23 (Strassen's algorithm gives rank 7 for 2x2, and for 3x3
    # there are algorithms with rank 23).
    #
    # For 3x3 * 3x6, we can think of it as two 3x3 multiplications sharing the same A.
    # If we use a rank r decomposition for 3x3*3x3, we need to "duplicate" the A factors.
    # Specifically, if we have U, V, W for 3x3*3x3 with rank r, then for 3x3*3x6:
    # U_new has size 9 x r, V_new has size 18 x r, W_new has size 18 x r.
    # We can set U_new[i*3+j, t] = U[i*3+j, t] for the first 9 columns (for B1)
    # and U_new[i*3+j, t+r] = U[i*3+j, t] for the next 9 columns (for B2).
    # But this gives rank 2r, which is not optimal.
    #
    # A better approach: use the fact that the two blocks share the same A.
    # We can interleave the factors.
    #
    # Let's try a direct search for a low-rank decomposition.
    # We start with rank 27 (same as 3x3*3x3) and see if we can find a solution.
    #
    # Actually, the minimal rank for 3x3*3x6 is likely 27 or slightly more.
    # Let's try to find a rank 27 decomposition.
    
    n, m, p = DIMENSIONS
    target_rank = 27  # Start with the rank for 3x3*3x3
    
    # The tensor T is a 9 x 18 x 18 tensor.
    # We want to find U (9 x r), V (18 x r), W (18 x r) such that
    # T[i, j, k] = sum_t U[i, t] * V[j, t] * W[k, t]
    
    # For a 3x3*3x3 multiplication, we have:
    # T3x3[i*3+j, j*3+k, k*3+i] = 1
    # This can be decomposed with rank 27 using the schoolbook method.
    #
    # For 3x3*3x6, we have:
    # T[i*3+j, j*3+k, k*3+i] = 1 for k in [0,6)
    # But wait, the output index is k*n+i where k is in [0,6) and n=3.
    # So the output index ranges from 0 to 17.
    #
    # Let's re-examine the tensor definition.
    # T[i*m+j, j*p+k, k*n+i] = 1
    # For n=3, m=3, p=6:
    # i in [0,3), j in [0,3), k in [0,6)
    # First index: i*3+j in [0,9)
    # Second index: j*6+k in [0,18)
    # Third index: k*3+i in [0,18)
    #
    # This is a 9 x 18 x 18 tensor.
    #
    # Notice that for a fixed i and j, the second index j*6+k runs over 6 consecutive values
    # (from j*6 to j*6+5), and the third index k*3+i also runs over 6 values but with a different
    # pattern.
    #
    # Let's try to construct a rank 27 decomposition.
    # We can use the rank 27 decomposition for 3x3*3x3 and adapt it.
    
    # For 3x3*3x3, the schoolbook decomposition is:
    # U[i*3+j, t] = 1 if t = i*3+j
    # V[j*3+k, t] = 1 if t = i*3+j (but this doesn't work directly)
    
    # Actually, the schoolbook decomposition for n x m * m x p is:
    # For each (i, j, k), we have a term:
    # U[i*m+j, t] = 1
    # V[j*p+k, t] = 1
    # W[k*n+i, t] = 1
    # where t = i*m*p + j*p + k
    
    # For 3x3*3x3, this gives rank 27.
    # For 3x3*3x6, this gives rank 54.
    
    # To get a lower rank, we need to exploit the structure.
    #
    # Let's try a different approach: use the known rank 23 decomposition for 3x3*3x3
    # and adapt it for 3x3*3x6.
    #
    # The rank 23 decomposition for 3x3*3x3 is non-trivial. Let's use a known one.
    #
    # Actually, let's try a simpler approach: use a rank 27 decomposition by
    # treating the 3x6 matrix as two 3x3 matrices and using a shared decomposition.
    
    # For now, let's return the schoolbook decomposition of rank 54.
    # This is guaranteed to be correct.
    
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
# EVOLVE-BLOCK-END
