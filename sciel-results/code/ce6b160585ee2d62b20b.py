"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # The multiplication tensor T_{n,m,p} has entries T[i,j,k] = 1 if we're computing (A*B)[i,k] using A[i,j] and B[j,k]
    # We seek a rank-r decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,j,k}
    # where delta_{i,j,k} = 1 if (i,j,k) corresponds to a valid multiplication term, else 0.
    
    # Schoolbook rank is n*m*p = 30
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    
    # For small matrices, we can try ranks from the lower bound up to schoolbook
    # and check if a valid decomposition exists
    
    # Since the problem is small, we can use a systematic search approach
    # For now, let's try a few candidate ranks and see if we can find a valid decomposition
    
    # Strategy: Try ranks from 15 to 30, and for each rank, attempt to find a valid decomposition
    # using a combination of analytical construction and numerical verification
    
    # For 2x3 * 3x5, the minimal rank is known to be 18 (from literature on matrix multiplication ranks)
    # But let's verify this by trying to construct a rank-18 decomposition
    
    # Actually, let's first try to find a better decomposition than schoolbook by
    # exploiting the structure of the problem
    
    # One approach: use the fact that for small matrices, we might be able to find
    # a decomposition with rank less than n*m*p by combining some rank-1 terms
    
    # Let's try a rank-18 decomposition (which is known to be achievable for 2x3 * 3x5)
    # The construction uses the following insight: we can combine some of the multiplication
    # terms by exploiting linear dependencies in the tensor structure
    
    # For 2x3 * 3x5, a rank-18 decomposition can be constructed as follows:
    # We have 2*3 = 6 input variables from A, 3*5 = 15 from B, and 2*5 = 10 outputs
    
    # The key is to find 18 rank-1 tensors that span the multiplication tensor
    
    # Let's construct a rank-18 decomposition using a known pattern for small matrices
    
    # Actually, for now, let's just return the schoolbook decomposition
    # and note that further optimization is needed
    
    # Schoolbook rank is 30
    rank = schoolbook_rank
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
