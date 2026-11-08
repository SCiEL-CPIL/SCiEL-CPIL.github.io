"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-minimal exact tensor decomposition using rank-1 updates
    # We are looking for rank r such that the multiplication tensor can be written as sum of r rank-1 tensors
    # The schoolbook rank is n*m*p = 30
    # Lower bounds: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # Known results: for 2x3 * 3x5, we can try to find a lower rank
    
    # Let's try to construct a decomposition using a systematic approach
    # We'll use the fact that matrix multiplication tensors have structure
    
    # For now, let's try to find a better decomposition than schoolbook
    # by exploiting the structure of the problem
    
    # One approach: use the fact that we can combine some of the rank-1 terms
    # if the matrices have special structure
    
    # Let's try a different approach: use bilinear algorithms
    # The key insight is that for small matrices, we might be able to find
    # a decomposition with rank less than n*m*p
    
    # For 2x3 * 3x5, let's try to find a rank-15 or lower decomposition
    
    # The multiplication tensor T has entries T[i,j,k,l] where:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,k,j] = 1 if we're computing (A*B)[i,k] using A[i,j] and B[j,k]
    
    # In the factorization:
    # U has shape (n*m, r) = (6, r)
    # V has shape (m*p, r) = (15, r) 
    # W has shape (p*n, r) = (10, r)
    # And we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # And for all other combinations, the sum should be 0
    
    # Let's try to find a rank-15 decomposition (the theoretical lower bound)
    
    # Actually, let's just return the schoolbook decomposition for now,
    # but structure it to allow for future improvements
    
    # Schoolbook rank is 30
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
