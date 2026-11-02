"""Optimized tensor decomposition for 3x5x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll build the decomposition in parts
    # Part 1: i in {0,1,2}, j in {0,1,2}, k in {0,1,2} -> 3x3x3, rank 23
    # Part 2: i in {0,1}, j in {0,1}, k in {3,4} -> 2x2x2, rank 7
    # Part 3: i in {0,1}, j in {0,1}, k in {5,6} -> 2x2x2, rank 7
    # Part 4: i in {0,1}, j=2, k in {3,4,5,6} -> 2x1x4, rank 8
    # Part 5: i=2, j in {0,1,2}, k in {3,4,5,6} -> 1x3x4, rank 12
    # Part 6: i in {0,1,2}, j in {3,4}, k in {0,1,2,3,4,5,6} -> 3x2x7, rank 39
    
    # Total: 23 + 7 + 7 + 8 + 12 + 39 = 96
    # But we need to be careful about overlaps. Let me restructure.
    
    # Actually, let me use a cleaner partition:
    # Block A: i in 0..2, j in 0..2, k in 0..2 (3x3x3)
    # Block B: i in 0..2, j in 0..2, k in 3..6 (3x3x4)
    # Block C: i in 0..2, j in 3..4, k in 0..6 (3x2x7)
    
    # For 3x3x4: split k into {3,4} and {5,6}
    #   i in 0..2, j in 0..2, k in {3,4}: 3x3x2
    #   i in 0..2, j in 0..2, k in {5,6}: 3x3x2
    # M(3,3,2) = 18 each (or better?)
    # Actually M(3,3,2) <= 18 via known results
    
    # Let me use a direct construction approach with the known bounds.
    # I'll implement the decomposition using bilinear form coefficients.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 for the (i,k)-th entry of A*B
    # We need U, V, W such that for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for other combinations it's 0.
    
    # Let me build this using the block structure.
    # I'll use the following strategy:
    # - For the 3x3x3 block, use Laderman's decomposition
    # - For 2x2x2 blocks, use Strassen
    # - For 1xm*p or similar, use schoolbook
    # - For 3x2x7, decompose further
    
    # Laderman's 23-rank decomposition for 3x3x3:
    # The 23 terms (U, V, W coefficients for each term)
    laderman = [
        # (U_coeff for row, V_coeff for col, W_coeff for output)
        # Each term is a triple of matrices that multiply to give a rank-1 tensor
        # For 3x3x3, index (i,j,k) in 0..2
        # U[i,j], V[j,k], W[k,i]
    ]
    
    # Let me just implement a known correct decomposition.
    # I'll use the approach of building it term by term.
    
    # Start with schoolbook (rank 105) and see if I can reduce.
    # For now, let me return the schoolbook as a baseline that's correct.
    
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
