"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Split 5x5 into 3x3 + 2x2 blocks:
    # A = [[A11(3x3), A12(3x2)],
    #      [A21(2x3), A22(2x2)]]
    # B = [[B11(3x3), B12(3x2)],
    #      [B21(2x3), B22(2x2)]]
    # C = [[C11(3x3), C12(3x2)],
    #      [C21(2x3), C22(2x2)]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # For 2x2*2x2 products, use Strassen (rank 7)
    # For rectangular products, use schoolbook
    #
    # C11: A11*B11 (3x3 schoolbook, rank 27) + A12*B21 (3x2*2x3 schoolbook, rank 18) = 45
    # C12: A11*B12 (3x3*3x2 schoolbook, rank 18) + A12*B22 (3x2*2x2 schoolbook, rank 12) = 30
    # C21: A21*B11 (2x3*3x3 schoolbook, rank 18) + A22*B21 (2x2*2x3 schoolbook, rank 12) = 30
    # C22: A21*B12 (2x3*3x2 schoolbook, rank 12) + A22*B22 (2x2*2x2 Strassen, rank 7) = 19
    #
    # Total: 45 + 30 + 30 + 19 = 124
    #
    # That's barely better than 125. Let me think more carefully.
    #
    # Actually, I should use Strassen for ALL 2x2 sub-products where possible.
    # But the rectangular terms can't use Strassen directly.
    #
    # Let me try a different split: 2x2 + 3x3
    # Or better: use the fact that we can apply Strassen at multiple levels.
    #
    # Alternative approach: Use the known result that 5x5 can be done with
    # rank at most 78 (Bini et al.). But that's very complex to implement.
    #
    # Let me try a cleaner approach: split into 2x2 blocks.
    # 5 = 2+2+1
    #
    # A = [[A11(2x2), A12(2x2), A13(2x1)],
    #      [A21(2x2), A22(2x2), A23(2x1)],
    #      [A31(1x2), A32(1x2), A33(1x1)]]
    #
    # This is getting complex. Let me just implement the 3+2 split with
    # Strassen for the 2x2 block product.
    #
    # Actually, let me reconsider. The simplest improvement:
    # Use Strassen for the 2x2*2x2 product in C22, saving 4-7=3 rank.
    # That gives 125-3=122. Not great.
    #
    # Better idea: use a 4+1 split.
    # 4x4*4x4 has rank at most 49 (known result).
    # But implementing rank-49 for 4x4 is also complex.
    #
    # Let me just implement the schoolbook decomposition correctly.
    # It's the safest approach and guarantees correctness.
    
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
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
