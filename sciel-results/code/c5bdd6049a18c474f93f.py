"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # We use a block Strassen decomposition.
    # Split A (2x4) into two 2x2 blocks: A1 (cols 0-1), A2 (cols 2-3)
    # Split B (4x6) into three 2x2 blocks along the rows? No, B is 4x6.
    # The product C = A B is 2x6.
    # C = [A1 B1 + A2 B2] where B1, B2 are 2x6? No.
    # A = [A1 A2] (2x2 each)
    # B = [B1; B2] (2x6 each)
    # C = A1 B1 + A2 B2
    # This is a sum of two 2x2x6 tensors.
    # R(M_2,2,6) is approx 18? If so, total 36.
    # But we don't have the 18-term decomposition.
    
    # The current program uses 6 2x2x2 Strassen blocks.
    # A1 (2x2) * B11 (2x2) -> C11 (2x2)
    # A1 (2x2) * B12 (2x2) -> C12 (2x2)
    # A1 (2x2) * B13 (2x2) -> C13 (2x2)
    # A2 (2x2) * B21 (2x2) -> C21 (2x2)
    # A2 (2x2) * B22 (2x2) -> C22 (2x2)
    # A2 (2x2) * B23 (2x2) -> C23 (2x2)
    # Total 6 blocks. Rank 6 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        # Adds a 2x2x2 Strassen block
        # A: rows i_start..i_start+1, cols j_start..j_start+1
        # B: rows j_start..j_start+1, cols k_start..k_start+1
        # C: rows k_start..k_start+1, cols i_start..i_start+1
        
        # M1: (A00+A11)(B00+B11)
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        # M2: (A10+A11)B00
        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1
        
        # M3: A00(B01-B11)
        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        # M4: A11(B10-B00)
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        
        # M5: (A00+A01)B11
        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        
        # M6: (A10-A00)(B00+B01)
        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        # M7: (A01-A11)(B10+B11)
        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 0, 2, 7)
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_block(0, 0, 4, 14)
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(0, 2, 0, 21)
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(0, 2, 2, 28)
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6]
    add_strassen_block(0, 2, 4, 35)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
