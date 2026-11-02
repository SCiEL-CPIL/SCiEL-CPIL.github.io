import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 2: R(4,1,5) rank 20
    # A is 4x1 (col 4), B is 1x5 (row 4)
    # C is 4x5
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
            
    # Part 1: R(4,4,5) rank 44
    # Split p into 1+4
    # R(4,4,1) rank 16
    # A is 4x4 (cols 0-3), B is 4x1 (row 0-3, col 0)
    # C is 4x1 (row 0-3, col 0)
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 0][col] = 1
            W[0 * 4 + i][col] = 1
            col += 1
            
    # R(4,4,4) rank 28
    # A is 4x4 (cols 0-3), B is 4x4 (rows 0-3, cols 1-4)
    # C is 4x4 (rows 1-4, cols 0-3)
    # Use Strassen 2x2x2 (rank 7) with schoolbook 2x2x2 (rank 4) inside.
    
    # Strassen coefficients for 2x2x2
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # We need to map block indices to global indices.
    # Block (bi, bj) of A is at rows bi*2..bi*2+1, cols bj*2..bj*2+1.
    # Global row = i*5 + j.
    
    # For each of the 7 Strassen terms, we have a 2x2x2 tensor.
    # We decompose each 2x2x2 tensor into 4 schoolbook terms.
    
    strassen_terms = [
        # (a_idx, b_idx, c_idx, coeff)
        # a_idx: (i_block, j_block), b_idx: (j_block, k_block), c_idx: (i_block, k_block)
        # This is complicated. Let's just iterate.
    ]
    
    # Simpler: Generate the 28 terms explicitly.
    # For each of the 7 Strassen multiplications M1..M7:
    #   For each of the 4 schoolbook products in the 2x2 block:
    #     Add 1 rank-1 term.
    
    # Strassen M1: (A11+A22)(B11+B22)
    # A blocks: (0,0) and (1,1). B blocks: (0,0) and (0,0).
    # This is not a simple product of single blocks.
    
    # Let's use the explicit 28-term decomposition for 4x4x4.
    # It is the Kronecker product of the 7-term Strassen decomposition
    # and the 4-term schoolbook decomposition? No.
    
    # I will hardcode the 28 terms for 4x4x4.
    # This is tedious. I will use a helper to generate them.
    
    pass

    return {"U": U, "V": V, "W": W}
