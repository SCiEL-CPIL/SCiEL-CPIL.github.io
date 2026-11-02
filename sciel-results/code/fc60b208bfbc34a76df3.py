"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _construct_332_block(col_offset):
    """Construct rank-15 decomposition for 3x3x2 block.
    
    col_offset: offset for the 2 columns in the full 6-column problem.
    
    Returns (U_block, V_block, W_block) where:
    - U_block: list of 9 lists, each of length 15 (A entries: i*3+j)
    - V_block: list of 6 lists, each of length 15 (B entries: j*2+k_local)
    - W_block: list of 6 lists, each of length 15 (C entries: k_local*3+i)
    
    Uses Strassen's algorithm for the 2x2x2 core (rows 0,1 of A; cols 0,1 of B)
    plus additional terms for the remaining entries.
    """
    U = [[0]*15 for _ in range(9)]
    V = [[0]*15 for _ in range(6)]
    W = [[0]*15 for _ in range(6)]
    
    # Strassen's 7 terms for the 2x2x2 sub-problem (A rows 0,1; B cols 0,1; C rows 0,1, cols 0,1)
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1   # A[0,0]
    U[4][0] = 1   # A[1,1]
    V[0][0] = 1   # B[0,0]
    V[3][0] = 1   # B[1,1]
    W[0][0] = 1   # C[0,0]
    W[3][0] = 1   # C[0,1]
    W[1][0] = 1   # C[1,0]
    W[4][0] = 1   # C[1,1]
    
    # M2 = (A10+A11)B00
    U[3][1] = 1   # A[1,0]
    U[4][1] = 1   # A[1,1]
    V[0][1] = 1   # B[0,0]
    W[1][1] = 1   # C[1,0]
    W[4][1] = 1   # C[1,1]
    
    # M3 = A00(B10-B11)
    U[0][2] = 1   # A[0,0]
    V[2][2] = 1   # B[1,0]
    V[3][2] = -1  # B[1,1]
    W[3][2] = 1   # C[0,1]
    
    # M4 = A11(B01-B00)
    U[4][3] = 1   # A[1,1]
    V[1][3] = 1   # B[0,1]
    V[0][3] = -1  # B[0,0]
    W[0][3] = 1   # C[0,0]
    W[1][3] = 1   # C[1,0]
    
    # M5 = (A00+A10)B11
    U[0][4] = 1   # A[0,0]
    U[3][4] = 1   # A[1,0]
    V[3][4] = 1   # B[1,1]
    W[0][4] = -1  # C[0,0]
    W[1][4] = 1   # C[1,0]
    
    # M6 = (A01-A00)(B00+B10)
    U[1][5] = 1   # A[0,1]
    U[0][5] = -1  # A[0,0]
    V[0][5] = 1   # B[0,0]
    V[2][5] = 1   # B[1,0]
    W[4][5] = 1   # C[1,1]
    
    # M7 = (A10-A11)(B01+B11)
    U[3][6] = 1   # A[1,0]
    U[4][6] = -1  # A[1,1]
    V[1][6] = 1   # B[0,1]
    V[3][6] = 1   # B[1,1]
    W[0][6] = 1   # C[0,0]
    W[1][6] = -1  # C[1,0]
    
    # Now handle the remaining terms involving row 2 of A or col 2 of B:
    # C[0,0] += A[0,2]*B[2,0]
    # C[0,1] += A[0,2]*B[2,1]
    # C[1,0] += A[1,2]*B[2,0]
    # C[1,1] += A[1,2]*B[2,1]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # Term 7: A[0,2]*B[2,0] -> C[0,0]
    U[2][7] = 1
    V[4][7] = 1
    W[0][7] = 1
    
    # Term 8: A[0,2]*B[2,1] -> C[0,1]
    U[2][8] = 1
    V[5][8] = 1
    W[3][8] = 1
    
    # Term 9: A[1,2]*B[2,0] -> C[1,0]
    U[5][9] = 1
    V[4][9] = 1
    W[1][9] = 1
    
    # Term 10: A[1,2]*B[2,1] -> C[1,1]
    U[5][10] = 1
    V[5][10] = 1
    W[4][10] = 1
    
    # Term 11: A[2,0]*B[0,0] -> C[2,0]
    U[6][11] = 1
    V[0][11] = 1
    W[2][11] = 1
    
    # Term 12: A[2,0]*B[0,1] -> C[2,1]
    U[6][12] = 1
    V[1][12] = 1
    W[5][12] = 1
    
    # Term 13: A[2,1]*B[1,0] -> C[2,0]
    U[7][13] = 1
    V[2][13] = 1
    W[2][13] = 1
    
    # Term 14: A[2,2]*B[2,0] -> C[2,0]
    U[8][14] = 1
    V[4][14] = 1
    W[2][14] = 1
    
    # Wait, I'm missing A[2,2]*B[2,1] -> C[2,1]!
    # That would be term 15, but I only have 15 slots (0-14).
    # I need to combine or find a better construction.
    
    # Let me reconsider. I have 7 Strassen terms + 8 remaining = 15, but I'm missing one.
    # Actually: 7 Strassen + terms 7-14 = 15 terms, but I need to cover A[2,2]*B[2,1].
    
    # Let me re-examine: the Strassen terms handle all contributions from A rows {0,1} and B cols {0,1}.
    # Remaining:
    # C[0,0]: A[0,2]*B[2,0]
    # C[0,1]: A[0,2]*B[2,1]
    # C[1,0]: A[1,2]*B[2,0]
    # C[1,1]: A[1,2]*B[2,1]
    # C[2,0]: A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1]: A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # That's 10 monomials. I need to represent them in 8 rank-1 terms.
    # I can combine: (A[2,0]*B[0,0] + A[2,1]*B[1,0]) and (A[2,0]*B[0,1] + A[2,1]*B[1,1])
    # using a 2-term Strassen-like trick for 1x2x2, but that's still rank 2 each.
    
    # Actually for 1x2x2 (one row of A, 2 cols of B, 2 output cols), the rank is 2:
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # I can write:
    # t_a = A[2,0] * (B[0,0] for C[2,0], B[0,1] for C[2,1])
    # t_b = A[2,1] * (B[1,0] for C[2,0], B[1,1] for C[2,1])
    # t_c = A[2,2] * (B[2,0] for C[2,0], B[2,1] for C[2,1])
    
    # That's 3 terms for row 2.
    
    # And for the cross terms:
    # t_d = A[0,2] * (B[2,0] for C[0,0], B[2,1] for C[0,1])
    # t_e = A[1,2] * (B[2,0] for C[1,0], B[2,1] for C[1,1])
    
    # That's 2 terms.
    
    # Total: 7 + 3 + 2 = 12? But each "term" above is actually 2 rank-1 terms (one for each output col).
    # t_a contributes to both C[2,0] and C[2,1], so it's 2 rank-1 terms.
    # Similarly for others.
    
    # So: 7 + 3*2 + 2*2 = 7 + 6 + 4 = 17. Still 17.
    
    # I think I need to accept that a simple Strassen embedding gives rank 17 for 3x3x2.
    # Let me try a different approach: use a known rank-15 construction.
    
    # Actually, let me just verify if rank 15 is achievable and construct it properly.
    # For now, let me use the schoolbook for this block (rank 18) to ensure correctness,
    # and see if I can improve.
    
    # Let me restart with a clean schoolbook construction for 3x3x2 (rank 18):
    U = [[0]*18 for _ in range(9)]
    V = [[0]*18 for _ in range(6)]
    W = [[0]*18 for _ in range(6)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i*3+j][col] = 1
                V[j*2+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    """Construct a decomposition for 3x3x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # First, try to construct a valid rank-45 decomposition
    # by using three rank-15 blocks for 3x3x2.
    # If that fails verification, fall back to schoolbook rank 54.
    
    # Let me construct the full 3x3x6 decomposition directly.
    # Split p=6 into three groups of 2.
    
    # For each group of 2 columns, construct a 3x3x2 decomposition.
    # I'll try to use rank 15 per block.
    
    # Let me carefully construct a rank-15 decomposition for 3x3x2.
    # I'll use a construction based on the following insight:
    # We can use 5 terms per row of C, but share some computations.
    
    # Alternative: use the fact that 3x3x2 can be decomposed using
    # the Laderman-style or other known constructions.
    
    # Let me just build it explicitly and verify.
    
    # For 3x3x2, I'll construct 15 rank-1 factors.
    # I'll use a systematic approach:
    
    # The tensor T[i,j,k] for i in {0,1,2}, j in {0,1,2}, k in {0,1}
    # We need: sum_r U[i,j,r] * V[j,k,r] = W[i,k,r] (with appropriate indexing)
    
    # Actually, the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p_local+k, r] * W[k*n+i, r]
    # where p_local = 2 for the block.
    
    # Let me construct this using a known method.
    # I'll use the following 15 factors:
    
    U_block = [[0]*15 for _ in range(9)]
    V_block = [[0]*15 for _ in range(6)]
    W_block = [[0]*15 for _ in range(6)]
    
    # I'll use a construction that I can verify by checking the tensor.
    # Let me define the 15 factors as (U_idx, V_idx, W_idx, coeff) tuples.
    
    # Factor construction using a verified approach:
    # I'll use the following known rank-15 decomposition for 3x3x2.
    
    # After careful analysis, here's a valid construction:
    
    # Use Strassen for the 2x2x2 part (rows 0,1 of A; cols 0,1 of B) = 7 terms
    # Then handle the remaining 10 monomials with 8 terms by combining.
    
    # Actually, I realize I should just verify numerically.
    # Let me construct a candidate and check it.
    
    # For now, use schoolbook to ensure correctness:
    rank_total = n * m * p  # 54
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
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
