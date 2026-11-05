import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use block-Strassen decomposition.
    # Partition 6x6 matrices into 3x3 blocks of 2x2.
    # Use Strassen (rank 7) for 2x2x2.
    # Total rank = 3 * 3 * 3 * 7 = 189.
    
    # Strassen matrices for 2x2x2
    # U, V, W for 2x2x2
    # Standard Strassen decomposition
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # We construct the global U, V, W matrices.
    # Global dimensions:
    # U: (6*6, 189) = (36, 189)
    # V: (6*6, 189) = (36, 189)
    # W: (6*6, 189) = (36, 189)
    
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen coefficients
    # Each term is a linear combination of blocks.
    # We iterate over the 3x3 block grid.
    # For each block position (i, j, k) in the 3x3 grid,
    # we have a 2x2x2 multiplication.
    # The block indices are (bi, bj, bk) where bi, bj, bk in 0..2.
    # The local indices within the block are (li, lj, lk) where li, lj, lk in 0..1.
    
    # Strassen decomposition details for 2x2x2:
    # The 7 terms are:
    # T1: (A00+A11)(B00+B11) -> C00 += T1 - T4 - T6 + T7, C01 += T4 + T6, C10 += T2 + T5, C11 += T1 - T2 + T3 - T5
    # Wait, the standard Strassen formulas are:
    # C00 = M1 - M4 - M6 + M7
    # C01 = M4 + M6
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 - M5
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map these to the global indices.
    # Global index for A: i*m + j.
    # Block bi, local li -> i = bi*2 + li.
    # Block bj, local lj -> j = bj*2 + lj.
    # Block bk, local lk -> k = bk*2 + lk.
    
    # For each block (bi, bj, bk), we have 7 terms.
    # Total terms = 3 * 3 * 3 * 7 = 189.
    
    term_index = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                # For this block multiplication A[bi, bj] * B[bj, bk] -> C[bi, bk]
                # We add 7 terms.
                for t in range(7):
                    # Determine the local indices for A, B, C involved in this term.
                    # This is complex to map directly.
                    # Instead, we can construct the U, V, W matrices for the 2x2x2 Strassen decomposition
                    # and then embed them.
                    
                    # Let's pre-compute the Strassen matrices for 2x2x2.
                    # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7.
                    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
                    
                    # We will fill U, V, W for the specific block.
                    pass
                    
    # This manual mapping is error-prone.
    # Better approach: Construct the full tensor and decompose? No, too slow.
    # Use the Kronecker product structure.
    
    # Let's define the Strassen matrices explicitly.
    # U_s (4x7), V_s (4x7), W_s (4x7)
    # Rows of U_s correspond to A00, A01, A10, A11.
    # Rows of V_s correspond to B00, B01, B10, B11.
    # Rows of W_s correspond to C00, C01, C10, C11.
    
    # M1: A00+A11, B00+B11 -> C00, C11
    # M2: A10+A11, B00 -> C10, C11
    # M3: A00, B01-B11 -> C11
    # M4: A11, B10-B00 -> C00, C01
    # M5: A00+A01, B11 -> C10, C11
    # M6: A10-A00, B00+B01 -> C00, C01
    # M7: A01-A11, B10+B11 -> C00
    
    # Let's build U_s, V_s, W_s.
    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]
    
    # Term 1: M1
    U_s[0][0] = 1; U_s[3][0] = 1
    V_s[0][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1
    
    # Term 2: M2
    U_s[2][1] = 1; U_s[3][1] = 1
    V_s[0][1] = 1
    W_s[2][1] = 1; W_s[3][1] = -1
    
    # Term 3: M3
    U_s[0][2] = 1
    V_s[1][2] = 1; V_s[3][2] = -1
    W_s[3][2] = 1
    
    # Term 4: M4
    U_s[3][3] = 1
    V_s[2][3] = 1; V_s[0][3] = -1
    W_s[0][3] = -1; W_s[1][3] = 1
    
    # Term 5: M5
    U_s[0][4] = 1; U_s[1][4] = 1
    V_s[3][4] = 1
    W_s[2][4] = 1; W_s[3][4] = -1
    
    # Term 6: M6
    U_s[2][5] = 1; U_s[0][5] = -1
    V_s[0][5] = 1; V_s[1][5] = 1
    W_s[0][5] = -1; W_s[1][5] = 1
    
    # Term 7: M7
    U_s[1][6] = 1; U_s[3][6] = -1
    V_s[2][6] = 1; V_s[3][6] = 1
    W_s[0][6] = 1
    
    # Now embed into 6x6x6.
    # Global U: 36x189
    # Global V: 36x189
    # Global W: 36x189
    
    U = [[0] * 189 for _ in range(36)]
    V = [[0] * 189 for _ in range(36)]
    W = [[0] * 189 for _ in range(36)]
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for t in range(7):
                    # A block (bi, bj)
                    # Local indices li, lj in 0..1
                    for li in range(2):
                        for lj in range(2):
                            local_a_idx = li * 2 + lj
                            global_a_idx = (bi * 2 + li) * 6 + (bj * 2 + lj)
                            if U_s[local_a_idx][t] != 0:
                                U[global_a_idx][col] = U_s[local_a_idx][t]
                    
                    # B block (bj, bk)
                    for lj in range(2):
                        for lk in range(2):
                            local_b_idx = lj * 2 + lk
                            global_b_idx = (bj * 2 + lj) * 6 + (bk * 2 + lk)
                            if V_s[local_b_idx][t] != 0:
                                V[global_b_idx][col] = V_s[local_b_idx][t]
                    
                    # C block (bi, bk)
                    for li in range(2):
                        for lk in range(2):
                            local_c_idx = li * 2 + lk
                            global_c_idx = (bk * 2 + lk) * 6 + (bi * 2 + li) # Trace convention W[k*n+i, r] -> C[i,k]
                            # Wait, the task says W[k*n+i, r].
                            # C[i,k] corresponds to W[k*6+i].
                            # So global_c_idx = lk * 6 + (bi * 2 + li).
                            if W_s[local_c_idx][t] != 0:
                                W[global_c_idx][col] = W_s[local_c_idx][t]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
