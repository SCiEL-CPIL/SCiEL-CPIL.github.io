"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

# The deadline for the evaluator is 150s wall-clock.
DEADLINE = time.time() + 140

# Strassen decomposition for 2x2 matrices
# U, V, W indices:
# A is 2x2: indices 0,1,2,3 (row major: a00, a01, a10, a11)
# B is 2x2: indices 0,1,2,3
# C is 2x2: indices 0,1,2,3
#
# Standard Strassen:
# M1 = (a00+a11)(b00+b11)
# M2 = (a10+a11)b00
# M3 = a00(b10-b11)
# M4 = a11(b01-b00)  # Note: standard formulas vary slightly, let's use a consistent set
# M5 = (a00+a01)b11
# M6 = (a10-a00)(b00+b01)
# M7 = (a01-a11)(b10+b11)
#
# c00 = M1 + M4 - M5 + M7
# c01 = M3 + M5
# c10 = M2 + M4
# c11 = M1 - M2 + M3 + M6
#
# Let's verify M4: Usually M4 = a11(b01 - b00) is correct for c10 = M2 + M4?
# c10 = a10 b00 + a11 b10.
# M2 = (a10+a11)b00 = a10 b00 + a11 b00.
# M4 = a11(b10 - b00) = a11 b10 - a11 b00.
# M2 + M4 = a10 b00 + a11 b10. Correct.
#
# Let's check c00: a00 b00 + a01 b10.
# M1 = (a00+a11)(b00+b11) = a00 b00 + a00 b11 + a11 b00 + a11 b11.
# M7 = (a01-a11)(b10+b11) = a01 b10 + a01 b11 - a11 b10 - a11 b11.
# M5 = (a00+a01)b11 = a00 b11 + a01 b11.
# M4 = a11(b10 - b00) = a11 b10 - a11 b00.
# M1 + M4 - M5 + M7
# = (a00 b00 + a00 b11 + a11 b00 + a11 b11) + (a11 b10 - a11 b00) - (a00 b11 + a01 b11) + (a01 b10 + a01 b11 - a11 b10 - a11 b11)
# = a00 b00 + (a00 b11 - a00 b11) + (a11 b00 - a11 b00) + (a11 b11 - a11 b11) + (a11 b10 - a11 b10) + (-a01 b11 + a01 b11) + a01 b10
# = a00 b00 + a01 b10. Correct.

# Let's define the Strassen decomposition for 2x2
# U shape (4, 7), V shape (4, 7), W shape (4, 7)
# A indices: 0,1,2,3
# B indices: 0,1,2,3
# C indices: 0,1,2,3

def get_strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    # U: a00=1, a11=1
    U[0][0] = 1; U[3][0] = 1
    # V: b00=1, b11=1
    V[0][0] = 1; V[3][0] = 1
    # W: c00=1, c11=1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    # U: a10=1, a11=1
    U[2][1] = 1; U[3][1] = 1
    # V: b00=1
    V[0][1] = 1
    # W: c10=1
    W[2][1] = 1
    
    # M3 = a00(b10-b11)
    # U: a00=1
    U[0][2] = 1
    # V: b10=1, b11=-1
    V[2][2] = 1; V[3][2] = -1
    # W: c01=1, c11=1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a11(b10-b00)  # Wait, earlier I said a11(b01-b00) was wrong, let's re-verify M4.
    # c10 = a10 b00 + a11 b10.
    # M2 = (a10+a11)b00.
    # M2 + M4 = a10 b00 + a11 b00 + M4.
    # We want a10 b00 + a11 b10.
    # So M4 must be a11 b10 - a11 b00 = a11(b10 - b00).
    # Yes, M4 = a11(b10 - b00).
    # U: a11=1
    U[3][3] = 1
    # V: b10=1, b00=-1
    V[2][3] = 1; V[0][3] = -1
    # W: c10=1
    W[2][3] = 1
    
    # M5 = (a00+a01)b11
    # U: a00=1, a01=1
    U[0][4] = 1; U[1][4] = 1
    # V: b11=1
    V[3][4] = 1
    # W: c00=-1, c01=1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    # U: a10=1, a00=-1
    U[2][5] = 1; U[0][5] = -1
    # V: b00=1, b01=1
    V[0][5] = 1; V[1][5] = 1
    # W: c11=1
    W[3][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    # U: a01=1, a11=-1
    U[1][6] = 1; U[3][6] = -1
    # V: b10=1, b11=1
    V[2][6] = 1; V[3][6] = 1
    # W: c00=1
    W[0][6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # We split A (2x4) into two 2x2 blocks: A0 = A[:, 0:2], A1 = A[:, 2:4]
    # We split B (4x4) into four 2x2 blocks:
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # C (2x4) = A0 B00 + A1 B10 (for cols 0:2)
    #          + A0 B01 + A1 B11 (for cols 2:4)
    
    # Total rank = 4 blocks * 7 per Strassen = 28.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    sU, sV, sW = get_strassen_2x2()
    
    # Block 1: A0 * B00 -> C[:, 0:2]
    # A0 indices in U: 0,1,2,3 (rows 0,1 of A)
    # B00 indices in V: 0,1,2,3 (rows 0,1 of B, cols 0,1)
    # C indices in W: 0,1 (rows 0,1 of C, cols 0,1) -> W indices: k*n + i.
    # W index for C[i, j] is j*n + i.
    # C[:, 0:2] means j=0,1.
    # C[0,0] -> W[0]
    # C[0,1] -> W[2]
    # C[1,0] -> W[1]
    # C[1,1] -> W[3]
    # Wait, W shape is (p*n, rank) = (8, 28).
    # W[k*n + i, r].
    # k is column index of C (0 to 3). i is row index of C (0 to 1).
    # So W[k*2 + i].
    # C[0,0] -> k=0, i=0 -> W[0]
    # C[0,1] -> k=1, i=0 -> W[2]
    # C[1,0] -> k=0, i=1 -> W[1]
    # C[1,1] -> k=1, i=1 -> W[3]
    
    for c in range(7):
        base_col = 0 + c
        # U
        for i in range(4):
            if sU[i][c] != 0:
                U[i][base_col] += sU[i][c]
        # V
        for j in range(4):
            if sV[j][c] != 0:
                V[j][base_col] += sV[j][c]
        # W
        # sW indices: 0->C[0,0], 1->C[0,1], 2->C[1,0], 3->C[1,1]
        # W index for C[i, k]: k*n + i
        # sW[0] is C[0,0] -> k=0, i=0 -> W[0]
        # sW[1] is C[0,1] -> k=1, i=0 -> W[2]
        # sW[2] is C[1,0] -> k=0, i=1 -> W[1]
        # sW[3] is C[1,1] -> k=1, i=1 -> W[3]
        for j in range(4):
            if sW[j][c] != 0:
                w_idx = j * n  # Wait, sW index j corresponds to C row j? No.
                # sW is 4x7. sW[i, c] is the coefficient for C[i, ?]
                # In get_strassen_2x2, W is 4x7. W[i, c] is the coefficient for C[i, j]?
                # No, in the 2x2 case, C is 2x2. The indices of C are 0,1,2,3.
                # My W in get_strassen_2x2 has shape (4, 7).
                # W[0] corresponds to C[0,0]
                # W[1] corresponds to C[0,1]
                # W[2] corresponds to C[1,0]
                # W[3] corresponds to C[1,1]
                # This is row-major order for C.
                # So index `j` in sW corresponds to C_flat[j].
                # C_flat[j] has i = j % 2, k = j // 2.
                # W index in global W is k*n + i.
                i_c = j % 2
                k_c = j // 2
                w_global_idx = k_c * n + i_c
                W[w_global_idx][base_col] += sW[j][c]

    # Block 2: A0 * B01 -> C[:, 2:4]
    # A0 indices in U: 0,1,2,3
    # B01 indices in V: 4,5,6,7 (rows 0,1 of B, cols 2,3)
    # C indices in W: k=2,3.
    # C[0,2] -> W[2*2+0] = W[4]
    # C[0,3] -> W[2*2+1] = W[5]
    # C[1,2] -> W[3*2+0] = W[6]
    # C[1,3] -> W[3*2+1] = W[7]
    
    for c in range(7):
        base_col = 7 + c
        # U
        for i in range(4):
            if sU[i][c] != 0:
                U[i][base_col] += sU[i][c]
        # V
        for j in range(4):
            if sV[j][c] != 0:
                V[4+j][base_col] += sV[j][c]
        # W
        for j in range(4):
            if sW[j][c] != 0:
                i_c = j % 2
                k_c = j // 2
                # k_c is 0 or 1 in local C. Global k is 2 + k_c.
                w_global_idx = (2 + k_c) * n + i_c
                W[w_global_idx][base_col] += sW[j][c]

    # Block 3: A1 * B10 -> C[:, 0:2]
    # A1 indices in U: 4,5,6,7 (rows 0,1 of A, cols 2,3)
    # B10 indices in V: 8,9,10,11 (rows 2,3 of B, cols 0,1)
    # C indices in W: k=0,1.
    
    for c in range(7):
        base_col = 14 + c
        # U
        for i in range(4):
            if sU[i][c] != 0:
                U[4+i][base_col] += sU[i][c]
        # V
        for j in range(4):
            if sV[j][c] != 0:
                V[8+j][base_col] += sV[j][c]
        # W
        for j in range(4):
            if sW[j][c] != 0:
                i_c = j % 2
                k_c = j // 2
                # Global k is 0 + k_c.
                w_global_idx = k_c * n + i_c
                W[w_global_idx][base_col] += sW[j][c]

    # Block 4: A1 * B11 -> C[:, 2:4]
    # A1 indices in U: 4,5,6,7
    # B11 indices in V: 12,13,14,15 (rows 2,3 of B, cols 2,3)
    # C indices in W: k=2,3.
    
    for c in range(7):
        base_col = 21 + c
        # U
        for i in range(4):
            if sU[i][c] != 0:
                U[4+i][base_col] += sU[i][c]
        # V
        for j in range(4):
            if sV[j][c] != 0:
                V[12+j][base_col] += sV[j][c]
        # W
        for j in range(4):
            if sW[j][c] != 0:
                i_c = j % 2
                k_c = j // 2
                # Global k is 2 + k_c.
                w_global_idx = (2 + k_c) * n + i_c
                W[w_global_idx][base_col] += sW[j][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
