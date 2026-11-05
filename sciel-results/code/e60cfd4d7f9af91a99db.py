import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_factors():
    # Returns U2, V2, W2 each of shape (4, 7) for 2x2x2 matrix multiplication.
    # Indexing convention:
    #   A is 2x2, flat index i = iS*2 + iL, where iS in {0,1}, iL in {0,1}
    #   B is 2x2, flat index j = jS*2 + jL
    #   C is 2x2, flat index k = kS*2 + kL
    #
    # U2[i, l]: coefficient for A[i] in the l-th Strassen product
    # V2[j, l]: coefficient for B[j] in the l-th Strassen product
    # W2[k, l]: coefficient for C[k] from the l-th Strassen product
    #
    # Strassen products:
    #   M0 = (A00 + A11)(B00 + B11)
    #   M1 = (A10 + A11)B00
    #   M2 = A00(B01 - B11)
    #   M3 = A11(B10 - B00)
    #   M4 = (A00 + A01)B11
    #   M5 = (A10 - A00)(B00 + B01)
    #   M6 = (A01 - A11)(B10 + B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # A flat indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B flat indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C flat indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    # M0 = (A00 + A11)(B00 + B11)
    U2[0, 0] = 1  # A00
    U2[3, 0] = 1  # A11
    V2[0, 0] = 1  # B00
    V2[3, 0] = 1  # B11
    W2[0, 0] = 1  # C00
    W2[3, 0] = 1  # C11

    # M1 = (A10 + A11)B00
    U2[2, 1] = 1  # A10
    U2[3, 1] = 1  # A11
    V2[0, 1] = 1  # B00
    W2[2, 1] = 1  # C10

    # M2 = A00(B01 - B11)
    U2[0, 2] = 1  # A00
    V2[1, 2] = 1  # B01
    V2[3, 2] = -1 # B11
    W2[1, 2] = 1  # C01

    # M3 = A11(B10 - B00)
    U2[3, 3] = 1  # A11
    V2[2, 3] = 1  # B10
    V2[0, 3] = -1 # B00
    W2[0, 3] = 1  # C00
    W2[2, 3] = 1  # C10

    # M4 = (A00 + A01)B11
    U2[0, 4] = 1  # A00
    U2[1, 4] = 1  # A01
    V2[3, 4] = 1  # B11
    W2[0, 4] = -1 # C00
    W2[1, 4] = 1  # C01

    # M5 = (A10 - A00)(B00 + B01)
    U2[2, 5] = 1  # A10
    U2[0, 5] = -1 # A00
    V2[0, 5] = 1  # B00
    V2[1, 5] = 1  # B01
    W2[3, 5] = 1  # C11

    # M6 = (A01 - A11)(B10 + B11)
    U2[1, 6] = 1  # A01
    U2[3, 6] = -1 # A11
    V2[2, 6] = 1  # B10
    V2[3, 6] = 1  # B11
    W2[0, 6] = 1  # C00

    return U2, V2, W2

def _build_4x4x8_factors():
    n, m, p = 4, 4, 8
    
    # Build 2x2x2 Strassen factors
    U2, V2, W2 = _strassen_2x2_factors()
    
    # For 4x4x4, we use a top-level 2x2 block Strassen decomposition.
    # A (4x4) is split into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
    # B (4x4) is split into 2x2 blocks: B00, B01, B10, B11 (each 2x2)
    # C (4x4) = A * B
    #
    # Top-level Strassen: 7 products of 2x2 matrices
    # Each 2x2 product is computed using rank-7 Strassen
    # Total rank = 7 * 7 = 49 for 4x4x4
    #
    # For 4x4x8, B is split into two 4x4 blocks: B_left (cols 0-3), B_right (cols 4-7)
    # C_left = A * B_left (4x4), C_right = A * B_right (4x4)
    # Total rank = 49 + 49 = 98
    #
    # The U factor is the same for both halves (since A is the same)
    # V and W factors are duplicated with appropriate indexing

    # Top-level 2x2x2 Strassen structure for block-level decomposition
    # Block indices: 0=A00/B00, 1=A01/B01, 2=A10/B10, 3=A11/B11
    # Top-level Strassen products (same structure as 2x2):
    #   T0 = (A00 + A11)(B00 + B11)
    #   T1 = (A10 + A11)B00
    #   T2 = A00(B01 - B11)
    #   T3 = A11(B10 - B00)
    #   T4 = (A00 + A01)B11
    #   T5 = (A10 - A00)(B00 + B01)
    #   T6 = (A01 - A11)(B10 + B11)
    #
    # C00 = T0 + T3 - T4 + T6
    # C01 = T2 + T4
    # C10 = T1 + T3
    # C11 = T0 - T1 + T2 + T5

    # Top-level coefficients (same as 2x2 Strassen)
    UT = np.zeros((4, 7), dtype=int)
    VT = np.zeros((4, 7), dtype=int)
    WT = np.zeros((4, 7), dtype=int)
    UT[:, :] = U2[:, :]
    VT[:, :] = V2[:, :]
    WT[:, :] = W2[:, :]

    # Build 4x4x4 factors
    # U4 shape: (16, 49) - A is 4x4, flat index i = row*4 + col
    # V4 shape: (16, 49) - B is 4x4, flat index j = row*4 + col
    # W4 shape: (16, 49) - C is 4x4, flat index k = row*4 + col
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)

    # For each top-level product t (0..6) and each inner Strassen product l (0..6):
    #   rank index r = t * 7 + l
    #   A flat index: aS * 4 + aL, where aS is block row (0 or 1), aL is local row*2+col in 2x2
    #   B flat index: bS * 4 + bL
    #   C flat index: cS * 4 + cL
    #
    # U4[aS*4 + aL, r] += UT[aS, t] * U2[aL, l]
    # V4[bS*4 + bL, r] += VT[bS, t] * V2[bL, l]
    # W4[cS*4 + cL, r] += WT[cS, t] * W2[cL, l]

    for t in range(7):
        for l in range(7):
            r = t * 7 + l
            for aS in range(2):
                for aL in range(4):
                    U4[aS * 4 + aL, r] += UT[aS, t] * U2[aL, l]
            for bS in range(2):
                for bL in range(4):
                    V4[bS * 4 + bL, r] += VT[bS, t] * V2[bL, l]
            for cS in range(2):
                for cL in range(4):
                    W4[cS * 4 + cL, r] += WT[cS, t] * W2[cL, l]

    # Now build 4x4x8 factors
    # U shape: (16, 98)
    # V shape: (32, 98)
    # W shape: (32, 98)
    #
    # U is just [U4, U4] (same A for both halves of B)
    # V is [V4_left, V4_right] where V4_left corresponds to B_left (cols 0-3) and V4_right to B_right (cols 4-7)
    #   V_left has rows 0..15 (B_left flat indices: row*4 + col, col in 0..3)
    #   V_right has rows 16..31 (B_right flat indices: 16 + row*4 + col, col in 0..3)
    # W is [W4_left, W4_right] where:
    #   W_left has rows 0..15 (C_left flat indices: row*4 + col, col in 0..3)
    #   W_right has rows 16..31 (C_right flat indices: 16 + row*4 + col, col in 0..3)

    U = np.zeros((16, 98), dtype=int)
    V = np.zeros((32, 98), dtype=int)
    W = np.zeros((32, 98), dtype=int)

    # U: same for both halves
    U[:, 0:49] = U4
    U[:, 49:98] = U4

    # V: left half (cols 0-3 of B) and right half (cols 4-7 of B)
    V[0:16, 0:49] = V4
    V[16:32, 49:98] = V4

    # W: left half (cols 0-3 of C) and right half (cols 4-7 of C)
    # C flat index: k*4 + i where k in 0..7, i in 0..3
    # For C_left (k in 0..3): flat index = k*4 + i, where k_local = k
    # For C_right (k in 4..7): flat index = k*4 + i, where k_local = k - 4
    # W4 is indexed by cS*4 + cL where cS is block row (0 or 1) and cL is local index
    # For C_left: cS*4 + cL corresponds to k_local = cS*2 + cL//2, i = cL%2
    #   But we need W[k*4 + i, r] where k in 0..3 for left half
    #   k = cS*2 + (cL//2), i = cL%2
    #   So W_left row = k*4 + i = (cS*2 + cL//2)*4 + cL%2
    #   But W4 row = cS*4 + cL
    #   These are different! We need to remap.
    #
    # Actually, W4 is built with C flat index = cS*4 + cL where cS is the top-level block row
    # and cL is the 2x2 local flat index (cL = cL_row*2 + cL_col).
    # C is 4x4, so flat index should be row*4 + col.
    # cS is the top-level block row (0 or 1), so C row = cS*2 + cL_row.
    # C col = cL_col.
    # C flat index = (cS*2 + cL_row)*4 + cL_col = cS*8 + cL_row*4 + cL_col.
    # But we set W4[cS*4 + cL, r] where cL = cL_row*2 + cL_col.
    # So W4 row = cS*4 + cL_row*2 + cL_col.
    # This is NOT the same as C flat index = cS*8 + cL_row*4 + cL_col.
    #
    # I made an error in the construction. Let me redo it properly.
    #
    # The C matrix is 4x4. Its flat index is row*4 + col, where row in 0..3, col in 0..3.
    # In the block decomposition:
    #   C00 (top-left 2x2): rows 0-1, cols 0-1
    #   C01 (top-right 2x2): rows 0-1, cols 2-3
    #   C10 (bottom-left 2x2): rows 2-3, cols 0-1
    #   C11 (bottom-right 2x2): rows 2-3, cols 2-3
    #
    # Top-level block index: cS (0 or 1) is the block row, cT (0 or 1) is the block col.
    # But in our 2x2 Strassen for blocks, we only have one index pair (block row, block col)
    # for A and B. For C, the output block is determined by the input blocks.
    #
    # Actually, in the standard Strassen decomposition for 2x2 blocks:
    #   C00 (block row 0, block col 0) = T0 + T3 - T4 + T6
    #   C01 (block row 0, block col 1) = T2 + T4
    #   C10 (block row 1, block col 0) = T1 + T3
    #   C11 (block row 1, block col 1) = T0 - T1 + T2 + T5
    #
    # So C has 4 blocks: (0,0), (0,1), (1,0), (1,1).
    # Each block is 2x2.
    # C flat index for block (cS, cT) and local position (lr, lc):
    #   row = cS*2 + lr, col = cT*2 + lc
    #   flat = (cS*2 + lr)*4 + (cT*2 + lc) = cS*8 + lr*4 + cT*2 + lc
    #
    # In the inner 2x2 Strassen, W2 is indexed by k = kS*2 + kL where kS is the local row and kL is...
    # Wait, in the 2x2 case, C is 2x2, flat index = row*2 + col.
    # kS = row, kL = col. So k = row*2 + col.
    #
    # For the 4x4 case, the inner 2x2 is within a block.
    # Block (cS, cT) has local 2x2 matrix.
    # Local flat index within the block: lr*2 + lc.
    # Global flat index: (cS*2 + lr)*4 + (cT*2 + lc) = cS*8 + lr*4 + cT*2 + lc.
    #
    # So W4 should be indexed by this global flat index.
    # Let me rebuild W4 properly.

    # Redo 4x4x4 factors with correct indexing
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)

    # A flat index: row*4 + col, where row = aS*2 + ar, col = aT*2 + ac
    #   aS in {0,1} (block row), aT in {0,1} (block col)
    #   ar in {0,1} (local row), ac in {0,1} (local col)
    #   flat = (aS*2 + ar)*4 + (aT*2 + ac) = aS*8 + ar*4 + aT*2 + ac
    #
    # B flat index: row*4 + col, where row = bS*2 + br, col = bT*2 + bc
    #   flat = bS*8 + br*4 + bT*2 + bc
    #
    # C flat index: row*4 + col, where row = cS*2 + cr, col = cT*2 + cc
    #   flat = cS*8 + cr*4 + cT*2 + cc

    # Top-level Strassen:
    #   T0 = (A00 + A11)(B00 + B11) -> contributes to C00 and C11
    #   T1 = (A10 + A11)B00 -> contributes to C10
    #   T2 = A00(B01 - B11) -> contributes to C01
    #   T3 = A11(B10 - B00) -> contributes to C00 and C10
    #   T4 = (A00 + A01)B11 -> contributes to C00 and C01
    #   T5 = (A10 - A00)(B00 + B01) -> contributes to C11
    #   T6 = (A01 - A11)(B10 + B11) -> contributes to C00

    # A block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    # Top-level coefficients
    # UT[a_block, t], VT[b_block, t], WT[c_block, t]
    UT = np.zeros((4, 7), dtype=int)
    VT = np.zeros((4, 7), dtype=int)
    WT = np.zeros((4, 7), dtype=int)

    # T0: A00 + A11, B00 + B11 -> C00, C11
    UT[0, 0] = 1; UT[3, 0] = 1
    VT[0, 0] = 1; VT[3, 0] = 1
    WT[0, 0] = 1; WT[3, 0] = 1

    # T1: A10 + A11, B00 -> C10
    UT[2, 1] = 1; UT[3, 1] = 1
    VT[0, 1] = 1
    WT[2, 1] = 1

    # T2: A00, B01 - B11 -> C01
    UT[0, 2] = 1
    VT[1, 2] = 1; VT[3, 2] = -1
    WT[1, 2] = 1

    # T3: A11, B10 - B00 -> C00, C10
    UT[3, 3] = 1
    VT[2, 3] = 1; VT[0, 3] = -1
    WT[0, 3] = 1; WT[2, 3] = 1

    # T4: A00 + A01, B11 -> C00, C01
    UT[0, 4] = 1; UT[1, 4] = 1
    VT[3, 4] = 1
    WT[0, 4] = -1; WT[1, 4] = 1

    # T5: A10 - A00, B00 + B01 -> C11
    UT[2, 5] = 1; UT[0, 5] = -1
    VT[0, 5] = 1; VT[1, 5] = 1
    WT[3, 5] = 1

    # T6: A01 - A11, B10 + B11 -> C00
    UT[1, 6] = 1; UT[3, 6] = -1
    VT[2, 6] = 1; VT[3, 6] = 1
    WT[0, 6] = 1

    # Inner 2x2 Strassen coefficients (U2, V2, W2)
    # U2[local_A_index, l], V2[local_B_index, l], W2[local_C_index, l]
    # local_A_index = ar*2 + ac (ar in 0..1, ac in 0..1)
    # local_B_index = br*2 + bc
    # local_C_index = cr*2 + cc

    for t in range(7):
        for l in range(7):
            r = t * 7 + l
            # A
            for a_block in range(4):
                coeff_a = UT[a_block, t]
                if coeff_a == 0:
                    continue
                aS = a_block // 2
                aT = a_block % 2
                for ar in range(2):
                    for ac in range(2):
                        local_a = ar * 2 + ac
                        coeff_inner = U2[local_a, l]
                        if coeff_inner == 0:
                            continue
                        a_flat = aS * 8 + ar * 4 + aT * 2 + ac
                        U4[a_flat, r] += coeff_a * coeff_inner
            # B
            for b_block in range(4):
                coeff_b = VT[b_block, t]
                if coeff_b == 0:
                    continue
                bS = b_block // 2
                bT = b_block % 2
                for br in range(2):
                    for bc in range(2):
                        local_b = br * 2 + bc
                        coeff_inner = V2[local_b, l]
                        if coeff_inner == 0:
                            continue
                        b_flat = bS * 8 + br * 4 + bT * 2 + bc
                        V4[b_flat, r] += coeff_b * coeff_inner
            # C
            for c_block in range(4):
                coeff_c = WT[c_block, t]
                if coeff_c == 0:
                    continue
                cS = c_block // 2
                cT = c_block % 2
                for cr in range(2):
                    for cc in range(2):
                        local_c = cr * 2 + cc
                        coeff_inner = W2[local_c, l]
                        if coeff_inner == 0:
                            continue
                        c_flat = cS * 8 + cr * 4 + cT * 2 + cc
                        W4[c_flat, r] += coeff_c * coeff_inner

    # Now build 4x4x8 factors
    # U shape: (16, 98)
    # V shape: (32, 98)
    # W shape: (32, 98)

    U = np.zeros((16, 98), dtype=int)
    V = np.zeros((32, 98), dtype=int)
    W = np.zeros((32, 98), dtype=int)

    # U: same for both halves of B
    U[:, 0:49] = U4
    U[:, 49:98] = U4

    # V: B is 4x8, split into B_left (cols 0-3) and B_right (cols 4-7)
    # B flat index: row*8 + col
    # B_left: row in 0..3, col in 0..3, flat = row*8 + col
    # B_right: row in 0..3, col in 4..7, flat = row*8 + col = row*8 + 4 + (col-4)
    #
    # V4 is indexed by B 4x4 flat index: row*4 + col (row in 0..3, col in 0..3)
    # For B_left: V4[row*4 + col, r] -> V[row*8 + col, r]
    # For B_right: V4[row*4 + col, r] -> V[row*8 + 4 + col, r]

    for row in range(4):
        for col in range(4):
            v4_idx = row * 4 + col
            for r in range(49):
                V[row * 8 + col, r] = V4[v4_idx, r]
                V[row * 8 + 4 + col, 49 + r] = V4[v4_idx, r]

    # W: C is 4x8, flat index = row*8 + col
    # C_left: row in 0..3, col in 0..3, flat = row*8 + col
    # C_right: row in 0..3, col in 4..7, flat = row*8 + 4 + col
    #
    # W4 is indexed by C 4x4 flat index: row*4 + col (row in 0..3, col in 0..3)
    # For C_left: W4[row*4 + col, r] -> W[row*8 + col, r]
    # For C_right: W4[row*4 + col, r] -> W[row*8 + 4 + col, 49 + r]

    for row in range(4):
        for col in range(4):
            w4_idx = row * 4 + col
            for r in range(49):
                W[row * 8 + col, r] = W4[w4_idx, r]
                W[row * 8 + 4 + col, 49 + r] = W4[w4_idx, r]

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_4x4x8_factors()
    # Convert to lists of lists
    U_list = [[int(U[i, r]) for r in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[int(V[i, r]) for r in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[int(W[i, r]) for r in range(W.shape[1])] for i in range(W.shape[0])]
    return {"U": U_list, "V": V_list, "W": W_list}
