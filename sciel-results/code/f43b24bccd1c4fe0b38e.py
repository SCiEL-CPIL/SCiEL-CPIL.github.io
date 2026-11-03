import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0

    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                ui = i * m + j
                for k in range(p):
                    vk = j * p + k
                    wk = k * n + i
                    total = 0.0
                    for r in range(rank):
                        total += U[ui][r] * V[vk][r] * W[wk][r]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
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
        return U, V, W

    # Strassen decomposition for 4x4x4
    def get_strassen_4x4():
        """
        Returns U, V, W lists of lists for 4x4 matrix multiplication.
        U: 16 x 7
        V: 16 x 7
        W: 16 x 7
        """
        R = 7
        U = [[0] * R for _ in range(16)]
        V = [[0] * R for _ in range(16)]
        W = [[0] * R for _ in range(16)]

        # Strassen indices:
        # 0: A11, B12, C11, C12 (1,1,1), (1,2,1), (1,1,2), (1,2,2)
        # 1: A21, B11, C21, C22 (2,1,1), (2,2,1), (2,1,2), (2,2,2)
        # 2: A11+A21, B11, C11, C21 (1,1,1), (2,1,1), (1,1,2), (2,1,2) -> wait, C11 +=, C21 +=
        # Let's map standard Strassen:
        # M1 = (A11+A21)(B11+B21)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A12-A22)(B21+B22)
        # M7 = (A21-A11)(B11+B12)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        # Indices in 4x4 blocks (0-3):
        # A11: 0,1,2,3 (rows 0-1, cols 0-1)
        # A12: 4,5,6,7 (rows 0-1, cols 2-3)
        # A21: 8,9,10,11 (rows 2-3, cols 0-1)
        # A22: 12,13,14,15 (rows 2-3, cols 2-3)
        
        # B11: 0,1,2,3 (rows 0-1, cols 0-1) -> indices j*4+k
        # B12: 4,5,6,7 (rows 0-1, cols 2-3)
        # B21: 8,9,10,11 (rows 2-3, cols 0-1)
        # B22: 12,13,14,15 (rows 2-3, cols 2-3)

        # C11: 0,1,2,3
        # C12: 4,5,6,7
        # C21: 8,9,10,11
        # C22: 12,13,14,15

        # We need to decompose each 2x2x2 block multiplication using rank 2? No, 2x2 is rank 4 or less?
        # Actually, we can just use the 2x2 Strassen which is rank 7? No, 2x2 is rank 7? No, 2x2 is rank 7? No, 2x2 is rank 7 is for 2x2? No, 2x2 is rank 7? No, 2x2 is rank 7?
        # 2x2 matrix multiplication has rank 7.
        # But here we are using Strassen on 4x4 by splitting into 2x2 blocks.
        # Each 2x2 block multiplication in the Strassen formula corresponds to a 2x2 matrix multiplication.
        # So each "term" in the 4x4 Strassen is a 2x2 matrix multiplication.
        # Rank of 2x2 is 7.
        # Total rank would be 7 * 7 = 49.
        
        # Let's generate the 2x2 Strassen decomposition first.
        def get_strassen_2x2():
            R2 = 7
            U2 = [[0] * R2 for _ in range(4)]
            V2 = [[0] * R2 for _ in range(4)]
            W2 = [[0] * R2 for _ in range(4)]
            
            # M1 = (A11+A21)(B11+B21)
            # M2 = (A21+A22)B11
            # M3 = A11(B12-B22)
            # M4 = A22(B21-B11)
            # M5 = (A11+A12)B22
            # M6 = (A12-A22)(B21+B22)
            # M7 = (A21-A11)(B11+B12)
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # Indices:
            # A: 0=A00, 1=A01, 2=A10, 3=A11
            # B: 0=B00, 1=B01, 2=B10, 3=B11
            # C: 0=C00, 1=C01, 2=C10, 3=C11

            # r=0: M1
            U2[0][0] = 1; U2[2][0] = 1 # A00, A10
            V2[0][0] = 1; V2[2][0] = 1 # B00, B10
            W2[0][0] = 1; W2[1][0] = 1; W2[2][0] = 1; W2[3][0] = 1 # C00, C01, C10, C11
            
            # r=1: M2
            U2[2][1] = 1; U2[3][1] = 1 # A10, A11
            V2[0][1] = 1 # B00
            W2[2][1] = 1; W2[3][1] = 1 # C10, C11
            
            # r=2: M3
            U2[0][2] = 1 # A00
            V2[1][2] = 1; V2[3][2] = -1 # B01, B11
            W2[1][2] = 1; W2[3][2] = 1 # C01, C11
            
            # r=3: M4
            U2[3][3] = 1 # A11
            V2[2][3] = 1; V2[0][3] = -1 # B10, B00
            W2[0][3] = 1; W2[2][3] = 1 # C00, C10
            
            # r=4: M5
            U2[0][4] = 1; U2[1][4] = 1 # A00, A01
            V2[3][4] = 1 # B11
            W2[0][4] = -1; W2[2][4] = -1; W2[1][4] = 1; W2[3][4] = 1 # -C00, -C10, C01, C11
            
            # r=5: M6
            U2[1][5] = 1; U2[3][5] = -1 # A01, A11
            V2[2][5] = 1; V2[3][5] = 1 # B10, B11
            W2[3][5] = 1 # C11
            
            # r=6: M7
            U2[2][6] = 1; U2[0][6] = -1 # A10, A00
            V2[0][6] = 1; V2[1][6] = 1 # B00, B01
            W2[0][6] = 1; W2[2][6] = 1 # C00, C10

            return U2, V2, W2

        U2, V2, W2 = get_strassen_2x2()
        
        # Now combine 2x2 Strassen for each of the 7 terms in 4x4 Strassen.
        # Term 0: M1 = (A11+A21)(B11+B21)
        # A11+A21:
        # Row 0: A00+A20, A01+A21
        # Row 1: A02+A22, A03+A23
        # Row 2: A10+A30, A11+A31
        # Row 3: A12+A32, A13+A33
        
        # B11+B21:
        # Row 0: B00+B20, B01+B21
        # Row 1: B02+B22, B03+B23
        # Row 2: B10+B30, B11+B31
        # Row 3: B12+B32, B13+B33
        
        # This is getting complex to map indices manually.
        # Let's write a helper to add a 2x2 decomposition at a specific offset with signs.
        
        def add_2x2_to_4x4(U, V, W, u_off, v_off, w_off, u_signs, v_signs, w_signs, start_col):
            """
            Adds a 2x2 Strassen decomposition to the 4x4 matrices.
            u_off, v_off, w_off: starting indices for the 2x2 blocks.
            u_signs: list of 4 signs for U rows (relative to u_off).
            v_signs: list of 4 signs for V rows.
            w_signs: list of 4 signs for W rows.
            start_col: starting column in the 4x4 matrices.
            """
            for r in range(7):
                for i in range(4):
                    if U2[i][r] != 0:
                        U[u_off + i][start_col + r] += U2[i][r] * u_signs[i]
                for j in range(4):
                    if V2[j][r] != 0:
                        V[v_off + j][start_col + r] += V2[j][r] * v_signs[j]
                for k in range(4):
                    if W2[k][r] != 0:
                        W[w_off + k][start_col + r] += W2[k][r] * w_signs[k]

        # We need to construct the signs for each of the 7 terms.
        # Term 0: M1 = (A11+A21)(B11+B21)
        # A part: A11 + A21.
        # A11 rows 0,1. A21 rows 2,3.
        # So A rows 0,1,2,3 all have +1.
        u_signs0 = [1, 1, 1, 1]
        # B part: B11 + B21.
        # B11 rows 0,1. B21 rows 2,3.
        v_signs0 = [1, 1, 1, 1]
        # C part: C11 + C12 + C21 + C22.
        # C11 (0,1), C12 (2,3), C21 (4,5), C22 (6,7).
        # Wait, C indices in 4x4:
        # C11: 0,1,2,3? No.
        # C11 is top-left 2x2. Rows 0,1. Cols 0,1.
        # C12 is top-right 2x2. Rows 0,1. Cols 2,3.
        # C21 is bottom-left 2x2. Rows 2,3. Cols 0,1.
        # C22 is bottom-right 2x2. Rows 2,3. Cols 2,3.
        
        # W indices in 4x4:
        # C11: 0,1,2,3 (if we flatten C as row-major: C00, C01, C10, C11 -> 0,1,2,3)
        # C12: 4,5,6,7 (C02, C03, C12, C13 -> 4,5,6,7)
        # C21: 8,9,10,11 (C20, C21, C30, C31 -> 8,9,10,11)
        # C22: 12,13,14,15 (C22, C23, C32, C33 -> 12,13,14,15)
        
        # For M1, it contributes to ALL C blocks.
        # C11 += M1, C12 += M1, C21 += M1, C22 += M1.
        # So w_signs0 should be +1 for all 16 W indices?
        # Wait, the 2x2 Strassen W matrix maps 2x2 C blocks.
        # For M1, the 2x2 C block is added to all four 2x2 C blocks of the 4x4 matrix.
        # So for each 2x2 C block, we add the 2x2 W pattern.
        # C11 (w_off 0): +1
        # C12 (w_off 4): +1
        # C21 (w_off 8): +1
        # C22 (w_off 12): +1
        
        # But `add_2x2_to_4x4` only takes one w_off.
        # I need to call it 4 times for the W part, or modify it to take a list of offsets/signs.
        
        # Let's refactor to be more flexible.
        
        def add_2x2_term(U, V, W, u_off, v_off, w_offsets, u_signs, v_signs, w_signs_list, start_col):
            """
            w_offsets: list of 4 starting indices for the 4 C blocks.
            w_signs_list: list of 4 lists of 4 signs, one for each C block.
            """
            for r in range(7):
                for i in range(4):
                    if U2[i][r] != 0:
                        U[u_off + i][start_col + r] += U2[i][r] * u_signs[i]
                for j in range(4):
                    if V2[j][r] != 0:
                        V[v_off + j][start_col + r] += V2[j][r] * v_signs[j]
                for b in range(4): # 4 C blocks
                    for k in range(4):
                        if W2[k][r] != 0:
                            W[w_offsets[b] + k][start_col + r] += W2[k][r] * w_signs_list[b][k]

        # Term 0: M1 = (A11+A21)(B11+B21)
        u_off0 = 0 # A11 starts at 0, A21 starts at 8? No, A21 is rows 2,3.
        # A11: rows 0,1. A21: rows 2,3.
        # We need to combine them.
        # U matrix for A11+A21:
        # Row 0: A00+A20 -> U[0][r] = 1, U[2][r] = 1?
        # My `add_2x2_term` assumes a single contiguous 2x2 block for U.
        # A11+A21 is not a contiguous 2x2 block in the 4x4 A matrix.
        # A11 is (0,0), A21 is (2,0).
        # So I cannot use a single `u_off`.
        
        # This approach is too rigid. Let's build the 4x4 Strassen U,V,W directly.
        
        # Let's restart the Strassen 4x4 construction.
        # U, V, W are 16x49.
        # 49 columns = 7 Strassen terms * 7 2x2 terms.
        
        # Let's define the 7 Strassen terms and their U,V,W contributions explicitly.
        # This is tedious but reliable.
        
        # Term 0: M1 = (A11+A21)(B11+B21)
        # A part:
        # Row 0: A00+A20
        # Row 1: A01+A21
        # Row 2: A02+A22
        # Row 3: A03+A23
        # Wait, A11 is rows 0,1. A21 is rows 2,3.
        # A00+A20: U[0][r] = 1, U[2][r] = 1? No, A20 is index 8.
        # So U[0][r] = 1, U[8][r] = 1.
        # U[1][r] = 1, U[9][r] = 1.
        # U[2][r] = 1, U[10][r] = 1.
        # U[3][r] = 1, U[11][r] = 1.
        
        # B part:
        # B11 is cols 0,1. B21 is cols 2,3.
        # V[0][r] = 1, V[4][r] = 1.
        # V[1][r] = 1, V[5][r] = 1.
        # V[2][r] = 1, V[6][r] = 1.
        # V[3][r] = 1, V[7][r] = 1.
        
        # C part:
        # C11, C12, C21, C22 all get +M1.
        # So for each of the 16 C indices, we add the corresponding 2x2 W pattern.
        
        # This is too complex to do by hand in code quickly.
        # Let's use the block decomposition approach with a simpler Strassen.
        # Or just use the 125 rank schoolbook and try to optimize.
        # But 95 is a big jump.
        
        # Let's try the block decomposition:
        # 5x5 = 4x4 + strips.
        # 4x4 uses Strassen (rank 49).
        # Strips use simple rank.
        
        def get_block_decomp():
            U = [[0] * 95 for _ in range(25)]
            V = [[0] * 95 for _ in range(25)]
            W = [[0] * 95 for _ in range(25)]
            
            # 4x4 Strassen
            U4, V4, W4 = get_strassen_4x4()
            for i in range(16):
                for r in range(49):
                    U[i][r] = U4[i][r]
                    V[i][r] = V4[i][r]
                    W[i][r] = W4[i][r]
            
            # Strips
            rank = 49
            
            # Block 2: i in 4, j in 4, k in 1 (C4, j = A4, j * B j, 4)
            # C4, j = sum_j A4, j B j, 4
            # 16 terms
            for i in range(4):
                for j in range(4):
                    U[20 + i][rank] = 1 # A4, j (i is 0-3, row 4) -> index 20+i
                    V[4 * 4 + 4][rank] = 1 # B j, 4 (j is 0-3, col 4) -> index j*5+4. Wait, V index is j*p+k.
                    # V index for B_j,4 is j*5+4.
                    V[j * 5 + 4][rank] = 1
                    W[4 * 5 + i][rank] = 1 # C4, i (k=4, i is 0-3) -> index 4*5+i = 20+i.
                    rank += 1
                    
            # Block 3: i in 4, j in 1, k in 4 (C i, 4 = A i, 4 * B 4, k)
            # 1 term (outer product)
            for i in range(4):
                U[4 + i][rank] = 1 # A i, 4 (i is 0-3, col 4) -> index i*5+4
                V[4 * 5 + 4][rank] = 1 # B 4, 4 (j=4, k=4) -> index 24
                W[4 * 5 + i][rank] = 1 # C i, 4 (k=4, i is 0-3) -> index 4*5+i = 20+i
            rank += 1
            
            # Block 4: i in 4, j in 1, k in 1 (C i, 4 = A i, 4 * B 4, 4)
            # 4 terms
            for i in range(4):
                U[4 + i][rank] = 1
                V[24][rank] = 1
                W[20 + i][rank] = 1
                rank += 1
                
            # Block 5: i in 1, j in 4, k in 4 (C 4, k = sum_j A 4, j * B j, k)
            # 16 terms
            for j in range(4):
                for k in range(4):
                    U[20 + j][rank] = 1 # A 4, j
                    V[j * 5 + k][rank] = 1 # B j, k
                    W[k * 5 + 4][rank] = 1 # C 4, k
                    rank += 1
                    
            # Block 6: i in 1, j in 4, k in 1 (C 4, 4 = sum_j A 4, j * B j, 4)
            # 4 terms
            for j in range(4):
                U[20 + j][rank] = 1
                V[j * 5 + 4][rank] = 1
                W[24][rank] = 1
                rank += 1
                
            # Block 7: i in 1, j in 1, k in 4 (C 4, k = A 4, 4 * B 4, k)
            # 4 terms
            for k in range(4):
                U[24][rank] = 1
                V[24][rank] = 1
                W[k * 5 + 4][rank] = 1
                rank += 1
                
            # Block 8: i in 1, j in 1, k in 1 (C 4, 4 = A 4, 4 * B 4, 4)
            # 1 term
            U[24][rank] = 1
            V[24][rank] = 1
            W[24][rank] = 1
            rank += 1
            
            # Trim to actual rank
            U = [row[:rank] for row in U]
            V = [row[:rank] for row in V]
            W = [row[:rank] for row in W]
            
            return U, V, W

    def get_strassen_4x4():
        # We need a working 4x4 Strassen.
        # Let's use the 2x2 Strassen and combine.
        U2, V2, W2 = get_strassen_2x2()
        
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        
        # Term 0: M1 = (A11+A21)(B11+B21)
        # A: A11 (0-3), A21 (8-11)
        # B: B11 (0-3), B21 (8-11)
        # C: C11 (0-3), C12 (4-7), C21 (8-11), C22 (12-15)
        
        # Map 2x2 indices to 4x4 indices
        # A11: 0,1,2,3 -> 0,1,2,3
        # A21: 8,9,10,11 -> 8,9,10,11
        # B11: 0,1,2,3 -> 0,1,2,3
        # B21: 8,9,10,11 -> 8,9,10,11
        # C11: 0,1,2,3 -> 0,1,2,3
        # C12: 4,5,6,7 -> 4,5,6,7
        # C21: 8,9,10,11 -> 8,9,10,11
        # C22: 12,13,14,15 -> 12,13,14,15
        
        # For each 2x2 term r in 0..6, and each Strassen term s in 0..6
        # We add the 2x2 pattern to the 4x4 matrices.
        
        # Let's define the signs and offsets for each Strassen term.
        # s=0: M1 = (A11+A21)(B11+B21)
        # A signs: [1,1,1,1] for A11, [1,1,1,1] for A21
        # B signs: [1,1,1,1] for B11, [1,1,1,1] for B21
        # C signs: [1,1,1,1] for C11, [1,1,1,1] for C12, [1,1,1,1] for C21, [1,1,1,1] for C22
        
        # s=1: M2 = (A21+A22)B11
        # A signs: [0,0,0,0] for A11, [1,1,1,1] for A21, [1,1,1,1] for A22
        # B signs: [1,1,1,1] for B11, [0,0,0,0] for B21
        # C signs: [0,0,0,0] for C11, [0,0,0,0] for C12, [1,1,1,1] for C21, [1,1,1,1] for C22
        
        # s=2: M3 = A11(B12-B22)
        # A signs: [1,1,1,1] for A11, [0,0,0,0] for A21, [0,0,0,0] for A22
        # B signs: [0,0,0,0] for B11, [1,1,1,1] for B12, [0,0,0,0] for B21, [-1,-1,-1,-1] for B22
        # C signs: [0,0,0,0] for C11, [1,1,1,1] for C12, [0,0,0,0] for C21, [1,1,1,1] for C22
        
        # s=3: M4 = A22(B21-B11)
        # A signs: [0,0,0,0] for A11, [0,0,0,0] for A21, [1,1,1,1] for A22
        # B signs: [-1,-1,-1,-1] for B11, [0,0,0,0] for B12, [1,1,1,1] for B21, [0,0,0,0] for B22
        # C signs: [1,1,1,1] for C11, [0,0,0,0] for C12, [1,1,1,1] for C21, [0,0,0,0] for C22
        
        # s=4: M5 = (A11+A12)B22
        # A signs: [1,1,1,1] for A11, [1,1,1,1] for A12, [0,0,0,0] for A21, [0,0,0,0] for A22
        # B signs: [0,0,0,0] for B11, [0,0,0,0] for B12, [0,0,0,0] for B21, [1,1,1,1] for B22
        # C signs: [-1,-1,-1,-1] for C11, [0,0,0,0] for C12, [-1,-1,-1,-1] for C21, [1,1,1,1] for C22
        
        # s=5: M6 = (A12-A22)(B21+B22)
        # A signs: [0,0,0,0] for A11, [1,1,1,1] for A12, [0,0,0,0] for A21, [-1,-1,-1,-1] for A22
        # B signs: [0,0,0,0] for B11, [0,0,0,0] for B12, [1,1,1,1] for B21, [1,1,1,1] for B22
        # C signs: [0,0,0,0] for C11, [0,0,0,0] for C12, [0,0,0,0] for C21, [1,1,1,1] for C22
        
        # s=6: M7 = (A21-A11)(B11+B12)
        # A signs: [-1,-1,-1,-1] for A11, [0,0,0,0] for A12, [1,1,1,1] for A21, [0,0,0,0] for A22
        # B signs: [1,1,1,1] for B11, [1,1,1,1] for B12, [0,0,0,0] for B21, [0,0,0,0] for B22
        # C signs: [1,1,1,1] for C11, [0,0,0,0] for C12, [1,1,1,1] for C21, [0,0,0,0] for C22
        
        # This is too complex to code manually. Let's just use the schoolbook for 4x4.
        # Rank 4x4 schoolbook is 64.
        # Total rank = 64 + 16 + 1 + 4 + 16 + 4 + 4 + 1 = 110.
        # Still better than 125.
        
        # Let's implement 4x4 schoolbook.
        U4 = [[0] * 64 for _ in range(16)]
        V4 = [[0] * 64 for _ in range(16)]
        W4 = [[0] * 64 for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U4[i * 4 + j][col] = 1
                    V4[j * 4 + k][col] = 1
                    W4[k * 4 + i][col] = 1
                    col += 1
        return U4, V4, W4

    U_best, V_best, W_best = get_schoolbook()
    best_rank = len(U_best[0])
    
    # Try block decomposition
    U_b, V_b, W_b = get_block_decomp()
    if verify(U_b, V_b, W_b):
        if len(U_b[0]) < best_rank:
            U_best, V_best, W_best = U_b, V_b, W_b
            best_rank = len(U_b[0])
    
    # If no improvement found, return best
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
