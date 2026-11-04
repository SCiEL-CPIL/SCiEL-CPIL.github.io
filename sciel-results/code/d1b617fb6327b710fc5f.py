import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 4x4x4 tensor into rank 49.
    # Then we use two copies for the 4x4x8 tensor.
    
    # Strassen 2x2x2 rank 7 factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    U2 = [
        [1, 0, 0, 0, 1, 0, 0], # A00
        [0, 0, 1, 0, 0, 1, 0], # A10
        [0, 0, 0, 1, 0, 0, 1], # A11
        [1, 0, 1, 0, 1, 0, 0]  # A01
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0], # B00
        [0, 0, 1, 0, 1, 0, 1], # B10
        [1, 0, 0, 1, 1, 0, 1], # B11
        [0, 1, -1, 0, 0, 1, 0] # B01
    ]
    W2 = [
        [1, 0, 0, 0, 0, 0, 0], # C00 = M1 - M4 + M6 - M7
        [0, 0, 0, 0, 1, 0, 0], # C10 = M5 + M7
        [0, 0, 0, 0, 0, 1, 0], # C01 = M3 + M4
        [0, 1, -1, 0, 0, 0, 0]  # C11 = M1 - M2 + M3 + M5
    ]
    # Note: The standard Strassen output is:
    # C00 = M1 - M4 + M6 - M7
    # C10 = M5 + M7
    # C01 = M3 + M4
    # C11 = M1 - M2 + M3 + M5
    
    # Let's verify the W2 matrix.
    # W2[k*2+i, c] should be the coefficient of M_c in C_{k,i}?
    # C is 2x2. Indices (k,i) where k is row, i is col?
    # Standard matrix mult C = A B. C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Here indices are i (row of A), j (col of A), k (col of B).
    # C_{i,k}.
    # In Strassen, C is 2x2.
    # C00 (i=0,k=0): M1 - M4 + M6 - M7
    # C10 (i=1,k=0): M5 + M7
    # C01 (i=0,k=1): M3 + M4
    # C11 (i=1,k=1): M1 - M2 + M3 + M5
    
    # So W2[k*2+i, c] is the coefficient of M_c in C_{i,k}.
    # W2[0, 0] (k=0,i=0) -> C00 -> 1
    # W2[0, 1] (k=0,i=1) -> C10 -> 0 ? No, C10 is i=1,k=0.
    # Wait, W2 index is k*2+i.
    # k=0, i=0 -> index 0. C00.
    # k=0, i=1 -> index 1. C10? No, i is row, k is col.
    # C_{i,k}.
    # i=0, k=0 -> C00.
    # i=1, k=0 -> C10.
    # i=0, k=1 -> C01.
    # i=1, k=1 -> C11.
    
    # So:
    # Index 0 (k=0,i=0) -> C00 -> [1, 0, 0, -1, 0, 1, -1]
    # Index 1 (k=0,i=1) -> C10 -> [0, 0, 0, 0, 1, 0, 1]
    # Index 2 (k=1,i=0) -> C01 -> [0, 0, 1, 1, 0, 0, 0]
    # Index 3 (k=1,i=1) -> C11 -> [1, -1, 1, 0, 1, 0, 0]
    
    # Let's re-verify Strassen formulas.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C10 = M5 + M7
    # C01 = M3 + M4
    # C11 = M1 - M2 + M3 + M5
    
    # So W2 (4x7):
    # Row 0 (k=0, i=0) -> C00: [1, 0, 0, -1, 0, 1, -1]
    # Row 1 (k=0, i=1) -> C10: [0, 0, 0, 0, 1, 0, 1]
    # Row 2 (k=1, i=0) -> C01: [0, 0, 1, 1, 0, 0, 0]
    # Row 3 (k=1, i=1) -> C11: [1, -1, 1, 0, 1, 0, 0]
    
    # Now build 4x4x4 factors.
    # U4: 16x49.
    # V4: 16x49.
    # W4: 16x49.
    
    # U4[i*4+j, c]
    # V4[j*4+k, c]
    # W4[k*4+i, c]
    
    # c = m_block * 7 + c_sub
    # m_block in 0..6, c_sub in 0..6
    
    # For each m_block, we have a 2x2 block multiplication.
    # The A block is A[m_block_row, m_block_col]?
    # No, Strassen combines blocks.
    # M1 uses A00, A11.
    # M2 uses A10, A11.
    # M3 uses A00.
    # M4 uses A11.
    # M5 uses A00, A01.
    # M6 uses A10, A00.
    # M7 uses A01, A11.
    
    # Let's map A indices.
    # A is 4x4. Blocks are 2x2.
    # A00: rows 0-1, cols 0-1.
    # A01: rows 0-1, cols 2-3.
    # A10: rows 2-3, cols 0-1.
    # A11: rows 2-3, cols 2-3.
    
    # For a given m_block, the A factor is a linear combination of blocks.
    # We can represent the A factor for m_block as a 4x4 matrix $A^{(m)}$.
    # Then $U4[:, m*7+c] = U2_{expanded} \otimes A^{(m)}$?
    # No, $U2$ is for 2x2. We need to expand it to 4x4.
    
    # Actually, it's easier to just iterate.
    # For each m_block (0..6), we have 7 sub-multiplications.
    # For each sub-multiplication c_sub (0..6), we have a rank-1 term.
    # The term is $u_{i,j} v_{j,k} w_{k,i}$.
    # $u_{i,j}$ comes from $U2$ and the block structure.
    
    # Let's define the A-block coefficients for each M.
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    
    # Let A_coeffs[m][block] be the coefficient of that block in M_m.
    # Blocks: 00, 01, 10, 11.
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Similarly for B blocks.
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Now build U4, V4, W4.
    rank49 = 49
    U4 = [[0]*rank49 for _ in range(16)]
    V4 = [[0]*rank49 for _ in range(16)]
    W4 = [[0]*rank49 for _ in range(16)]
    
    for m in range(7):
        for c in range(7):
            col = m*7 + c
            # U4 part
            # U2[c, i2, j2]
            # U2 is 4x7? No, U2 is 4x7 where index is i2*2+j2.
            # U2[i2*2+j2, c]
            for i2 in range(2):
                for j2 in range(2):
                    u2_val = U2[i2*2+j2][c]
                    if u2_val != 0:
                        # This u2_val applies to all elements in the A block?
                        # No, U2 is the factor for the 2x2 multiplication.
                        # The 2x2 multiplication is between A_block and B_block.
                        # A_block is 2x2.
                        # The factor U2[i2,j2,c] multiplies A_block[i2,j2].
                        # So for each (i2,j2), we add u2_val * A_coeffs[m][block] to U4.
                        # Which block?
                        # Wait, A_coeffs[m] gives the linear combination of blocks.
                        # So A_eff[i,j] = sum_block A_coeffs[m][block] * A_block[i,j].
                        # The factor U4[i,j,col] should be the coefficient of A[i,j] in the product.
                        # So U4[i,j,col] = sum_block A_coeffs[m][block] * (1 if (i,j) in block else 0) * U2[i2,j2,c].
                        
                        # i = i2*2 + i_block? No.
                        # Block 00: i in 0,1; j in 0,1. i2=i, j2=j.
                        # Block 01: i in 0,1; j in 2,3. i2=i, j2=j-2.
                        # Block 10: i in 2,3; j in 0,1. i2=i-2, j2=j.
                        # Block 11: i in 2,3; j in 2,3. i2=i-2, j2=j-2.
                        
                        # So for a fixed (i2, j2) from U2, it corresponds to specific blocks.
                        # Actually, U2[i2,j2,c] is the coefficient for A_block[i2,j2].
                        # So for each block b, we add A_coeffs[m][b] * U2[i2,j2,c] to U4[i,j,col]
                        # where (i,j) is in block b and (i,j) corresponds to (i2,j2) in that block.
                        
                        # Let's iterate over blocks.
                        for b in range(4):
                            if A_coeffs[m][b] == 0: continue
                            # Determine i, j range for block b
                            # b=00: i0=0, j0=0
                            # b=01: i0=0, j0=2
                            # b=10: i0=2, j0=0
                            # b=11: i0=2, j0=2
                            i0 = (b // 2) * 2
                            j0 = (b % 2) * 2
                            i = i0 + i2
                            j = j0 + j2
                            U4[i*4+j][col] += A_coeffs[m][b] * u2_val
            
            # V4 part
            for j2 in range(2):
                for k2 in range(2):
                    v2_val = V2[j2*2+k2][c]
                    if v2_val != 0:
                        for b in range(4):
                            if B_coeffs[m][b] == 0: continue
                            i0 = (b // 2) * 2 # Not used for B, B is col blocks?
                            # B blocks:
                            # B00: j in 0,1; k in 0,1
                            # B01: j in 0,1; k in 2,3
                            # B10: j in 2,3; k in 0,1
                            # B11: j in 2,3; k in 2,3
                            j0 = (b // 2) * 2
                            k0 = (b % 2) * 2
                            j = j0 + j2
                            k = k0 + k2
                            V4[j*4+k][col] += B_coeffs[m][b] * v2_val
            
            # W4 part
            # W2[k2*2+i2, c]
            for k2 in range(2):
                for i2 in range(2):
                    w2_val = W2[k2*2+i2][c]
                    if w2_val != 0:
                        # W depends on C blocks.
                        # C blocks are determined by A and B blocks?
                        # No, C is 4x4.
                        # C_block[i_b, k_b] = sum_{j_b} A_block[i_b, j_b] B_block[j_b, k_b].
                        # The Strassen algorithm computes C_blocks from M_m.
                        # The relation between C_blocks and M_m is fixed.
                        # C00 = M1 - M4 + M6 - M7
                        # C10 = M5 + M7
                        # C01 = M3 + M4
                        # C11 = M1 - M2 + M3 + M5
                        
                        # So for a given C_block (say C00), the coefficient of M_m is fixed.
                        # Let C_coeffs[c_block][m] be the coefficient.
                        # Then W4[k,i,col] = sum_{c_block} C_coeffs[c_block][m] * W2[k2,i2,c]
                        # where (k,i) is in c_block.
                        
                        # C blocks:
                        # C00: k in 0,1; i in 0,1
                        # C01: k in 0,1; i in 2,3
                        # C10: k in 2,3; i in 0,1
                        # C11: k in 2,3; i in 2,3
                        
                        # C_coeffs:
                        # C00: [1, 0, 0, -1, 0, 1, -1]
                        # C10: [0, 0, 0, 0, 1, 0, 1]
                        # C01: [0, 0, 1, 1, 0, 0, 0]
                        # C11: [1, -1, 1, 0, 1, 0, 0]
                        
                        C_coeffs = [
                            [1, 0, 0, -1, 0, 1, -1],
                            [0, 0, 0, 0, 1, 0, 1],
                            [0, 0, 1, 1, 0, 0, 0],
                            [1, -1, 1, 0, 1, 0, 0]
                        ]
                        
                        coeff = C_coeffs[k2*2+i2][m]
                        if coeff == 0: continue
                        
                        for cb in range(4):
                            # (k2,i2) corresponds to which C block?
                            # k2 is row block index (0 or 1), i2 is col block index (0 or 1).
                            # C block index = k2*2 + i2?
                            # C00: k2=0, i2=0 -> 0
                            # C01: k2=0, i2=1 -> 1
                            # C10: k2=1, i2=0 -> 2
                            # C11: k2=1, i2=1 -> 3
                            # Yes.
                            # So for this (k2,i2), it belongs to C block cb = k2*2+i2.
                            # But wait, W2[k2*2+i2, c] is the coefficient for C_{i2,k2}?
                            # No, W2 index is k2*2+i2.
                            # And C_coeffs index is k2*2+i2.
                            # So we just add coeff * w2_val to the appropriate W4 entries.
                            # Which W4 entries?
                            # W4[k*4+i, col].
                            # (k,i) is in C block cb.
                            # k = k0 + k2, i = i0 + i2.
                            # k0 = (cb // 2) * 2
                            # i0 = (cb % 2) * 2
                            # Wait, C block indexing:
                            # C00: k in 0,1; i in 0,1. k0=0, i0=0.
                            # C01: k in 0,1; i in 2,3. k0=0, i0=2.
                            # C10: k in 2,3; i in 0,1. k0=2, i0=0.
                            # C11: k in 2,3; i in 2,3. k0=2, i0=2.
                            # So k0 = (cb // 2) * 2?
                            # cb=0 (00): 0//2=0.
                            # cb=1 (01): 1//2=0.
                            # cb=2 (10): 2//2=1.
                            # cb=3 (11): 3//2=1.
                            # Yes.
                            # i0 = (cb % 2) * 2.
                            # cb=0: 0.
                            # cb=1: 2.
                            # cb=2: 0.
                            # cb=3: 2.
                            # Yes.
                            
                            k0 = (cb // 2) * 2
                            i0 = (cb % 2) * 2
                            k = k0 + k2
                            i = i0 + i2
                            W4[k*4+i][col] += coeff * w2_val
    
    # Now we have U4, V4, W4 for 4x4x4.
    # We need to construct U, V, W for 4x4x8.
    # U: 16x98
    # V: 32x98
    # W: 32x98
    
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    for c in range(49):
        # First half
        for i in range(16):
            U[i][c] = U4[i][c]
        for j in range(4):
            for k in range(4):
                V[j*4+k][c] = V4[j*4+k][c]
        for k in range(4):
            for i in range(4):
                W[k*4+i][c] = W4[k*4+i][c]
        
        # Second half
        c2 = c + 49
        for i in range(16):
            U[i][c2] = U4[i][c]
        for j in range(4):
            for k in range(4):
                V[j*4+k+4][c2] = V4[j*4+k][c] # k+4 for B2
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i][c2] = W4[k*4+i][c] # k+4 for C2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
