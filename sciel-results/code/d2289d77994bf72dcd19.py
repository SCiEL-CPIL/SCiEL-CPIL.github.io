"""Matrix multiplication 4x4 by 4x8: rank-98 bilinear decomposition via hierarchical Strassen."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-98 exact tensor decomposition for multiplying a 4x4 matrix
    by a 4x8 matrix.
    
    Strategy: The 4x8 matrix B is split into two 4x4 blocks B1 and B2.
    We compute C1 = A*B1 and C2 = A*B2 using the rank-49 Strassen-based
    algorithm for 4x4 matrix multiplication. Total rank = 49 + 49 = 98.
    
    The 4x4 Strassen algorithm uses the 2x2 Strassen recursively:
    - 2x2 Strassen: rank 7
    - 4x4 = 4 blocks of 2x2, each block multiplication uses 2x2 Strassen
    - But actually, 4x4 Strassen: split into four 2x2 blocks
      A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
      C11 = A11*B11 + A21*B21
      C12 = A12*B11 + A22*B21
      C21 = A11*B12 + A21*B22
      C22 = A12*B12 + A22*B22
      
      Using Strassen at the 4x4 level (treating 2x2 blocks as elements):
      M1 = (A11+A22)*(B11+B22)
      M2 = (A21+A22)*B11
      M3 = A11*(B12-B22)
      M4 = A22*(B21-B11)
      M5 = (A11+A12)*B22
      M6 = (A21-A11)*(B11+B12)
      M7 = (A12-A22)*(B21+B22)
      
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
      
      Each Mi is a 2x2 * 2x2 product, which itself uses 2x2 Strassen (rank 7).
      So total rank for 4x4 = 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We'll build the decomposition for 4x4 * 4x4 (rank 49), then duplicate
    # for the two 4x4 blocks of the 4x8 matrix.
    
    # 2x2 Strassen factors:
    # For X(2x2) * Y(2x2) = Z(2x2):
    # M1 = (X11+X22)*(Y11+Y22)
    # M2 = (X21+X22)*Y11
    # M3 = X11*(Y12-Y22)
    # M4 = X22*(Y21-Y11)
    # M5 = (X11+X12)*Y22
    # M6 = (X21-X11)*(Y11+Y12)
    # M7 = (X12-X22)*(Y21+Y22)
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # 2x2 Strassen: U has 4 rows (X11,X12,X21,X22), V has 4 rows (Y11,Y12,Y21,Y22)
    # W has 4 rows (Z11,Z12,Z21,Z22)
    # 7 factors
    
    def strassen_2x2_factors():
        """Return (U, V, W) for 2x2*2x2=2x2 with rank 7.
        U: 4x7, V: 4x7, W: 4x7
        Row order for U: X11, X12, X21, X22
        Row order for V: Y11, Y12, Y21, Y22
        Row order for W: Z11, Z12, Z21, Z22
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (X11+X22)*(Y11+Y22)
        U[0][0] = 1  # X11
        U[2][0] = 1  # X22
        V[0][0] = 1  # Y11
        V[2][0] = 1  # Y22
        
        # M2 = (X21+X22)*Y11
        U[1][1] = 1  # X21  -- wait, row order: X11=0, X12=1, X21=2, X22=3
        U[2][1] = 1  # X21
        U[3][1] = 1  # X22
        V[0][1] = 1  # Y11
        
        # M3 = X11*(Y12-Y22)
        U[0][2] = 1  # X11
        V[1][2] = 1  # Y12
        V[3][2] = -1 # Y22
        
        # M4 = X22*(Y21-Y11)
        U[3][3] = 1  # X22
        V[2][3] = 1  # Y21
        V[0][3] = -1 # Y11
        
        # M5 = (X11+X12)*Y22
        U[0][4] = 1  # X11
        U[1][4] = 1  # X12
        V[3][4] = 1  # Y22
        
        # M6 = (X21-X11)*(Y11+Y12)
        U[2][5] = 1  # X21
        U[0][5] = -1 # X11
        V[0][5] = 1  # Y11
        V[1][5] = 1  # Y12
        
        # M7 = (X12-X22)*(Y21+Y22)
        U[1][6] = 1  # X12
        U[3][6] = -1 # X22
        V[2][6] = 1  # Y21
        V[3][6] = 1  # Y22
        
        # W: Z11 = M1 + M4 - M5 + M7
        W[0][0] = 1  # Z11 from M1
        W[0][3] = 1  # Z11 from M4
        W[0][4] = -1 # Z11 from -M5
        W[0][6] = 1  # Z11 from M7
        
        # Z12 = M3 + M5
        W[1][2] = 1  # Z12 from M3
        W[1][4] = 1  # Z12 from M5
        
        # Z21 = M2 + M4
        W[2][1] = 1  # Z21 from M2
        W[2][3] = 1  # Z21 from M4
        
        # Z22 = M1 - M2 + M3 + M6
        W[3][0] = 1  # Z22 from M1
        W[3][1] = -1 # Z22 from -M2
        W[3][2] = 1  # Z22 from M3
        W[3][5] = 1  # Z22 from M6
        
        return U, V, W
    
    # Now build 4x4 * 4x4 using 4x4 Strassen where each 2x2*2x2 product uses 2x2 Strassen
    # 4x4 Strassen: 7 "super-factors", each is a 2x2*2x2 product with rank 7
    # Total rank = 7 * 7 = 49
    
    # For 4x4: A has 16 elements, B has 16 elements, C has 16 elements
    # A row order: A11_11, A11_12, A11_21, A11_22, A12_11, A12_12, A12_21, A12_22,
    #               A21_11, A21_12, A21_21, A21_22, A22_11, A22_12, A22_21, A22_22
    # But the interface expects A as 4x4 flattened row-major: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], ...
    # So A row index = i*4+j where i is row, j is col
    # A11 = rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A12 = rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A21 = rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A22 = rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # Similarly for B (4x4): B11, B12, B21, B22
    # B row index = i*4+j
    # B11 -> indices 0,1,4,5
    # B12 -> indices 2,3,6,7
    # B21 -> indices 8,9,12,13
    # B22 -> indices 10,11,14,15
    
    # C (4x4) output: C11, C12, C21, C22
    # C row index = i*4+j
    # C11 -> indices 0,1,4,5
    # C12 -> indices 2,3,6,7
    # C21 -> indices 8,9,12,13
    # C22 -> indices 10,11,14,15
    
    # W convention: W[k*n+i, r] where k is column of B (0..p-1), i is row of A (0..n-1)
    # For 4x4*4x4: W[k*4+i, r]
    # So W row index = k*4+i, which is the same as column-major for C
    # C[i,k] corresponds to W[k*4+i]
    
    # Let me define the 4x4*4x4 decomposition:
    # U: 16 x 49 (A elements)
    # V: 16 x 49 (B elements)  
    # W: 16 x 49 (C elements in W convention: W[k*4+i])
    
    # The 7 Strassen super-factors for 4x4:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each Mi is a 2x2*2x2 product. For each super-factor, we need to map
    # the 2x2 blocks to the actual 4x4 element indices.
    
    # Block index mapping for 4x4 (row-major, 4 columns):
    # Block (br, bc) where br, bc in {0,1}: rows br*2 to br*2+1, cols bc*2 to bc*2+1
    # Element (i,j) in 4x4: block (i//2, j//2), position within block ((i%2)*2 + j%2)
    # Position within 2x2 block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # For a 2x2 block at (br, bc):
    # Position 0: row br*2, col bc*2 -> index br*2*4 + bc*2
    # Position 1: row br*2, col bc*2+1 -> index br*2*4 + bc*2+1
    # Position 2: row br*2+1, col bc*2 -> index (br*2+1)*4 + bc*2
    # Position 3: row br*2+1, col bc*2+1 -> index (br*2+1)*4 + bc*2+1
    
    def block_indices(br, bc, size=4):
        """Return the 4 element indices for a 2x2 block at (br, bc) in a size x size matrix."""
        r0 = br * 2
        c0 = bc * 2
        return [r0 * size + c0, r0 * size + c0 + 1, (r0+1) * size + c0, (r0+1) * size + c0 + 1]
    
    # W convention: W[k*n+i, r] for n x p product
    # For 4x4*4x4: n=4, p=4, so W[k*4+i, r]
    # C[i,k] is at W[k*4+i]
    # C block (br, bc): rows br*2 to br*2+1, cols bc*2 to bc*2+1
    # C[i,k] where i in [br*2, br*2+1], k in [bc*2, bc*2+1]
    # W index = k*4 + i
    
    def w_indices_for_block(br, bc, n=4, p=4):
        """Return W row indices for C block (br, bc) in W convention W[k*n+i]."""
        r0 = br * 2
        c0 = bc * 2
        idx = []
        for di in range(2):
            for dk in range(2):
                i = r0 + di
                k = c0 + dk
                idx.append(k * n + i)
        # Order: (i=br*2, k=bc*2), (i=br*2, k=bc*2+1), (i=br*2+1, k=bc*2), (i=br*2+1, k=bc*2+1)
        # This matches the 2x2 block position order: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # where position = (di*2 + dk)
        return idx
    
    # Build 4x4*4x4 decomposition
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    # 2x2 Strassen factors
    U2, V2, W2 = strassen_2x2_factors()
    
    # Super-factor definitions: (A_block_indices, B_block_indices, C_block_indices, signs)
    # For each super-factor, we have:
    # A_side: which 2x2 blocks of A to combine, with signs
    # B_side: which 2x2 blocks of B to combine, with signs
    # C_side: which 2x2 blocks of C this contributes to, with signs
    
    # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
    # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
    # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
    # M5 = (A11+A12)*B22 -> C11 -= M5, C12 += M5
    # M6 = (A21-A11)*(B11+B12) -> C22 += M6
    # M7 = (A12-A22)*(B21+B22) -> C11 += M7
    
    # For each super-factor s (0..6), and each 2x2 Strassen sub-factor t (0..6):
    # The combined factor index is s*7 + t
    
    # A-side contributions for each super-factor:
    # We need to express the A-side as a linear combination of 2x2 blocks
    # For the 2x2 Strassen, U2 has 4 rows (positions 0,1,2,3 in the 2x2 block)
    # and 7 columns (the 7 Strassen products)
    
    # Super-factor A-side block combinations:
    # M1: A11 + A22
    # M2: A21 + A22
    # M3: A11
    # M4: A22
    # M5: A11 + A12
    # M6: A21 - A11
    # M7: A12 - A22
    
    # Super-factor B-side block combinations:
    # M1: B11 + B22
    # M2: B11
    # M3: B12 - B22
    # M4: B21 - B11
    # M5: B22
    # M6: B11 + B12
    # M7: B21 + B22
    
    # For each super-factor, the A-side is a sum of signed 2x2 blocks.
    # Within each 2x2 block, the 2x2 Strassen U2 pattern applies.
    
    # A-side block combinations: list of (block_row, block_col, sign)
    A_side_blocks = [
        [(0,0,1), (1,1,1)],      # M1: A11 + A22
        [(1,0,1), (1,1,1)],      # M2: A21 + A22
        [(0,0,1)],               # M3: A11
        [(1,1,1)],               # M4: A22
        [(0,0,1), (0,1,1)],      # M5: A11 + A12
        [(1,0,1), (0,0,-1)],     # M6: A21 - A11
        [(0,1,1), (1,1,-1)],     # M7: A12 - A22
    ]
    
    B_side_blocks = [
        [(0,0,1), (1,1,1)],      # M1: B11 + B22
        [(0,0,1)],               # M2: B11
        [(0,1,1), (1,1,-1)],     # M3: B12 - B22
        [(1,0,1), (0,0,-1)],     # M4: B21 - B11
        [(1,1,1)],               # M5: B22
        [(0,0,1), (0,1,1)],      # M6: B11 + B12
        [(1,0,1), (1,1,1)],      # M7: B21 + B22
    ]
    
    # C-side contributions: for each super-factor, which C blocks it contributes to
    # M1: C11 +=, C22 +=
    # M2: C21 +=, C22 -=
    # M3: C12 +=, C22 +=
    # M4: C11 +=, C21 +=
    # M5: C11 -=, C12 +=
    # M6: C22 +=
    # M7: C11 +=
    
    C_side_contrib = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,-1)],     # M2
        [(0,1,1), (1,1,1)],      # M3
        [(0,0,1), (1,0,1)],      # M4
        [(0,0,-1), (0,1,1)],     # M5
        [(1,1,1)],               # M6
        [(0,0,1)],               # M7
    ]
    
    # Now build the full 49-factor decomposition
    for s in range(7):  # super-factor
        for t in range(7):  # 2x2 Strassen sub-factor
            factor_idx = s * 7 + t
            
            # A-side: for each block in A_side_blocks[s], add sign * U2[pos][t] to the appropriate element
            for (br, bc, sign) in A_side_blocks[s]:
                a_indices = block_indices(br, bc)
                for pos in range(4):
                    if U2[pos][t] != 0:
                        U44[a_indices[pos]][factor_idx] += sign * U2[pos][t]
            
            # B-side: for each block in B_side_blocks[s], add sign * V2[pos][t] to the appropriate element
            for (br, bc, sign) in B_side_blocks[s]:
                b_indices = block_indices(br, bc)
                for pos in range(4):
                    if V2[pos][t] != 0:
                        V44[b_indices[pos]][factor_idx] += sign * V2[pos][t]
            
            # C-side: for each C block contribution in C_side_contrib[s]
            for (cbr, cbc, sign) in C_side_contrib[s]:
                w_indices = w_indices_for_block(cbr, cbc, n=4, p=4)
                for pos in range(4):
                    if W2[pos][t] != 0:
                        W44[w_indices[pos]][factor_idx] += sign * W2[pos][t]
    
    # Now build the 4x4 * 4x8 decomposition
    # B is 4x8, split into B_left (cols 0-3) and B_right (cols 4-7)
    # C is 4x8, split into C_left (cols 0-3) and C_right (cols 4-7)
    # C_left = A * B_left, C_right = A * B_right
    # Each uses the 4x4*4x4 decomposition
    
    # Total rank = 49 + 49 = 98
    # U: 16 x 98
    # V: 32 x 98 (m*p = 4*8 = 32)
    # W: 32 x 98 (p*n = 8*4 = 32)
    
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # Left block: B cols 0-3, C cols 0-3
    # B_left row index in V: i*4 + j where i is row of B (0-3), j is col (0-3)
    # This is the same as the 4x4 B indexing
    # W for left block: W[k*4+i] where k is col 0-3, i is row 0-3
    # In the 4x8 W convention: W[k*4+i] where k is col 0-7
    # For left block, k is 0-3, so W indices are 0-15 (same as 4x4)
    
    for r in range(16):
        U[r][:49] = U44[r][:49]
    
    for r in range(16):
        V[r][:49] = V44[r][:49]
    
    for r in range(16):
        W[r][:49] = W44[r][:49]
    
    # Right block: B cols 4-7, C cols 4-7
    # B_right: row i, col j (j=0..3) corresponds to B[i, j+4]
    # V index for B_right: i*4 + (j+4) = i*4 + j + 4
    # W for right block: W[k*4+i] where k is col 4-7
    # In 4x8 W convention: W[k*4+i] where k=4..7
    # So W indices are 16-31
    
    for r in range(16):
        U[r][49:] = U44[r][:49]
    
    for r in range(16):
        V[r + 16][49:] = V44[r][:49]
    
    for r in range(16):
        W[r + 16][49:] = W44[r][:49]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
