"""Strassen-based rank-49 tensor decomposition for 4x4 matrix multiplication."""
import time
import sys

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 exact tensor decomposition for 4x4 matrix multiplication
    using the 2x2 Strassen algorithm applied to 2x2 blocks.
    
    The 4x4 matrices are partitioned into 2x2 blocks:
    A = [[A00, A01], [A10, A11]]  where each block is 2x2
    B = [[B00, B01], [B10, B11]]
    C = A*B = [[C00, C01], [C10, C11]]
    
    Using Strassen on the block level (7 multiplications of 2x2 blocks),
    and then applying Strassen again within each 2x2 block multiplication,
    we get 7 * 7 = 49 scalar multiplications total.
    
    Index conventions:
    - U has shape (n*m, rank) = (16, 49): U[i*m+j, r]
    - V has shape (m*p, rank) = (16, 49): V[j*p+k, r]  
    - W has shape (p*n, rank) = (16, 49): W[k*n+i, r]
    
    For matrix multiplication C = A*B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Tensor entry: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Strassen 2x2 formulas:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    # Strassen intermediate matrices for 2x2 block level
    # Each "multiplication" at the block level corresponds to 7 scalar Strassen ops
    # So we have 7 block-level multiplications, each contributing 7 scalar multiplications
    
    # Define the 7 Strassen combinations for 2x2 matrices
    # For A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    
    # Block-level Strassen: 7 multiplications
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Within each block multiplication, we apply Strassen again on 2x2 blocks
    # For a 2x2 matrix product X*Y where X,Y are 2x2:
    # x11, x12, x21, x22 are the entries
    # Strassen gives 7 scalar multiplications
    
    # Let's define the scalar-level Strassen for a 2x2 product
    # For X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # s1 = x00 + x11, s2 = y00 + y11 -> m1 = s1*s2
    # s3 = x10 + x11, s4 = y00 -> m2 = s3*s4
    # s5 = x00, s6 = y01 - y11 -> m3 = s5*s6
    # s7 = x11, s8 = y10 - y00 -> m4 = s7*s8
    # s9 = x00 + x01, s10 = y11 -> m5 = s9*s10
    # s11 = x10 - x00, s12 = y00 + y01 -> m6 = s11*s12
    # s13 = x01 - x11, s14 = y10 + y11 -> m7 = s13*s14
    
    # Result:
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # Now, for the block-level Strassen, each "multiplication" is a 2x2 * 2x2 product
    # which itself decomposes into 7 scalar multiplications via Strassen.
    
    # The 7 block-level multiplications produce 7 intermediate 2x2 matrices.
    # Each of these is then used in linear combinations to form the final 4x4 result.
    
    # Let's index the 49 scalar multiplications as (block_idx, scalar_idx) where
    # block_idx in 0..6 and scalar_idx in 0..6
    
    # For each block-level multiplication, we need to determine:
    # 1. Which entries of A (as 2x2 blocks) contribute to the left factor
    # 2. Which entries of B (as 2x2 blocks) contribute to the right factor
    # 3. How the resulting 2x2 product contributes to the final C entries
    
    # Block indices: 
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
    #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3)
    #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3)
    #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    # Block-level Strassen formulas (where each "product" is a 2x2 matrix product):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block-level multiplication M_b = X_b * Y_b (where X_b, Y_b are 2x2 matrices
    # formed from linear combinations of A and B blocks), we apply scalar Strassen.
    
    # Define the 7 block-level left factors (as combinations of A block entries)
    # and right factors (as combinations of B block entries)
    
    # A block entries indexed as: A00 = [[a00, a01], [a10, a11]]
    #                            A01 = [[a02, a03], [a12, a13]]
    #                            A10 = [[a20, a21], [a30, a31]]
    #                            A11 = [[a22, a23], [a32, a33]]
    
    # B block entries indexed as: B00 = [[b00, b01], [b10, b11]]
    #                            B01 = [[b02, b03], [b12, b13]]
    #                            B10 = [[b20, b21], [b30, b31]]
    #                            B11 = [[b22, b23], [b32, b33]]
    
    # C block entries indexed as: C00 = [[c00, c01], [c10, c11]]
    #                            C01 = [[c02, c03], [c12, c13]]
    #                            C10 = [[c20, c21], [c30, c31]]
    #                            C11 = [[c22, c23], [c32, c33]]
    
    # For the block-level Strassen, the 7 left factors are 2x2 matrices:
    # L1 = A00 + A11 = [[a00+a22, a01+a23], [a10+a32, a11+a33]]
    # L2 = A10 + A11 = [[a20+a22, a21+a23], [a30+a32, a31+a33]]
    # L3 = A00       = [[a00, a01], [a10, a11]]
    # L4 = A11       = [[a22, a23], [a32, a33]]
    # L5 = A00 + A01 = [[a00+a02, a01+a03], [a10+a12, a11+a13]]
    # L6 = A10 - A00 = [[a20-a00, a21-a01], [a30-a10, a31-a11]]
    # L7 = A01 - A11 = [[a02-a22, a03-a23], [a12-a32, a13-a33]]
    
    # The 7 right factors are 2x2 matrices:
    # R1 = B00 + B11 = [[b00+b22, b01+b23], [b10+b32, b11+b33]]
    # R2 = B00       = [[b00, b01], [b10, b11]]
    # R3 = B01 - B11 = [[b02-b22, b03-b23], [b12-b32, b13-b33]]
    # R4 = B10 - B00 = [[b20-b00, b21-b01], [b30-b10, b31-b11]]
    # R5 = B11       = [[b22, b23], [b32, b33]]
    # R6 = B00 + B01 = [[b00+b02, b01+b03], [b10+b12, b11+b13]]
    # R7 = B10 + B11 = [[b20+b22, b21+b23], [b30+b32, b31+b33]]
    
    # For each block-level multiplication M_b = L_b * R_b, we apply scalar Strassen.
    # L_b = [[l00, l01], [l10, l11]], R_b = [[r00, r01], [r10, r11]]
    
    # Scalar Strassen for M_b:
    # m_b1 = (l00 + l11) * (r00 + r11)
    # m_b2 = (l10 + l11) * r00
    # m_b3 = l00 * (r01 - r11)
    # m_b4 = l11 * (r10 - r00)
    # m_b5 = (l00 + l01) * r11
    # m_b6 = (l10 - l00) * (r00 + r01)
    # m_b7 = (l01 - l11) * (r10 + r11)
    
    # Result of M_b (2x2 matrix):
    # M_b[0,0] = m_b1 + m_b4 - m_b5 + m_b7
    # M_b[0,1] = m_b3 + m_b5
    # M_b[1,0] = m_b2 + m_b4
    # M_b[1,1] = m_b1 - m_b2 + m_b3 + m_b6
    
    # Now, the final C blocks are linear combinations of M_1 through M_7:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each C block is 2x2, so we have 16 scalar entries in C.
    # Each C[i,k] is a linear combination of the 49 scalar multiplications m_bj.
    
    # Let's build the W matrix (which encodes the linear combinations for C).
    # W[k*n+i, r] where r = block_idx * 7 + scalar_idx
    
    # First, let's define which scalar multiplications contribute to each C entry.
    
    # C00[0,0] = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    # M1[0,0] = m_11 + m_14 - m_15 + m_17
    # M4[0,0] = m_41 + m_44 - m_45 + m_47
    # M5[0,0] = m_51 + m_54 - m_55 + m_57
    # M7[0,0] = m_71 + m_74 - m_75 + m_77
    
    # So C00[0,0] = m_11 + m_14 - m_15 + m_17 + m_41 + m_44 - m_45 + m_47 
    #               - m_51 - m_54 + m_55 - m_57 + m_71 + m_74 - m_75 + m_77
    
    # This is getting complex. Let me use a systematic approach.
    
    # Define the coefficients for each scalar multiplication in each C entry.
    
    # For block b (0-indexed, so b=0 is M1, b=1 is M2, etc.), and scalar index s (0-6):
    # The scalar multiplication m_{b,s} contributes to M_b's 2x2 result as:
    # M_b[0,0] gets: s=0: +1, s=3: +1, s=4: -1, s=6: +1
    # M_b[0,1] gets: s=2: +1, s=4: +1
    # M_b[1,0] gets: s=1: +1, s=3: +1
    # M_b[1,1] gets: s=0: +1, s=1: -1, s=2: +1, s=5: +1
    
    # And M_b contributes to C blocks as:
    # M0 (M1): C00: +1, C11: +1
    # M1 (M2): C10: +1, C11: -1
    # M2 (M3): C01: +1, C11: +1
    # M3 (M4): C00: +1, C10: +1
    # M4 (M5): C00: -1, C01: +1
    # M5 (M6): C11: +1
    # M6 (M7): C00: +1
    
    # C block to scalar index mapping:
    # C00: rows 0-1, cols 0-1 -> C[0,0], C[0,1], C[1,0], C[1,1]
    # C01: rows 0-1, cols 2-3 -> C[0,2], C[0,3], C[1,2], C[1,3]
    # C10: rows 2-3, cols 0-1 -> C[2,0], C[2,1], C[3,0], C[3,1]
    # C11: rows 2-3, cols 2-3 -> C[2,2], C[2,3], C[3,2], C[3,3]
    
    # M_b[0,0] -> C block entry [0,0]
    # M_b[0,1] -> C block entry [0,1]
    # M_b[1,0] -> C block entry [1,0]
    # M_b[1,1] -> C block entry [1,1]
    
    # For C00 (block at rows 0-1, cols 0-1):
    #   M_b[0,0] -> C[0,0], M_b[0,1] -> C[0,1], M_b[1,0] -> C[1,0], M_b[1,1] -> C[1,1]
    # For C01 (block at rows 0-1, cols 2-3):
    #   M_b[0,0] -> C[0,2], M_b[0,1] -> C[0,3], M_b[1,0] -> C[1,2], M_b[1,1] -> C[1,3]
    # For C10 (block at rows 2-3, cols 0-1):
    #   M_b[0,0] -> C[2,0], M_b[0,1] -> C[2,1], M_b[1,0] -> C[3,0], M_b[1,1] -> C[3,1]
    # For C11 (block at rows 2-3, cols 2-3):
    #   M_b[0,0] -> C[2,2], M_b[0,1] -> C[2,3], M_b[1,0] -> C[3,2], M_b[1,1] -> C[3,3]
    
    # Block contribution coefficients:
    # block_contrib[b][c_block] = coefficient
    # c_block: 0=C00, 1=C01, 2=C10, 3=C11
    
    block_contrib = [
        [1, 0, 0, 1],   # M0 (M1): C00:+1, C11:+1
        [0, 0, 1, -1],  # M1 (M2): C10:+1, C11:-1
        [0, 1, 0, 1],   # M2 (M3): C01:+1, C11:+1
        [1, 0, 1, 0],   # M3 (M4): C00:+1, C10:+1
        [-1, 1, 0, 0],  # M4 (M5): C00:-1, C01:+1
        [0, 0, 0, 1],   # M5 (M6): C11:+1
        [1, 0, 0, 0],   # M6 (M7): C00:+1
    ]
    
    # Scalar contribution within each M_b:
    # scalar_contrib[s][pos] where pos: 0=[0,0], 1=[0,1], 2=[1,0], 3=[1,1]
    scalar_contrib = [
        [1, 0, 0, 1],   # s=0 (m1): [0,0]:+1, [1,1]:+1
        [0, 0, 1, -1],  # s=1 (m2): [1,0]:+1, [1,1]:-1
        [0, 1, 0, 1],   # s=2 (m3): [0,1]:+1, [1,1]:+1
        [1, 0, 1, 0],   # s=3 (m4): [0,0]:+1, [1,0]:+1
        [-1, 1, 0, 0],  # s=4 (m5): [0,0]:-1, [0,1]:+1
        [0, 0, 0, 1],   # s=5 (m6): [1,1]:+1
        [1, 0, 0, 0],   # s=6 (m7): [0,0]:+1
    ]
    
    # C block to (row_offset, col_offset) mapping:
    # C00: row_offset=0, col_offset=0
    # C01: row_offset=0, col_offset=2
    # C10: row_offset=2, col_offset=0
    # C11: row_offset=2, col_offset=2
    
    c_block_offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
    # Build W matrix
    # W[k*n+i, r] where r = b*7 + s
    for b in range(7):
        for s in range(7):
            r = b * 7 + s
            for pos in range(4):
                sc_coeff = scalar_contrib[s][pos]
                if sc_coeff == 0:
                    continue
                # pos maps to [0,0], [0,1], [1,0], [1,1] within the C block
                if pos == 0:
                    local_r, local_c = 0, 0
                elif pos == 1:
                    local_r, local_c = 0, 1
                elif pos == 2:
                    local_r, local_c = 1, 0
                else:
                    local_r, local_c = 1, 1
                
                for cb in range(4):
                    bc_coeff = block_contrib[b][cb]
                    if bc_coeff == 0:
                        continue
                    row_off, col_off = c_block_offsets[cb]
                    i = row_off + local_r
                    k = col_off + local_c
                    # W[k*n + i, r]
                    W[k * n + i][r] += bc_coeff * sc_coeff
    
    # Now build U and V matrices
    # U[i*m+j, r] and V[j*p+k, r]
    
    # For each block-level multiplication M_b = L_b * R_b:
    # L_b is a 2x2 matrix formed from linear combinations of A block entries
    # R_b is a 2x2 matrix formed from linear combinations of B block entries
    
    # The scalar Strassen for M_b = L_b * R_b where:
    # L_b = [[l00, l01], [l10, l11]], R_b = [[r00, r01], [r10, r11]]
    
    # m_b1 = (l00 + l11) * (r00 + r11)
    # m_b2 = (l10 + l11) * r00
    # m_b3 = l00 * (r01 - r11)
    # m_b4 = l11 * (r10 - r00)
    # m_b5 = (l00 + l01) * r11
    # m_b6 = (l10 - l00) * (r00 + r01)
    # m_b7 = (l01 - l11) * (r10 + r11)
    
    # Each of these is a product of a linear combination of A entries and a linear combination of B entries.
    # In the tensor decomposition, each scalar multiplication corresponds to a rank-1 term.
    # U and V encode the coefficients of A and B entries respectively.
    
    # Define L_b and R_b as 2x2 matrices of coefficients for A and B block entries.
    # A block entries: A00, A01, A10, A11 (each 2x2)
    # We'll represent each 2x2 block entry by its 4 scalar entries.
    
    # For U: U[i*m+j, r] = coefficient of A[i,j] in the left factor of multiplication r
    # For V: V[j*p+k, r] = coefficient of B[j,k] in the right factor of multiplication r
    
    # Let's define the left and right factors for each of the 7 block-level multiplications.
    
    # L1 = A00 + A11
    # L2 = A10 + A11
    # L3 = A00
    # L4 = A11
    # L5 = A00 + A01
    # L6 = A10 - A00
    # L7 = A01 - A11
    
    # R1 = B00 + B11
    # R2 = B00
    # R3 = B01 - B11
    # R4 = B10 - B00
    # R5 = B11
    # R6 = B00 + B01
    # R7 = B10 + B11
    
    # For each L_b, it's a 2x2 matrix where each entry is a linear combination of A entries.
    # For each R_b, it's a 2x2 matrix where each entry is a linear combination of B entries.
    
    # Let's define the coefficient maps for A and B entries.
    # A entries: A[i,j] for i in 0..3, j in 0..3
    # B entries: B[j,k] for j in 0..3, k in 0..3
    
    # L1 = A00 + A11:
    # L1[0,0] = A[0,0] + A[2,2]
    # L1[0,1] = A[0,1] + A[2,3]
    # L1[1,0] = A[1,0] + A[3,2]
    # L1[1,1] = A[1,1] + A[3,3]
    
    # L2 = A10 + A11:
    # L2[0,0] = A[2,0] + A[2,2]
    # L2[0,1] = A[2,1] + A[2,3]
    # L2[1,0] = A[3,0] + A[3,2]
    # L2[1,1] = A[3,1] + A[3,3]
    
    # L3 = A00:
    # L3[0,0] = A[0,0]
    # L3[0,1] = A[0,1]
    # L3[1,0] = A[1,0]
    # L3[1,1] = A[1,1]
    
    # L4 = A11:
    # L4[0,0] = A[2,2]
    # L4[0,1] = A[2,3]
    # L4[1,0] = A[3,2]
    # L4[1,1] = A[3,3]
    
    # L5 = A00 + A01:
    # L5[0,0] = A[0,0] + A[0,2]
    # L5[0,1] = A[0,1] + A[0,3]
    # L5[1,0] = A[1,0] + A[1,2]
    # L5[1,1] = A[1,1] + A[1,3]
    
    # L6 = A10 - A00:
    # L6[0,0] = A[2,0] - A[0,0]
    # L6[0,1] = A[2,1] - A[0,1]
    # L6[1,0] = A[3,0] - A[1,0]
    # L6[1,1] = A[3,1] - A[1,1]
    
    # L7 = A01 - A11:
    # L7[0,0] = A[0,2] - A[2,2]
    # L7[0,1] = A[0,3] - A[2,3]
    # L7[1,0] = A[1,2] - A[3,2]
    # L7[1,1] = A[1,3] - A[3,3]
    
    # R1 = B00 + B11:
    # R1[0,0] = B[0,0] + B[2,2]
    # R1[0,1] = B[0,1] + B[2,3]
    # R1[1,0] = B[1,0] + B[3,2]
    # R1[1,1] = B[1,1] + B[3,3]
    
    # R2 = B00:
    # R2[0,0] = B[0,0]
    # R2[0,1] = B[0,1]
    # R2[1,0] = B[1,0]
    # R2[1,1] = B[1,1]
    
    # R3 = B01 - B11:
    # R3[0,0] = B[0,2] - B[2,2]
    # R3[0,1] = B[0,3] - B[2,3]
    # R3[1,0] = B[1,2] - B[3,2]
    # R3[1,1] = B[1,3] - B[3,3]
    
    # R4 = B10 - B00:
    # R4[0,0] = B[2,0] - B[0,0]
    # R4[0,1] = B[2,1] - B[0,1]
    # R4[1,0] = B[3,0] - B[1,0]
    # R4[1,1] = B[3,1] - B[1,1]
    
    # R5 = B11:
    # R5[0,0] = B[2,2]
    # R5[0,1] = B[2,3]
    # R5[1,0] = B[3,2]
    # R5[1,1] = B[3,3]
    
    # R6 = B00 + B01:
    # R6[0,0] = B[0,0] + B[0,2]
    # R6[0,1] = B[0,1] + B[0,3]
    # R6[1,0] = B[1,0] + B[1,2]
    # R6[1,1] = B[1,1] + B[1,3]
    
    # R7 = B10 + B11:
    # R7[0,0] = B[2,0] + B[2,2]
    # R7[0,1] = B[2,1] + B[2,3]
    # R7[1,0] = B[3,0] + B[3,2]
    # R7[1,1] = B[3,1] + B[3,3]
    
    # Now, for each block-level multiplication b and scalar index s, we have:
    # m_{b,s} = (linear combo of A entries) * (linear combo of B entries)
    
    # The scalar Strassen formulas:
    # s=0: (l00+l11) * (r00+r11)
    # s=1: (l10+l11) * r00
    # s=2: l00 * (r01-r11)
    # s=3: l11 * (r10-r00)
    # s=4: (l00+l01) * r11
    # s=5: (l10-l00) * (r00+r01)
    # s=6: (l01-l11) * (r10+r11)
    
    # For each (b, s), the left factor is a linear combination of A entries,
    # and the right factor is a linear combination of B entries.
    
    # U[i*m+j, r] = coefficient of A[i,j] in the left factor
    # V[j*p+k, r] = coefficient of B[j,k] in the right factor
    
    # Let's define the left and right factor coefficients for each (b, s).
    
    # For block b, L_b = [[L_b[0,0], L_b[0,1]], [L_b[1,0], L_b[1,1]]]
    # where each L_b[i,j] is a dict mapping A index (i_a, j_a) to coefficient
    
    # Similarly for R_b.
    
    # Define L_b as list of 4 dicts (for positions [0,0], [0,1], [1,0], [1,1])
    # Each dict maps (i_a, j_a) -> coeff
    
    # L1:
    L_defs = [
        # L1 = A00 + A11
        [
            {(0,0): 1, (2,2): 1},  # [0,0]
            {(0,1): 1, (2,3): 1},  # [0,1]
            {(1,0): 1, (3,2): 1},  # [1,0]
            {(1,1): 1, (3,3): 1},  # [1,1]
        ],
        # L2 = A10 + A11
        [
            {(2,0): 1, (2,2): 1},
            {(2,1): 1, (2,3): 1},
            {(3,0): 1, (3,2): 1},
            {(3,1): 1, (3,3): 1},
        ],
        # L3 = A00
        [
            {(0,0): 1},
            {(0,1): 1},
            {(1,0): 1},
            {(1,1): 1},
        ],
        # L4 = A11
        [
            {(2,2): 1},
            {(2,3): 1},
            {(3,2): 1},
            {(3,3): 1},
        ],
        # L5 = A00 + A01
        [
            {(0,0): 1, (0,2): 1},
            {(0,1): 1, (0,3): 1},
            {(1,0): 1, (1,2): 1},
            {(1,1): 1, (1,3): 1},
        ],
        # L6 = A10 - A00
        [
            {(2,0): 1, (0,0): -1},
            {(2,1): 1, (0,1): -1},
            {(3,0): 1, (1,0): -1},
            {(3,1): 1, (1,1): -1},
        ],
        # L7 = A01 - A11
        [
            {(0,2): 1, (2,2): -1},
            {(0,3): 1, (2,3): -1},
            {(1,2): 1, (3,2): -1},
            {(1,3): 1, (3,3): -1},
        ],
    ]
    
    # R_defs similar
    R_defs = [
        # R1 = B00 + B11
        [
            {(0,0): 1, (2,2): 1},
            {(0,1): 1, (2,3): 1},
            {(1,0): 1, (3,2): 1},
            {(1,1): 1, (3,3): 1},
        ],
        # R2 = B00
        [
            {(0,0): 1},
            {(0,1): 1},
            {(1,0): 1},
            {(1,1): 1},
        ],
        # R3 = B01 - B11
        [
            {(0,2): 1, (2,2): -1},
            {(0,3): 1, (2,3): -1},
            {(1,2): 1, (3,2): -1},
            {(1,3): 1, (3,3): -1},
        ],
        # R4 = B10 - B00
        [
            {(2,0): 1, (0,0): -1},
            {(2,1): 1, (0,1): -1},
            {(3,0): 1, (1,0): -1},
            {(3,1): 1, (1,1): -1},
        ],
        # R5 = B11
        [
            {(2,2): 1},
            {(2,3): 1},
            {(3,2): 1},
            {(3,3): 1},
        ],
        # R6 = B00 + B01
        [
            {(0,0): 1, (0,2): 1},
            {(0,1): 1, (0,3): 1},
            {(1,0): 1, (1,2): 1},
            {(1,1): 1, (1,3): 1},
        ],
        # R7 = B10 + B11
        [
            {(2,0): 1, (2,2): 1},
            {(2,1): 1, (2,3): 1},
            {(3,0): 1, (3,2): 1},
            {(3,1): 1, (3,3): 1},
        ],
    ]
    
    # For each (b, s), determine the left and right factor coefficients.
    # l_positions: which positions of L_b are used, with what signs
    # r_positions: which positions of R_b are used, with what signs
    
    # s=0: (l00+l11) * (r00+r11)
    # s=1: (l10+l11) * r00
    # s=2: l00 * (r01-r11)
    # s=3: l11 * (r10-r00)
    # s=4: (l00+l01) * r11
    # s=5: (l10-l00) * (r00+r01)
    # s=6: (l01-l11) * (r10+r11)
    
    # Position indices: 0=[0,0], 1=[0,1], 2=[1,0], 3=[1,1]
    
    s_defs = [
        # (l_coeffs, r_coeffs) where each is list of (pos, coeff)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # s=0
        ([(2, 1), (3, 1)], [(0, 1)]),            # s=1
        ([(0, 1)], [(1, 1), (3, -1)]),           # s=2
        ([(3, 1)], [(2, 1), (0, -1)]),           # s=3
        ([(0, 1), (1, 1)], [(3, 1)]),            # s=4
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # s=5
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # s=6
    ]
    
    # Now build U and V
    for b in range(7):
        for s in range(7):
            r = b * 7 + s
            l_coeffs, r_coeffs = s_defs[s]
            
            # Build left factor coefficient map
            left_map = {}  # (i_a, j_a) -> coeff
            for pos, coeff in l_coeffs:
                for (ia, ja), c in L_defs[b][pos].items():
                    left_map[(ia, ja)] = left_map.get((ia, ja), 0) + coeff * c
            
            # Build right factor coefficient map
            right_map = {}  # (jb, kb) -> coeff
            for pos, coeff in r_coeffs:
                for (jb, kb), c in R_defs[b][pos].items():
                    right_map[(jb, kb)] = right_map.get((jb, kb), 0) + coeff * c
            
            # Fill U
            for (ia, ja), coeff in left_map.items():
                if coeff != 0:
                    U[ia * m + ja][r] = coeff
            
            # Fill V
            for (jb, kb), coeff in right_map.items():
                if coeff != 0:
                    V[jb * p + kb][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
