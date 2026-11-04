import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s kill

    # Strassen coefficients for 2x2x2
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Coeffs: list of (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict (I,J) -> val
    # b_coeffs: dict (J,K) -> val
    # c_coeffs: dict (I,K) -> val
    
    strassen_terms = [
        (
            {(0,0): 1, (1,1): 1},   # A
            {(0,0): 1, (1,1): 1},   # B
            {(0,0): 1, (1,1): 1}    # C
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1}
        ),
        (
            {(0,0): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1}
        ),
        (
            {(1,1): 1},
            {(0,1): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): -1}
        ),
    ]

    # Total rank: 28 (4x4x4) + 28 (4x4x2) + 16 (4x4x1) = 72
    total_rank = 28 + 28 + 16

    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 72
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 72

    col = 0

    # Helper to add a Strassen-based 4x4xK decomposition
    # K is the number of columns of B to process
    # cols_offset is the starting column index in B
    def add_strassen_4x4xK(K, cols_offset):
        nonlocal col
        # 7 Strassen terms * 4 schoolbook terms per term = 28 terms
        for t_idx in range(7):
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[t_idx]
            # Expand a_coeffs to linear forms in A
            # A is 4x4. Blocks A[I,J] are 2x2.
            # A[2I+i, 2J+j]
            
            # We need to compute 4 schoolbook terms for the 2x2 product
            # M_t = A'_t B'_t
            # M_t[ip, kp] = sum_{jp} A'_t[ip, jp] B'_t[jp, kp]
            
            for ip in range(2):
                for jp in range(2):
                    for kp in range(2):
                        # This is one rank-1 term
                        # U part: A'_t[ip, jp]
                        # V part: B'_t[jp, kp]
                        # W part: c_coeffs contribution to C[2I+ip, 2K+kp]
                        
                        # Construct U vector (length 16)
                        u_vec = [0] * 16
                        # A'_t[ip, jp] = sum_{I,J} a_IJ * A[2I+ip, 2J+jp]
                        # But wait, a_coeffs is for blocks.
                        # For a specific ip, jp, which blocks contribute?
                        # A[2I+ip, 2J+jp] is in block (I,J).
                        # So we sum over all I,J in a_coeffs.
                        for (I, J), val in a_coeffs.items():
                            row = (2*I + ip) * 4 + (2*J + jp) # A index: i*m + j
                            u_vec[row] += val

                        # Construct V vector (length 28)
                        v_vec = [0] * 28
                        # B'_t[jp, kp] = sum_{J,K} b_JK * B[2J+jp, 2K+kp]
                        # B is 4x7. We are processing columns cols_offset to cols_offset+K-1.
                        # The block structure of B in this part:
                        # B is 4xK. Split into 2x2 blocks?
                        # If K=2, B is 4x2. Blocks B[0,0], B[1,0] (2x2 each? No, 2 rows, 2 cols).
                        # B[2J+jp, 2K+kp + cols_offset]
                        for (J, K), val in b_coeffs.items():
                            # Check if K is within the range of this part
                            if K < K: # K is block index, 0 or 1
                                row = (2*J + jp) * 7 + (2*K + kp + cols_offset) # B index: j*p + k
                                v_vec[row] += val
                            # If K >= K, this term is 0 (not part of this decomposition)
                            
                        # Construct W vector (length 28)
                        w_vec = [0] * 28
                        # C contribution: c_coeffs[I,K] * M_t[ip, kp]
                        # M_t[ip, kp] contributes to C[2I+ip, 2K+kp]
                        # But we only want columns in [cols_offset, cols_offset+K)
                        for (I, K), val in c_coeffs.items():
                            # Check if the resulting column is in range
                            col_c = 2*K + kp + cols_offset
                            if cols_offset <= col_c < cols_offset + K:
                                row = col_c * 4 + (2*I + ip) # W index: k*n + i
                                w_vec[row] += val

                        # Add to global matrices
                        for i in range(16):
                            if u_vec[i] != 0:
                                U[i][col] = u_vec[i]
                        for j in range(28):
                            if v_vec[j] != 0:
                                V[j][col] = v_vec[j]
                        for k in range(28):
                            if w_vec[k] != 0:
                                W[k][col] = w_vec[k]
                        
                        col += 1

    # Part 1: 4x4x4 (cols 0-3)
    add_strassen_4x4xK(4, 0)
    
    # Part 2: 4x4x2 (cols 4-5)
    # Note: The Strassen structure for B is the same, but we only use the first 2 columns of the 4x4 result?
    # No, we treat B as 4x2. The block indices J,K in b_coeffs refer to the 2x2 blocks of the 4x2 matrix.
    # So K can only be 0.
    # Wait, my helper function assumes b_coeffs has K in {0,1}.
    # For 4x4x2, B is 4x2. It has one block column (K=0) and two block rows (J=0,1).
    # The Strassen terms for 2x2x2 assume B is 2x2 (blocks 0,0 and 1,1).
    # For 4x4x2, we can't directly use the same b_coeffs because the "K" dimension is size 1 block.
    # We need to adapt the Strassen terms for rectangular B.
    
    # Let's restart Part 2 logic.
    # For 4x4x2, we can use the same A coefficients.
    # For B, we form linear combinations of the 2x2 blocks B_0 (rows 0-1, cols 4-5) and B_1 (rows 2-3, cols 4-5).
    # Let's call them B_00 and B_10 in the 2x2 block grid of the 4x2 matrix.
    # The Strassen algorithm for 2x2x2 uses B_00, B_01, B_10, B_11.
    # Here we only have B_00, B_10. B_01, B_11 are zero.
    # So we just set the coefficients for B_01, B_11 to 0.
    
    # My helper function `add_strassen_4x4xK` does this correctly if I pass K=2.
    # The loop `for (J, K), val in b_coeffs.items():` will iterate over K=0,1.
    # If K=1, `2*K + kp + cols_offset` will be out of bounds for the 4x2 part?
    # No, cols_offset=4. K=1 -> col 6,7. But we only have cols 4,5.
    # So the condition `if K < K` (which is `if K < 2`) allows K=0,1.
    # But we need to check if the column is valid.
    # `col_c = 2*K + kp + cols_offset`.
    # If K=1, col_c = 2+kp+4 = 6+kp.
    # We only want cols 4,5. So 6,7 are out of range.
    # So the `if cols_offset <= col_c < cols_offset + K` check in W part handles this?
    # No, the V part also needs to be zero.
    # In V part, I added `if K < K`. This is `if K < 2`.
    # I need to check if the column index is within the valid range for B.
    # B has columns 0..6.
    # For Part 2, we only use cols 4,5.
    # So in V part, we should only add if `2*K + kp + cols_offset` is in [4, 6].
    
    # Let's refine the helper to take explicit B column range.
    
    # Actually, simpler:
    # Part 1: 4x4x4. B cols 0-3.
    # Part 2: 4x4x2. B cols 4-5.
    # Part 3: 4x4x1. B col 6.
    
    # My current `col` is 28 after Part 1.
    
    # Let's re-implement Part 2 with a specific check for B columns.
    
    def add_strassen_4x4xK_specific(K, cols_offset, b_cols):
        nonlocal col
        # b_cols is the list of valid column indices in B for this part
        for t_idx in range(7):
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[t_idx]
            for ip in range(2):
                for jp in range(2):
                    for kp in range(2):
                        u_vec = [0] * 16
                        for (I, J), val in a_coeffs.items():
                            row = (2*I + ip) * 4 + (2*J + jp)
                            u_vec[row] += val

                        v_vec = [0] * 28
                        for (J, K), val in b_coeffs.items():
                            b_col = 2*K + kp + cols_offset
                            if b_col in b_cols:
                                row = (2*J + jp) * 7 + b_col
                                v_vec[row] += val
                            
                        w_vec = [0] * 28
                        for (I, K), val in c_coeffs.items():
                            c_col = 2*K + kp + cols_offset
                            if c_col in b_cols: # C columns match B columns
                                row = c_col * 4 + (2*I + ip)
                                w_vec[row] += val

                        for i in range(16):
                            if u_vec[i] != 0:
                                U[i][col] = u_vec[i]
                        for j in range(28):
                            if v_vec[j] != 0:
                                V[j][col] = v_vec[j]
                        for k in range(28):
                            if w_vec[k] != 0:
                                W[k][col] = w_vec[k]
                        
                        col += 1

    # Part 2: 4x4x2 (cols 4-5)
    add_strassen_4x4xK_specific(2, 4, [4, 5])
    
    # Part 3: 4x4x1 (col 6) - Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(1): # Only col 6
                U[i * 4 + j][col] = 1
                V[j * 7 + (6 + k)][col] = 1
                W[(6 + k) * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
