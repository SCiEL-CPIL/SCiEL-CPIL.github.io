import time

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    This uses the block Strassen algorithm:
    Split A into A_L (2x2) and A_R (2x2).
    Split B into B_TL (2x2), B_TR (2x2) for top rows, and B_BL (2x2), B_BR (2x2) for bottom rows.
    Actually, B is 4x4.
    C = A B = [A_L, A_R] * [B_1; B_2] where B_1 is 2x4 and B_2 is 2x4.
    C = A_L B_1 + A_R B_2.
    
    B_1 = [B_TL, B_TR] (2x2, 2x2)
    B_2 = [B_BL, B_BR] (2x2, 2x2)
    
    So C = A_L [B_TL, B_TR] + A_R [B_BL, B_BR]
       = [A_L B_TL, A_L B_TR] + [A_R B_BL, A_R B_BR]
       = [A_L B_TL + A_R B_BL, A_L B_TR + A_R B_BR]
    
    Each term A_X B_Y is a 2x2 * 2x2 multiplication.
    There are 4 such terms.
    Each 2x2 * 2x2 multiplication can be done with rank 7 (Strassen).
    Total rank = 4 * 7 = 28.
    
    Interface:
      U shape: (n*m, rank) = (8, 28)
      V shape: (m*p, rank) = (16, 28)
      W shape: (p*n, rank) = (8, 28)
      
    Indexing:
      A[i,j]: i in {0,1}, j in {0,1,2,3}. U index: i*4+j
      B[j,k]: j in {0,1,2,3}, k in {0,1,2,3}. V index: j*4+k
      C[i,k]: i in {0,1}, k in {0,1,2,3}. W index: k*2+i (trace convention W[k*n+i])
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0.0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]   # 8 x 28
    
    # Strassen patterns for 2x2 * 2x2 -> 2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11)
    # M3 = (a11)(b12-b22)
    # M4 = (a22)(b21-b11)
    # M5 = (a11+a12)(b22)
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # Local indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_patterns = [
        # a_coeffs, b_coeffs, c_coeffs
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1},     {2: 1, 3: -1}),        # M2
        ({0: 1},     {1: 1, 3: -1}, {1: 1, 3: 1}),        # M3
        ({3: 1},     {2: 1, 0: -1}, {0: 1, 2: 1}),        # M4
        ({0: 1, 1: 1}, {3: 1},     {0: -1, 1: 1}),        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),            # M7
    ]
    
    # Define the 4 blocks of 2x2 multiplications
    # Each block is defined by:
    # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # 
    # Block 1: A_L * B_TL -> C[:, 0:2]
    # A_L is A[:, 0:2] (cols 0,1)
    # B_TL is B[0:2, 0:2] (rows 0,1 cols 0,1)
    # Result adds to C[:, 0:2]
    #
    # Block 2: A_R * B_BL -> C[:, 0:2]
    # A_R is A[:, 2:4] (cols 2,3)
    # B_BL is B[2:4, 0:2] (rows 2,3 cols 0,1)
    # Result adds to C[:, 0:2]
    #
    # Block 3: A_L * B_TR -> C[:, 2:4]
    # A_L is A[:, 0:2]
    # B_TR is B[0:2, 2:4] (rows 0,1 cols 2,3)
    # Result adds to C[:, 2:4]
    #
    # Block 4: A_R * B_BR -> C[:, 2:4]
    # A_R is A[:, 2:4]
    # B_BR is B[2:4, 2:4] (rows 2,3 cols 2,3)
    # Result adds to C[:, 2:4]
    
    blocks = [
        (0, 0, 0, 0),  # A_L * B_TL
        (2, 2, 0, 0),  # A_R * B_BL
        (0, 0, 2, 2),  # A_L * B_TR
        (2, 2, 2, 2),  # A_R * B_BR
    ]
    
    col = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for a_coeffs, b_coeffs, c_coeffs in strassen_patterns:
            # Fill U (A part)
            for local_idx, coeff in a_coeffs.items():
                r = local_idx // 2
                c = local_idx % 2
                global_i = r
                global_j = a_off + c
                U[global_i * m + global_j][col] += coeff
            
            # Fill V (B part)
            for local_idx, coeff in b_coeffs.items():
                r = local_idx // 2
                c = local_idx % 2
                global_j = b_row_off + r
                global_k = b_col_off + c
                V[global_j * p + global_k][col] += coeff
            
            # Fill W (C part)
            for local_idx, coeff in c_coeffs.items():
                r = local_idx // 2
                c = local_idx % 2
                global_i = r
                global_k = c_off + c
                # W index is k*n + i
                W[global_k * n + global_i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
