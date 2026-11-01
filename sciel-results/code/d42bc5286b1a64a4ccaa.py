"""
Rewritten program for Matrix Multiplication 2x2x16 using Rank-56 Strassen Decomposition.
"""

def construct():
    n, m, p = 2, 2, 16
    # The multiplication decomposes into 8 independent 2x2x2 blocks.
    # Each 2x2x2 block can be multiplied with rank 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    rank = 56
    
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen coefficients for 2x2 matrix multiplication.
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U coefficients (for A entries: a00, a01, a10, a11)
    # Order of A entries in U rows:
    # Row 0: a00 (i=0, j=0)
    # Row 1: a01 (i=0, j=1)
    # Row 2: a10 (i=1, j=0)
    # Row 3: a11 (i=1, j=1)
    
    U_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [1, 0, -1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # V coefficients (for B entries in a 2x2 block: b00, b01, b10, b11)
    # Order of B entries for a fixed j, k_local:
    # V Row for j=0, k_local=0: b00
    # V Row for j=0, k_local=1: b01
    # V Row for j=1, k_local=0: b10
    # V Row for j=1, k_local=1: b11
    
    V_coeffs = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    # W coefficients (for C entries in a 2x2 block: c00, c01, c10, c11)
    # Output equations:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # W matrix columns correspond to M1..M7. Rows correspond to c00, c01, c10, c11.
    # W is transposed in the sense that we fill columns.
    # W[row_c][col_m] = coeff
    W_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 1, 0, 0, 1, 0]    # c11
    ]
    
    num_blocks = p // 2  # 8 blocks
    
    for l in range(num_blocks):
        # Global k indices for this block
        k0 = 2 * l
        k1 = 2 * l + 1
        
        for m_idx in range(7):
            col = l * 7 + m_idx
            
            # Fill U
            # U rows: 0:a00, 1:a01, 2:a10, 3:a11
            for u_row in range(4):
                U[u_row][col] = U_coeffs[m_idx][u_row]
                
            # Fill V
            # V rows for this block:
            # j=0, k=k0 (b00) -> row 0*16 + k0
            # j=0, k=k1 (b01) -> row 0*16 + k1
            # j=1, k=k0 (b10) -> row 1*16 + k0
            # j=1, k=k1 (b11) -> row 1*16 + k1
            # V_coeffs[m_idx] = [coeff_b00, coeff_b01, coeff_b10, coeff_b11]
            V[0 * 16 + k0][col] = V_coeffs[m_idx][0]
            V[0 * 16 + k1][col] = V_coeffs[m_idx][1]
            V[1 * 16 + k0][col] = V_coeffs[m_idx][2]
            V[1 * 16 + k1][col] = V_coeffs[m_idx][3]
            
            # Fill W
            # W rows for this block:
            # i=0, k=k0 (c00) -> row k0*2 + 0
            # i=0, k=k1 (c01) -> row k1*2 + 0
            # i=1, k=k0 (c10) -> row k0*2 + 1
            # i=1, k=k1 (c11) -> row k1*2 + 1
            # W_coeffs[c_row][m_idx]
            W[k0 * 2 + 0][col] = W_coeffs[0][m_idx] # c00
            W[k1 * 2 + 0][col] = W_coeffs[1][m_idx] # c01
            W[k0 * 2 + 1][col] = W_coeffs[2][m_idx] # c10
            W[k1 * 2 + 1][col] = W_coeffs[3][m_idx] # c11
            
    return {"U": U, "V": V, "W": W}
