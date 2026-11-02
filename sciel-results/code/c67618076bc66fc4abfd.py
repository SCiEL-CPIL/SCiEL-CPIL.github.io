import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    # 6 blocks of 2 columns -> 6 * 7 = 42
    # 1 block of 1 column  -> 1 * 4 = 4
    # Total 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a Strassen block for columns k0, k1
    def add_strassen_block(k0, k1, start_col):
        # M1 = (a00 + a11) * (B0k0 + B1k0 + B0k1 + B1k1) ? 
        # Standard Strassen for A * B where B is 2x2.
        # B = [[B0k0, B0k1], [B1k0, B1k1]]
        # A = [[a00, a01], [a10, a11]]
        
        # M1 = (a00 + a11) * (B0k0 + B1k1) ?? No.
        # Let's use the standard 7 products:
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11)b00
        # M3 = a00(b01 - b11)
        # M4 = a11(b00 - b10)
        # M5 = (a00 + a01)b11
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        
        # In our case, the "elements" b00, b01 etc are scalars from B.
        # b00 = B[0, k0]
        # b01 = B[0, k1]
        # b10 = B[1, k0]
        # b11 = B[1, k1]
        
        # Output C:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W.
        # U corresponds to A coefficients.
        # V corresponds to B coefficients.
        # W corresponds to C coefficients.
        
        # Term 1: M1
        U[0][start_col] += 1  # a00
        U[3][start_col] += 1  # a11
        V[0 * p + k0][start_col] += 1 # b00
        V[1 * p + k1][start_col] += 1 # b11
        # C00 (idx 0), C11 (idx 2*1+1=3? No, W[k*n+i])
        # C00: k=0, i=0 -> W[0*2+0] = W[0]
        # C11: k=1, i=1 -> W[1*2+1] = W[3]
        W[0][start_col] += 1
        W[3][start_col] += 1
        
        # Term 2: M2
        U[2][start_col] += 1 # a10
        U[3][start_col] += 1 # a11
        V[0 * p + k0][start_col] += 1 # b00
        # C10: k=0, i=1 -> W[0*2+1] = W[1]
        W[1][start_col] += 1
        
        # Term 3: M3
        U[0][start_col] += 1 # a00
        V[0 * p + k1][start_col] += 1 # b01
        V[1 * p + k1][start_col] -= 1 # -b11
        # C01: k=1, i=0 -> W[1*2+0] = W[2]
        W[2][start_col] += 1
        
        # Term 4: M4
        U[3][start_col] += 1 # a11
        V[0 * p + k0][start_col] += 1 # b00
        V[1 * p + k0][start_col] -= 1 # -b10
        # C00 (W[0]) and C10 (W[1])
        W[0][start_col] += 1
        W[1][start_col] += 1
        
        # Term 5: M5
        U[0][start_col] += 1 # a00
        U[1][start_col] += 1 # a01
        V[1 * p + k1][start_col] += 1 # b11
        # C00 (W[0]) and C01 (W[2])
        W[0][start_col] -= 1
        W[2][start_col] += 1
        
        # Term 6: M6
        U[2][start_col] += 1 # a10
        U[0][start_col] -= 1 # -a00
        V[0 * p + k0][start_col] += 1 # b00
        V[0 * p + k1][start_col] += 1 # b01
        # C11 (W[3])
        W[3][start_col] += 1
        
        # Term 7: M7
        U[1][start_col] += 1 # a01
        U[3][start_col] -= 1 # -a11
        V[1 * p + k0][start_col] += 1 # b10
        V[1 * p + k1][start_col] += 1 # b11
        # C00 (W[0]) and C11 (W[3])
        W[0][start_col] += 1
        W[3][start_col] -= 1
        
        return start_col + 7

    # Process 6 blocks of 2 columns
    for b in range(6):
        k0 = 2 * b
        k1 = 2 * b + 1
        col_idx = add_strassen_block(k0, k1, col_idx)
        
    # Process 1 block of 1 column (column 12)
    # Schoolbook for 2x2 by 2x1
    # C0 = a00*B0 + a01*B1
    # C1 = a10*B0 + a11*B1
    
    # Term 1: a00 * B0
    U[0][col_idx] = 1
    V[0 * p + 12][col_idx] = 1
    W[12 * 2 + 0][col_idx] = 1 # C[0, 12]
    col_idx += 1
    
    # Term 2: a01 * B1
    U[1][col_idx] = 1
    V[1 * p + 12][col_idx] = 1
    W[12 * 2 + 0][col_idx] = 1
    col_idx += 1
    
    # Term 3: a10 * B0
    U[2][col_idx] = 1
    V[0 * p + 12][col_idx] = 1
    W[12 * 2 + 1][col_idx] = 1 # C[1, 12]
    col_idx += 1
    
    # Term 4: a11 * B1
    U[3][col_idx] = 1
    V[1 * p + 12][col_idx] = 1
    W[12 * 2 + 1][col_idx] = 1
    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
