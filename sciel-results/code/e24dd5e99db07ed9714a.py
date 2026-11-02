"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen decomposition for 2x2x2 blocks
    # n=2, m=4, p=8
    # Partition m into 2 blocks of 2, p into 4 blocks of 2
    # Total 8 blocks of 2x2x2
    # Each block has rank 7
    
    # Strassen matrices for 2x2x2
    # U, V, W are 4x7, 4x7, 4x7
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U_strassen = [
        [1, 0, 0, 0, 0, 0, 0], # i=0, j=0
        [0, 1, 0, 0, 1, 0, 0], # i=0, j=1
        [0, 0, 1, 1, 0, 1, 0], # i=1, j=0
        [0, 0, 0, 0, 1, 1, 1]  # i=1, j=1
    ]
    
    V_strassen = [
        [1, 0, 1, 0, 0, 0, 0], # j=0, k=0
        [0, 1, 0, 1, 0, 0, 0], # j=0, k=1
        [0, 0, 0, 0, 1, 1, 0], # j=1, k=0
        [0, 0, 0, 0, 0, 1, 1]  # j=1, k=1
    ]
    
    W_strassen = [
        [1, 0, 0, 0, 0, 1, 0], # k=0, i=0
        [0, 0, 0, 0, 1, 0, 1], # k=0, i=1
        [0, 0, 1, 0, 0, 0, 0], # k=1, i=0
        [0, 1, 0, 1, 0, 0, 0]  # k=1, i=1
    ]
    
    # Verify Strassen:
    # M1 = A00 * B00
    # M2 = A10 * (B00 + B01)
    # M3 = (A00 + A01) * B10
    # M4 = A11 * B11
    # M5 = (A00 + A10) * (B00 + B10)
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A10 - A00) * (B01 - B11)
    # C00 = M1 + M4 - M5 + M6
    # C01 = M3 + M4
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let's re-verify the matrices against this.
    # U (A):
    # r0: A00
    # r1: A10
    # r2: A00
    # r3: A01
    # r4: A00
    # r5: A01
    # r6: A10
    # Wait, my U_strassen above:
    # r0: A00 (1,0,0,0)
    # r1: A10 (0,1,0,0)
    # r2: A00 (0,0,1,0) -> A10? No.
    
    # Let's build it correctly.
    # r0: A00, B00, C00
    # r1: A10, B00+B01, C00? No.
    
    # Standard Strassen:
    # M1 = A00 * B00
    # M2 = A10 * (B00 + B01)
    # M3 = (A00 + A01) * B10
    # M4 = A11 * B11
    # M5 = (A00 + A10) * (B00 + B10)
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A10 - A00) * (B01 - B11)
    
    # U (4x7):
    # r0: 1, 0, 0, 0 (A00)
    # r1: 0, 1, 0, 0 (A10)
    # r2: 1, 0, 1, 0 (A00+A01)
    # r3: 0, 0, 0, 1 (A11)
    # r4: 1, 1, 0, 0 (A00+A10)
    # r5: 0, 0, 1, -1 (A01-A11)
    # r6: -1, 1, 0, 0 (A10-A00)
    
    U_s = [
        [1, 0, 1, 0, 1, 0, -1],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, -1, 0]
    ]
    
    # V (4x7):
    # r0: 1, 0, 0, 0 (B00)
    # r1: 1, 1, 0, 0 (B00+B01)
    # r2: 0, 0, 1, 0 (B10)
    # r3: 0, 0, 0, 1 (B11)
    # r4: 1, 0, 1, 0 (B00+B10)
    # r5: 0, 0, 1, 1 (B10+B11)
    # r6: 0, 1, 0, -1 (B01-B11)
    
    V_s = [
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, -1]
    ]
    
    # W (4x7):
    # C00 = M1 + M4 - M5 + M6
    # C10 = M2 + M4
    # C01 = M3 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # W[k*2+i, r]
    # k=0, i=0 (C00): 1, 0, 0, 1, -1, 1, 0
    # k=0, i=1 (C10): 0, 1, 0, 1, 0, 0, 0
    # k=1, i=0 (C01): 0, 0, 1, 1, 0, 0, 0
    # k=1, i=1 (C11): 1, -1, 1, 0, 0, 0, 1
    
    W_s = [
        [1, 0, 0, 1, -1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 1]
    ]
    
    rank = 8 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_block in range(n // 2):
        for j_block in range(m // 2):
            for k_block in range(p // 2):
                for r in range(7):
                    # U index: (i_block*2 + i, j_block*2 + j)
                    for i in range(2):
                        for j in range(2):
                            u_val = U_s[i*2+j][r]
                            if u_val != 0:
                                U[(i_block*2+i)*m + (j_block*2+j)][col] = u_val
                    for j in range(2):
                        for k in range(2):
                            v_val = V_s[j*2+k][r]
                            if v_val != 0:
                                V[(j_block*2+j)*p + (k_block*2+k)][col] = v_val
                    for k in range(2):
                        for i in range(2):
                            w_val = W_s[k*2+i][r]
                            if w_val != 0:
                                W[(k_block*2+k)*n + (i_block*2+i)][col] = w_val
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
