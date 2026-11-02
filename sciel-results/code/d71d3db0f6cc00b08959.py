"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen factors for 2x2x2 block:
    # Each factor: (U_entries, V_entries, W_entries)
    # U_entries: list of (i, j, coeff) for A[i,j]
    # V_entries: list of (j, k_local, coeff) for B[j, k_local]
    # W_entries: list of (i, k_local, coeff) for C[i, k_local]
    
    strassen_factors = [
        # M1 = (a00+a11)(b00+b11), contributes to C00+=1, C11+=1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (a10+a11)b00, contributes to C10+=1, C11+=-1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = a00(b01-b11), contributes to C01+=1, C11+=1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = a11(b10-b00), contributes to C00+=1, C10+=1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (a00+a01)b11, contributes to C00+=-1, C01+=1
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (a10-a00)(b00+b01), contributes to C11+=1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (a01-a11)(b10+b11), contributes to C00+=1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_entries, v_entries, w_entries = strassen_factors[s]
            
            # Fill U: row = i*m + j = i*2 + j
            for (i, j, coeff) in u_entries:
                U[i * m + j][col] = coeff
            
            # Fill V: row = j*p + k = j*16 + (2t + k_local)
            for (j, k_local, coeff) in v_entries:
                V[j * p + (2 * t + k_local)][col] = coeff
            
            # Fill W: row = k*n + i = (2t + k_local)*2 + i
            for (i, k_local, coeff) in w_entries:
                W[(2 * t + k_local) * n + i][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
