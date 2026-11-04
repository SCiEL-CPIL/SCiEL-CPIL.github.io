"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-7 decomposition for 2x2x2
    # Indices for U, V: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for W: k*n+i -> 00, 10, 01, 11 -> 0, 1, 2, 3 (k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
    # Note: W index is k*2 + i.
    
    U7 = [
        [1, 0, 0, 1],  # P0
        [0, 0, 1, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 0, 0, 1],  # P3
        [1, 1, 0, 0],  # P4
        [-1, 0, 1, 0], # P5
        [0, 1, 0, -1]  # P6
    ]
    V7 = [
        [1, 0, 0, 1],  # P0
        [1, 0, 0, 0],  # P1
        [0, 0, 1, -1], # P2
        [-1, 1, 0, 0], # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [0, 0, 1, 1]   # P6
    ]
    W7 = [
        [1, 0, 0, 1],  # P0: C00, C11
        [0, 1, 0, -1], # P1: C10, C11
        [0, 0, 1, 1],  # P2: C01, C11
        [1, 0, 1, 0],  # P3: C00, C01
        [-1, 0, 1, 0], # P4: C00, C01
        [0, 1, 0, 1],  # P5: C10, C11
        [-1, 0, 0, 0]  # P6: C00
    ]
    
    # Verify 2x2x2
    def verify_2x2():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for r in range(7):
                        s += U7[r][i*2+j] * V7[r][j*2+k] * W7[r][k*2+i]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_2x2():
        # Fallback to schoolbook for 4x4x5
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Construct 4x4x4 from tensor product of 2x2x2
    rank444 = 49
    U444 = [[0] * rank444 for _ in range(16)]
    V444 = [[0] * rank444 for _ in range(16)]
    W444 = [[0] * rank444 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            u_val = U7[r1][i1*2+j1] * U7[r2][i2*2+j2]
                            U444[i*4+j][r] = u_val
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    v_val = V7[r1][j1*2+k1] * V7[r2][j2*2+k2]
                                    V444[j*4+k][r] = v_val
                                    w_val = W7[r1][k1*2+i1] * W7[r2][k2*2+i2]
                                    W444[k*4+i][r] = w_val

    # Verify 4x4x4
    def verify_4x4x4():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0.0
                    for r in range(rank444):
                        s += U444[i*4+j][r] * V444[j*4+k][r] * W444[k*4+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_4x4x4():
        # Fallback to schoolbook for 4x4x5
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Extend to 4x4x5
    # Add 16 terms for k=4
    total_rank = rank444 + 16
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy 4x4x4 part
    for row in range(16):
        for r in range(rank444):
            U[row][r] = U444[row][r]
            V[row][r] = V444[row][r]
            W[row][r] = W444[row][r]
    
    # Add schoolbook for k=4
    col = rank444
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    # Final verification
    def verify_full():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_full():
        # Fallback to pure schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
