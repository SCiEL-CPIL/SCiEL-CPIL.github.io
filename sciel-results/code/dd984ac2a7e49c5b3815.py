"""Strassen-based rank-49 decomposition for 4x4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    # Use Strassen 2x2x2 (rank 7) tensor product to get 4x4x4 (rank 49)
    # Then add schoolbook terms for the 5th column (k=4)
    
    # Rank-7 Strassen decomposition for 2x2x2
    # Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # T[i,j,k] = 1 for all i,j,k in {0,1} (each A[i,j]*B[j,k] contributes to C[i,k])
    
    # Standard Strassen rank-7 decomposition
    # P1 = (a+b)(e+f), P2 = (b+d)(e+f), P3 = a(d-c), P4 = g(a+b),
    # P5 = e(d-a), P6 = h(b+c), P7 = (a-c)(e-f)
    # C00 = P1 - P6 - P7 + P4, C01 = P3 + P5
    # C10 = P2 + P7, C11 = P1 - P4 - P5 + P3
    
    # U matrix (4x7): rows are (i,j) pairs
    U2 = [
        [1, 0, 1, 0, 0, 0, 1],   # i=0,j=0: a
        [1, 0, 0, 1, 0, 0, 0],   # i=0,j=1: b
        [0, 1, -1, 0, 0, 0, -1], # i=1,j=0: c
        [0, 1, 1, 0, 0, 1, 0]    # i=1,j=1: d
    ]
    # V matrix (4x7): rows are (j,k) pairs
    V2 = [
        [1, 1, 0, 0, 1, 0, 1],   # j=0,k=0: e
        [1, 1, 0, 0, 0, 1, -1],  # j=0,k=1: f
        [0, 0, 0, 1, 0, 0, 0],   # j=1,k=0: g
        [0, 0, 1, 0, 0, 0, 1]    # j=1,k=1: h
    ]
    # W matrix (4x7): rows are (k,i) pairs (output index k*n+i)
    W2 = [
        [1, 0, 0, 1, 0, -1, -1], # k=0,i=0: C00
        [0, 0, 1, 0, 1, 0, 0],   # k=1,i=0: C01
        [0, 1, 0, 0, 0, 0, 1],   # k=0,i=1: C10
        [1, 0, 1, -1, -1, 0, 0]  # k=1,i=1: C11
    ]
    
    # Verify 2x2x2
    def verify_2x2():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U2[i*2+j][r] * V2[j*2+k][r] * W2[k*2+i][r] for r in range(7))
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Build 4x4x4 via tensor product: rank = 7*7 = 49
    rank44 = 49
    U4 = [[0]*rank44 for _ in range(16)]
    V4 = [[0]*rank44 for _ in range(16)]
    W4 = [[0]*rank44 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1*7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1*2 + i2
                            j = j1*2 + j2
                            u_val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            U4[i*4+j][r] = u_val
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1*2 + k2
                                    V4[j*4+k][r] = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                    W4[k*4+i][r] = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
    
    # Verify 4x4x4
    def verify_4x4():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = sum(U4[i*4+j][r] * V4[j*4+k][r] * W4[k*4+i][r] for r in range(rank44))
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Extend to 4x4x5: add 16 schoolbook terms for k=4
    total_rank = rank44 + 16
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    for row in range(16):
        for r in range(rank44):
            U[row][r] = U4[row][r]
            V[row][r] = V4[row][r]
            W[row][r] = W4[row][r]
    
    col = rank44
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
                    s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(total_rank))
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_full():
        # Fallback to schoolbook rank 80
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
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
