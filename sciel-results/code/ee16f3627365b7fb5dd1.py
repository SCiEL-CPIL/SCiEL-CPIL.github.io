"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using 3x3 schoolbook and 2x2 Strassen
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 factors (size 4x7)
    # Indices: 0:11, 1:12, 2:21, 3:22
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 1, 0, 0, 1, 0],  # B12
        [0, 0, 0, 1, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C11 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],  # C12 (k=1,i=0) -> Wait, W index is k*n+i.
        # For 2x2: k=0,i=0 -> 0. k=0,i=1 -> 1. k=1,i=0 -> 2. k=1,i=1 -> 3.
        # My W2 above was defined by C indices (row, col).
        # C11 is row 0 col 0. In W index k*n+i, if C is n x p (2x2), i is row, k is col.
        # So C11 (i=0,k=0) -> 0.
        # C12 (i=0,k=1) -> 1.
        # C21 (i=1,k=0) -> 2.
        # C22 (i=1,k=1) -> 3.
        # Let's re-verify W2 values based on P definitions.
        # P1: C11 + C22 -> W[0]=1, W[3]=1.
        # P2: C21 -> W[2]=1.
        # P3: C12 + C22 -> W[1]=1, W[3]=1.
        # P4: C11 + C21 -> W[0]=1, W[2]=1.
        # P5: -C11 + C12 -> W[0]=-1, W[1]=1.
        # P6: C22 -> W[3]=1.
        # P7: C11 + C22 -> W[0]=1, W[3]=1.
    ]
    # Correcting W2 based on standard Strassen output mapping to W[k*n+i]
    # W index: k (col of C) * 2 + i (row of C)
    # C11: i=0, k=0 -> 0
    # C12: i=0, k=1 -> 1
    # C21: i=1, k=0 -> 2
    # C22: i=1, k=1 -> 3
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # P1..P7 for C11 (idx 0)
        [0, 0, 1, 0, 1, 0, 0],  # for C12 (idx 1)
        [0, 1, 0, 1, 0, 0, 0],  # for C21 (idx 2)
        [1, 0, 1, 0, 0, 1, 1]   # for C22 (idx 3)
    ]

    # Schoolbook 3x3 factors (size 9x27)
    # We don't need to store them, just iterate
    for r3 in range(27):
        i1 = r3 // 9
        j1 = (r3 % 9) // 3
        k1 = r3 % 3
        
        # U3 index: i1*3 + j1
        # V3 index: j1*3 + k1
        # W3 index: k1*3 + i1
        
        for r2 in range(7):
            r = r3 * 7 + r2
            
            # Iterate over all 2x2 block positions
            for i2 in range(2):
                for j2 in range(2):
                    # U index in 6x6: i = 2*i1 + i2, j = 2*j1 + j2
                    i = 2 * i1 + i2
                    j = 2 * j1 + j2
                    u_idx = i * m + j
                    
                    # U2 index: i2*2 + j2
                    u2_idx = i2 * 2 + j2
                    val_u = U3_val(i1, j1, r3) * U2[u2_idx][r2]
                    if val_u != 0:
                        U[u_idx][r] = val_u

                    # V index in 6x6: j = 2*j1 + j2, k = 2*k1 + k2
                    for k2 in range(2):
                        k = 2 * k1 + k2
                        v_idx = j * p + k
                        v2_idx = j2 * 2 + k2
                        val_v = V3_val(j1, k1, r3) * V2[v2_idx][r2]
                        if val_v != 0:
                            V[v_idx][r] = val_v
                        
                        # W index in 6x6: k = 2*k1 + k2, i = 2*i1 + i2
                        w_idx = k * n + i
                        w2_idx = k2 * 2 + i2
                        val_w = W3_val(k1, i1, r3) * W2[w2_idx][r2]
                        if val_w != 0:
                            W[w_idx][r] = val_w

    return {"U": U, "V": V, "W": W}

def U3_val(i, j, r3):
    # Schoolbook 3x3: r3 = i*9 + j*3 + k
    # U is 1 if r3 corresponds to this i,j
    return 1 if (r3 // 9 == i and (r3 % 9) // 3 == j) else 0

def V3_val(j, k, r3):
    return 1 if ((r3 % 9) // 3 == j and r3 % 3 == k) else 0

def W3_val(k, i, r3):
    return 1 if (r3 % 3 == k and r3 // 9 == i) else 0

# EVOLVE-BLOCK-END
