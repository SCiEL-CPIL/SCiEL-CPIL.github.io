"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Sub-problem 1: 2x2x2 (Strassen)
    n1, m1, p1 = 2, 2, 2
    R1 = 7
    U1 = [[0]*R1 for _ in range(n1*m1)]
    V1 = [[0]*R1 for _ in range(m1*p1)]
    W1 = [[0]*R1 for _ in range(p1*n1)]
    
    # Strassen coefficients
    # l=0
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    # l=1
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1
    # l=2
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1
    # l=3
    U1[3][3] = 1
    V1[0][3] = -1; V1[2][3] = 1
    W1[0][3] = 1; W1[1][3] = 1
    # l=4
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[2][4] = 1; W1[3][4] = 1
    # l=5
    U1[0][5] = -1; U1[2][5] = 1
    V1[0][5] = 1; V1[1][5] = 1
    W1[0][5] = -1
    # l=6
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = -1; W1[3][6] = 1

    # Sub-problem 2: 1x2x3 (Trivial)
    n2, m2, p2 = 1, 2, 3
    R2 = n2 * m2 * p2
    U2 = [[0]*R2 for _ in range(n2*m2)]
    V2 = [[0]*R2 for _ in range(m2*p2)]
    W2 = [[0]*R2 for _ in range(p2*n2)]
    
    col = 0
    for j2 in range(m2):
        for k2 in range(p2):
            U2[j2][col] = 1
            V2[j2*p2 + k2][col] = 1
            W2[k2*n2][col] = 1
            col += 1

    # Combine
    R = R1 * R2
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for r1 in range(R1):
                    u1_val = U1[i1*m1 + j1][r1]
                    v1_val = V1[j1*p1 + k1][r1]
                    w1_val = W1[k1*n1 + i1][r1]
                    
                    if u1_val == 0 and v1_val == 0 and w1_val == 0:
                        continue
                    
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                for r2 in range(R2):
                                    u2_val = U2[i2*m2 + j2][r2]
                                    v2_val = V2[j2*p2 + k2][r2]
                                    w2_val = W2[k2*n2 + i2][r2]
                                    
                                    if u2_val == 0 and v2_val == 0 and w2_val == 0:
                                        continue
                                    
                                    r = r1 * R2 + r2
                                    
                                    # Indices
                                    i = i1 * n2 + i2
                                    j = j1 * m2 + j2
                                    k = k1 * p2 + k2
                                    
                                    U_idx = i * m + j
                                    V_idx = j * p + k
                                    W_idx = k * n + i
                                    
                                    U[U_idx][r] += u1_val * u2_val
                                    V[V_idx][r] += v1_val * v2_val
                                    W[W_idx][r] += w1_val * w2_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
