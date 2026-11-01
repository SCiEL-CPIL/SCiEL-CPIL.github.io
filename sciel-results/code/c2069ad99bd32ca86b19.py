import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen 2x2x2 core
    n1, m1, p1 = 2, 2, 2
    r1 = 7
    U1 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V1 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W1 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    # Schoolbook 1x2x3 core
    n2, m2, p2 = 1, 2, 3
    r2 = 6
    U2 = [[1, 0], [0, 1]] # 2x6? No, 2 rows, 6 cols. Identity padded.
    # U2[j2, c2] = 1 if j2 == c2
    U2 = [[0]*6 for _ in range(2)]
    for j in range(2):
        U2[j][j] = 1
        
    # V2[j2*p2 + k2, c2] = 1 if j2 == c2
    V2 = [[0]*6 for _ in range(6)]
    for j in range(2):
        for k in range(3):
            V2[j*3 + k][j] = 1
            
    # W2[k2, c2] = 1
    W2 = [[1]*6 for _ in range(3)]

    n, m, p = DIMENSIONS
    rank = r1 * r2
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            
            # U construction
            # i = i1 (since n2=1)
            # j = j1*m2 + j2
            for i1 in range(n1):
                for j1 in range(m1):
                    for j2 in range(m2):
                        i = i1 * n2 + 0 # i1
                        j = j1 * m2 + j2
                        u_val = U1[i1*m1 + j1][c1] * U2[j2][c2]
                        if u_val != 0:
                            U[i*m + j][c] = u_val

            # V construction
            # j = j1*m2 + j2
            # k = k1*p2 + k2
            for j1 in range(m1):
                for j2 in range(m2):
                    for k1 in range(p1):
                        for k2 in range(p2):
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            v_val = V1[j1*p1 + k1][c1] * V2[j2*p2 + k2][c2]
                            if v_val != 0:
                                V[j*p + k][c] = v_val

            # W construction
            # k = k1*p2 + k2
            # i = i1 (since n2=1)
            for k1 in range(p1):
                for k2 in range(p2):
                    for i1 in range(n1):
                        k = k1 * p2 + k2
                        i = i1 * n2 + 0
                        w_val = W1[k1*n1 + i1][c1] * W2[k2][c2]
                        if w_val != 0:
                            W[k*n + i][c] = w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
