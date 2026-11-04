DIMENSIONS = [4, 4, 6]

def construct():
    # Strassen decomposition for 2x2x2 (rank 7)
    U1 = [
        [1,  0, 1,  0, 1,  0, 0],
        [0,  0, 0,  0, 0,  1, 0],
        [0,  1, 0,  0, 0,  0, 1],
        [1,  1, 0,  1, 0,  1, 0],
    ]
    V1 = [
        [1,  1, 0,  0, 0,  0, 1],
        [0,  0, 1,  0, 0,  1, 0],
        [0,  0, 0,  1, 0,  0, 1],
        [1,  0,-1,  1, 1, -1, 0],
    ]
    W1 = [
        [1, -1, 0, -1, 0,  0, 1],
        [1,  1, 0,  0, 0,  0, 0],
        [0,  0, 1,  1, 0,  0, 0],
        [1,  0,-1,  0, 1, -1, 0],
    ]
    
    # Schoolbook for 2x2x3 (rank 12)
    n2, m2, p2 = 2, 2, 3
    r2 = 12
    U2 = [[0]*r2 for _ in range(4)]
    V2 = [[0]*r2 for _ in range(6)]
    W2 = [[0]*r2 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U2[i*2+j][col] = 1
                V2[j*3+k][col] = 1
                W2[k*2+i][col] = 1
                col += 1
    
    n1, m1, p1 = 2, 2, 2
    r1 = 7
    n, m, p = 4, 4, 6
    rank = r1 * r2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for a in range(r1):
        for b in range(r2):
            r = a * r2 + b
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            U[i*m + j][r] = U1[i1*m1+j1][a] * U2[i2*m2+j2][b]
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = j1*m2 + j2
                            k = k1*p2 + k2
                            V[j*p + k][r] = V1[j1*p1+k1][a] * V2[j2*p2+k2][b]
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = k1*p2 + k2
                            i = i1*n2 + i2
                            W[k*n + i][r] = W1[k1*n1+i1][a] * W2[k2*n2+i2][b]
    
    return {"U": U, "V": V, "W": W}
