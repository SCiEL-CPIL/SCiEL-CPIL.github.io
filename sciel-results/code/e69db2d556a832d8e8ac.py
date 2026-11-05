def construct():
    n, m, p = 6, 6, 6
    
    # Rank 7 Strassen for 2x2
    R2 = 7
    U2 = [[0]*R2 for _ in range(4)]
    V2 = [[0]*R2 for _ in range(4)]
    W2 = [[0]*R2 for _ in range(4)]
    
    # r=0: P1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1: P2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1 # k=0, i=1 -> idx 1
    
    # r=2: P3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1 # k=1, i=0 -> idx 2
    W2[3][2] = -1 # k=1, i=1 -> idx 3
    
    # r=3: P4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[1][3] = 1 # k=0, i=1
    W2[0][3] = -1 # k=0, i=0
    
    # r=4: P5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[2][4] = 1 # k=1, i=0
    
    # r=5: P6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # k=1, i=1
    
    # r=6: P7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1 # k=0, i=0

    # Rank 27 Schoolbook for 3x3
    R3 = 27
    U3 = [[0]*R3 for _ in range(9)]
    V3 = [[0]*R3 for _ in range(9)]
    W3 = [[0]*R3 for _ in range(9)]
    
    col3 = 0
    for i2 in range(3):
        for j2 in range(3):
            for k2 in range(3):
                U3[i2*3+j2][col3] = 1
                V3[j2*3+k2][col3] = 1
                W3[k2*3+i2][col3] = 1
                col3 += 1

    TotalRank = R2 * R3
    U = [[0]*TotalRank for _ in range(36)]
    V = [[0]*TotalRank for _ in range(36)]
    W = [[0]*TotalRank for _ in range(36)]
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(3):
                    for j2 in range(3):
                        for k2 in range(3):
                            # Indices in 6x6
                            i = i1*3 + i2
                            j = j1*3 + j2
                            k = k1*3 + k2
                            
                            u_idx = i*6 + j
                            v_idx = j*6 + k
                            w_idx = k*6 + i
                            
                            # Iterate over rank terms
                            # c = c2 * R3 + c3
                            for c2 in range(R2):
                                for c3 in range(R3):
                                    c = c2*R3 + c3
                                    
                                    u_val = U2[i1*2+j1][c2] * U3[i2*3+j2][c3]
                                    v_val = V2[j1*2+k1][c2] * V3[j2*3+k2][c3]
                                    w_val = W2[k1*2+i1][c2] * W3[k2*3+i2][c3]
                                    
                                    if u_val != 0:
                                        U[u_idx][c] += u_val
                                    if v_val != 0:
                                        V[v_idx][c] += v_val
                                    if w_val != 0:
                                        W[w_idx][c] += w_val

    return {"U": U, "V": V, "W": W}
