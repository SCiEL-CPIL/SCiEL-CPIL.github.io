def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    def _strassen_2x2():
        terms = []
        terms.append(((1,0,1,0), (1,1,0,0), (1,0,0,1)))
        terms.append(((0,0,1,1), (1,0,0,0), (0,0,1,1)))
        terms.append(((1,0,0,0), (0,1,-1,0), (0,1,0,-1)))
        terms.append(((0,0,0,1), (0,-1,1,0), (-1,0,1,0)))
        terms.append(((1,1,0,0), (0,0,0,1), (1,1,0,0)))
        terms.append(((0,0,1,-1), (1,1,0,0), (0,0,0,-1)))
        terms.append(((0,1,0,-1), (0,0,1,1), (-1,0,0,0)))
        return terms
    
    s2 = _strassen_2x2()
    
    A_coeff = [
        (1,0,0,1), (0,0,1,1), (1,0,0,0), (0,0,0,1),
        (1,1,0,0), (-1,0,1,0), (0,1,0,-1),
    ]
    B_coeff = [
        (1,0,0,1), (1,0,0,0), (0,1,0,-1), (-1,0,1,0),
        (0,0,0,1), (1,1,0,0), (0,0,1,1),
    ]
    C_coeff = [
        (1,0,0,1), (0,0,1,1), (0,1,0,-1), (-1,0,1,0),
        (1,1,0,0), (0,0,0,-1), (-1,0,0,0),
    ]
    
    # Generate 49 terms for M_{4,4,4}
    s44_terms = []
    for t in range(7):
        for s in range(7):
            u2, v2, w2 = s2[s]
            u4 = [0]*16
            v4 = [0]*16
            w4 = [0]*16
            for bi in range(2):
                for bj in range(2):
                    idx = bi*2+bj
                    u_val = u2[idx] * A_coeff[t][idx]
                    v_val = v2[idx] * B_coeff[t][idx]
                    w_val = w2[idx] * C_coeff[t][idx]
                    for i in range(2):
                        for j in range(2):
                            pos = (bi*2+i)*4 + (bj*2+j)
                            u4[pos] = u_val
                            v4[pos] = v_val
                            w4[pos] = w_val
            s44_terms.append((u4, v4, w4))
    
    rank = 49 + 49 + 16
    U = [[0]*rank for _ in range(n*m)]  # 16 x rank
    V = [[0]*rank for _ in range(m*p)]  # 36 x rank
    W = [[0]*rank for _ in range(p*n)]  # 36 x rank
    
    col = 0
    # C1 = A * B1 (cols 0-3 of B)
    for u4, v4, w4 in s44_terms:
        for i in range(16):
            U[i][col] = u4[i]
            V[i][col] = v4[i]  # B cols 0-3
            W[i][col] = w4[i]  # C cols 0-3
        col += 1
    
    # C2 = A * B2 (cols 4-7 of B)
    for u4, v4, w4 in s44_terms:
        for i in range(16):
            U[i][col] = u4[i]
            # B cols 4-7: index i in v4 corresponds to B[j,k] where k = i%4, j = i//4
            # We need to map to B[j, k+4]
            b_idx = i  # 0..15, representing B[j,k] for j,k in 0..3
            # In V (4x9), B[j,k] is at position j*9+k
            # b_idx = j*4+k, so j = b_idx//4, k = b_idx%4
            j = b_idx // 4
            k = b_idx % 4
            v_pos = j*9 + (k+4)
            V[v_pos][col] = v4[i]
            # Similarly for W
            c_idx = i
            cj = c_idx // 4
            ck = c_idx % 4
            w_pos = ck*4 + cj  # W[k*n+i] where k=ck, i=cj
            W[w_pos][col] = w4[i]
        col += 1
