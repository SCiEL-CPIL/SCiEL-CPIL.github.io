"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal (or near-minimal) decomposition via Kronecker product of
    # a rank-7 Strassen 2x2 decomposition and a rank-27 schoolbook 3x3 decomposition.
    # Total rank = 7 * 27 = 189.
    #
    # Matrix multiplication tensor for 6x6: (n,m,p) = (6,6,6).
    # We view 6 = 2 * 3. The 6x6 matrix multiplication can be seen as
    # the Kronecker product of 2x2 and 3x3 matrix multiplications.
    #
    # Convention:
    #   U shape (n*m, rank) = (36, rank), row index = i*m + j
    #   V shape (m*p, rank) = (36, rank), row index = j*p + k
    #   W shape (p*n, rank) = (36, rank), row index = k*n + i
    #
    # For the 2x2 part (n1=2, m1=2, p1=2):
    #   U1 shape (4, r1), V1 shape (4, r1), W1 shape (4, r1)
    #   Row indices: i1*m1+j1, j1*p1+k1, k1*n1+i1
    #
    # For the 3x3 part (n2=3, m2=3, p2=3):
    #   U2 shape (9, r2), V2 shape (9, r2), W2 shape (9, r2)
    #   Row indices: i2*m2+j2, j2*p2+k2, k2*n2+i2
    #
    # Kronecker product combination:
    #   The 6x6 multiplication tensor T = T2 x T3 (tensor product).
    #   U row index for (i,j) where i = i1*3 + i2, j = j1*3 + j2? 
    #   Wait, need to be careful with the flattening order.
    #
    # Actually, the standard approach: if A is 6x6 and we write A as a 2x2 block 
    # matrix where each block is 3x3, then the multiplication is not simply the 
    # Kronecker product of two independent matrix multiplications. The Kronecker 
    # product approach works when the overall operation is (A1 ⊗ A2)(B1 ⊗ B2) = 
    # (A1B1) ⊗ (A2B2). This means the input matrices must be in Kronecker form.
    #
    # However, the tensor decomposition perspective: the matrix multiplication 
    # tensor for (n1*n2) x (m1*m2) x (p1*p2) is isomorphic to the tensor product 
    # of the (n1,m1,p1) and (n2,m2,p2) tensors. The factors of the tensor product 
    # are Kronecker products of the individual factors, with appropriate index 
    # reordering.
    #
    # Let's define the 2x2 Strassen factors:
    n1, m1, p1 = 2, 2, 2
    # Strassen's 7 products for 2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+b)(e+f)
    # m2 = c(e+f)
    # m3 = a(g-h)
    # m4 = d(e+f-h)  -- wait, standard Strassen is:
    # m1 = (a+b)(e+f)
    # m2 = (c+d)(e+f)
    # m3 = a(g-h)
    # m4 = d(f-g)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(e+g)
    # m7 = (b+d)(f+h)
    # C[0,0] = m1 - m2 + m4 - m7
    # C[0,1] = m3 + m7
    # C[1,0] = m2 - m4 + m6
    # C[1,1] = m1 - m3 + m5 - m6
    
    r1 = 7
    U1 = [[0]*r1 for _ in range(n1*m1)]
    V1 = [[0]*r1 for _ in range(m1*p1)]
    W1 = [[0]*r1 for _ in range(p1*n1)]
    
    # Indexing: U1[i1*m1+j1, col], V1[j1*p1+k1, col], W1[k1*n1+i1, col]
    # A indices: (i1, j1), B indices: (j1, k1), C indices: (k1, i1)
    
    # m1 = (a+b)(e+f): U1 gets a,b; V1 gets e,f
    U1[0][0] = 1; U1[1][0] = 1  # a, b (i1=0,j1=0) and (i1=0,j1=1)
    V1[0][0] = 1; V1[1][0] = 1  # e, f (j1=0,k1=0) and (j1=0,k1=1)
    W1[0][0] = 1  # c[0,0] (k1=0,i1=0)
    W1[1][0] = 1  # c[0,1] (k1=0,i1=1) -- wait, C[0,1] corresponds to k1=0? No.
    # Let me re-examine. C[k1*n1+i1, col] where n1=2.
    # C[0,0] -> k1=0, i1=0 -> index 0
    # C[0,1] -> k1=0, i1=1 -> index 1
    # C[1,0] -> k1=1, i1=0 -> index 2
    # C[1,1] -> k1=1, i1=1 -> index 3
    
    # C[0,0] = m1 - m2 + m4 - m7
    # C[0,1] = m3 + m7
    # C[1,0] = m2 - m4 + m6
    # C[1,1] = m1 - m3 + m5 - m6
    
    # Let me reset and do this carefully.
    U1 = [[0]*r1 for _ in range(4)]
    V1 = [[0]*r1 for _ in range(4)]
    W1 = [[0]*r1 for _ in range(4)]
    
    # m1 = (a+b)(e+f)
    U1[0][0] = 1; U1[1][0] = 1
    V1[0][0] = 1; V1[1][0] = 1
    W1[0][0] = 1  # C[0,0] += m1
    W1[3][0] = 1  # C[1,1] += m1
    
    # m2 = (c+d)(e+f)
    U1[2][1] = 1; U1[3][1] = 1  # c=(i1=1,j1=0), d=(i1=1,j1=1)
    V1[0][1] = 1; V1[1][1] = 1
    W1[0][1] = -1  # C[0,0] -= m2
    W1[2][1] = 1   # C[1,0] += m2
    
    # m3 = a(g-h)
    U1[0][2] = 1  # a
    V1[2][2] = 1; V1[3][2] = -1  # g=(j1=1,k1=0), h=(j1=1,k1=1)
    W1[1][2] = 1  # C[0,1] += m3
    W1[3][2] = -1 # C[1,1] -= m3
    
    # m4 = d(f-g)
    U1[3][3] = 1  # d
    V1[1][3] = 1; V1[2][3] = -1  # f=(j1=0,k1=1), g=(j1=1,k1=0)
    W1[0][3] = 1  # C[0,0] += m4
    W1[2][3] = -1 # C[1,0] -= m4
    
    # m5 = (a+d)(e+h)
    U1[0][4] = 1; U1[3][4] = 1
    V1[0][4] = 1; V1[3][4] = 1  # e=(j1=0,k1=0), h=(j1=1,k1=1)
    W1[3][4] = 1  # C[1,1] += m5
    
    # m6 = (b+c)(e+g)
    U1[1][5] = 1; U1[2][5] = 1
    V1[0][5] = 1; V1[2][5] = 1  # e=(j1=0,k1=0), g=(j1=1,k1=0)
    W1[2][5] = 1  # C[1,0] += m6
    W1[3][5] = -1 # C[1,1] -= m6
    
    # m7 = (b+d)(f+h)
    U1[1][6] = 1; U1[3][6] = 1
    V1[1][6] = 1; V1[3][6] = 1  # f=(j1=0,k1=1), h=(j1=1,k1=1)
    W1[0][6] = -1 # C[0,0] -= m7
    W1[1][6] = 1  # C[0,1] += m7
    
    # Now the 3x3 schoolbook factors:
    n2, m2, p2 = 3, 3, 3
    r2 = n2 * m2 * p2  # 27
    U2 = [[0]*r2 for _ in range(n2*m2)]
    V2 = [[0]*r2 for _ in range(m2*p2)]
    W2 = [[0]*r2 for _ in range(p2*n2)]
    col2 = 0
    for i2 in range(n2):
        for j2 in range(m2):
            for k2 in range(p2):
                U2[i2*m2 + j2][col2] = 1
                V2[j2*p2 + k2][col2] = 1
                W2[k2*n2 + i2][col2] = 1
                col2 += 1
    
    # Now combine via Kronecker product.
    # The 6x6 tensor is T1 ⊗ T2 where T1 is 2x2x2 and T2 is 3x3x3.
    # Rank = r1 * r2 = 7 * 27 = 189.
    rank = r1 * r2
    
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Row indexing for the 6x6 case:
    # U row: i*6 + j where i = i1*3 + i2, j = j1*3 + j2
    #   => row = (i1*3 + i2)*6 + (j1*3 + j2) = i1*18 + i2*6 + j1*3 + j2
    # V row: j*6 + k where j = j1*3 + j2, k = k1*3 + k2
    #   => row = (j1*3 + j2)*6 + (k1*3 + k2) = j1*18 + j2*6 + k1*3 + k2
    # W row: k*6 + i where k = k1*3 + k2, i = i1*3 + i2
    #   => row = (k1*3 + k2)*6 + (i1*3 + i2) = k1*18 + k2*6 + i1*3 + i2
    
    # For each pair (c1, c2) where c1 in [0,7), c2 in [0,27):
    #   rank index = c1 * 27 + c2
    #   For each nonzero entry in U1[i1*2+j1, c1], U2[i2*3+j2, c2]:
    #     U[(i1*3+i2)*6 + (j1*3+j2), c1*27+c2] += U1[i1*2+j1][c1] * U2[i2*3+j2][c2]
    #   Similarly for V and W.
    
    # Build the combined factors
    for c1 in range(r1):
        for c2 in range(r2):
            col = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1*m1 + j1][c1]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2*m2 + j2][c2]
                            if u2_val == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            row = i * m + j  # m = 6
                            U[row][col] += u1_val * u2_val
            
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1*p1 + k1][c1]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2*p2 + k2][c2]
                            if v2_val == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            row = j * p + k  # p = 6
                            V[row][col] += v1_val * v2_val
            
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1*n1 + i1][c1]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2*n2 + i2][c2]
                            if w2_val == 0:
                                continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            row = k * n + i  # n = 6
                            W[row][col] += w1_val * w2_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
