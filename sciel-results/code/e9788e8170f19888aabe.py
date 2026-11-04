"""Matrix multiplication 6x6x6 via Kronecker product of Winograd 3x3 and Strassen 2x2."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _winograd_3x3():
    """Winograd's rank-8 decomposition for 3x3 matrix multiplication.
    Returns U (9x8), V (9x8), W (9x8).
    Convention: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    U = [[0]*8 for _ in range(9)]
    V = [[0]*8 for _ in range(9)]
    W = [[0]*8 for _ in range(9)]

    # Product 0: (A00+A10+A20)*(B00+B01+B02)
    U[0][0] = 1; U[3][0] = 1; U[6][0] = 1
    V[0][0] = 1; V[1][0] = 1; V[2][0] = 1

    # Product 1: (A00-A10+A20)*(B00-B01+B02)
    U[0][1] = 1; U[3][1] = -1; U[6][1] = 1
    V[0][1] = 1; V[1][1] = -1; V[2][1] = 1

    # Product 2: (A01+A21)*(B10+B11)
    U[1][2] = 1; U[7][2] = 1
    V[3][2] = 1; V[4][2] = 1

    # Product 3: (A01-A21)*(B10-B11)
    U[1][3] = 1; U[7][3] = -1
    V[3][3] = 1; V[4][3] = -1

    # Product 4: (A02+A22)*(B20+B21)
    U[2][4] = 1; U[8][4] = 1
    V[6][4] = 1; V[7][4] = 1

    # Product 5: (A02-A22)*(B20-B21)
    U[2][5] = 1; U[8][5] = -1
    V[6][5] = 1; V[7][5] = -1

    # Product 6: (A10+A11+A12)*(B10+B11+B12)
    U[3][6] = 1; U[4][6] = 1; U[5][6] = 1
    V[3][6] = 1; V[4][6] = 1; V[5][6] = 1

    # Product 7: (A10-A11+A12)*(B10-B11+B12)
    U[3][7] = 1; U[4][7] = -1; U[5][7] = 1
    V[3][7] = 1; V[4][7] = -1; V[5][7] = 1

    # W[k*3+i, c]: coefficient for C[i,k]
    # C00 = M0+M1+M2+M3+M4+M5  -> k=0,i=0 -> idx 0
    W[0][0]=1; W[0][1]=1; W[0][2]=1; W[0][3]=1; W[0][4]=1; W[0][5]=1
    # C01 = M2-M3+M4-M5  -> k=1,i=0 -> idx 3
    W[3][2]=1; W[3][3]=-1; W[3][4]=1; W[3][5]=-1
    # C02 = M4-M5  -> k=2,i=0 -> idx 6
    W[6][4]=1; W[6][5]=-1
    # C10 = M0-M1+M6-M7  -> k=0,i=1 -> idx 1
    W[1][0]=1; W[1][1]=-1; W[1][6]=1; W[1][7]=-1
    # C11 = M6+M7  -> k=1,i=1 -> idx 4
    W[4][6]=1; W[4][7]=1
    # C12 = M6-M7  -> k=2,i=1 -> idx 7
    W[7][6]=1; W[7][7]=-1
    # C20 = M0+M1-M2-M3+M4+M5  -> k=0,i=2 -> idx 2
    W[2][0]=1; W[2][1]=1; W[2][2]=-1; W[2][3]=-1; W[2][4]=1; W[2][5]=1
    # C21 = M2-M3-M4+M5  -> k=1,i=2 -> idx 5
    W[5][2]=1; W[5][3]=-1; W[5][4]=-1; W[5][5]=1
    # C22 = M4+M5  -> k=2,i=2 -> idx 8
    W[8][4]=1; W[8][5]=1

    return U, V, W


def _strassen_2x2():
    """Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M0 = (A00+A11)*(B00+B11)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1

    # M1 = (A10+A11)*B00
    U[2][1]=1; U[3][1]=1
    V[0][1]=1

    # M2 = A00*(B01-B11)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1

    # M3 = A11*(B10-B00)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1

    # M4 = (A00+A01)*B11
    U[0][4]=1; U[1][4]=1
    V[3][4]=1

    # M5 = (A10-A00)*(B00+B01)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1

    # M6 = (A01-A11)*(B10+B11)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1

    # W[k*2+i, c]:
    # C00 = M0+M3-M4-M6  -> k=0,i=0 -> idx 0
    W[0][0]=1; W[0][3]=1; W[0][4]=-1; W[0][6]=-1
    # C01 = M2+M4  -> k=1,i=0 -> idx 2
    W[2][2]=1; W[2][4]=1
    # C10 = M1+M3  -> k=0,i=1 -> idx 1
    W[1][1]=1; W[1][3]=1
    # C11 = M0-M1+M2+M5  -> k=1,i=1 -> idx 3
    W[3][0]=1; W[3][1]=-1; W[3][2]=1; W[3][5]=1

    return U, V, W


def _kron_decompose(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Kronecker product of two decompositions."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    rank = r1 * r2

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for c1 in range(r1):
                    u1 = U1[i1*m1+j1][c1]
                    v1 = V1[j1*p1+k1][c1]
                    w1 = W1[k1*n1+i1][c1]
                    if u1 == 0 or v1 == 0 or w1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                for c2 in range(r2):
                                    u2 = U2[i2*m2+j2][c2]
                                    v2 = V2[j2*p2+k2][c2]
                                    w2 = W2[k2*n2+i2][c2]
                                    if u2 == 0 or v2 == 0 or w2 == 0:
                                        continue
                                    ii = i1*n2 + i2
                                    jj = j1*m2 + j2
                                    kk = k1*p2 + k2
                                    cc = c1*r2 + c2
                                    U[ii*m+jj][cc] += u1 * u2
                                    V[jj*p+kk][cc] += v1 * v2
                                    W[kk*n+ii][cc] += w1 * w2

    return U, V, W


def construct():
    U3, V3, W3 = _winograd_3x3()  # 9x8
    U2, V2, W2 = _strassen_2x2()  # 4x7

    # 6x6x6 = (3x3) ⊗ (2x2), rank = 8*7 = 56
    U, V, W = _kron_decompose(U3, V3, W3, 3, 3, 3, U2, V2, W2, 2, 2, 2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
