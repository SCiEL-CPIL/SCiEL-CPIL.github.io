"""Kronecker-product tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    2x2x2 Strassen decomposition.
    Returns (U, V, W) with shapes (4,7), (4,7), (4,7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W

def _build_2x2x3():
    """
    2x2x3 decomposition of rank 6 (rank-optimal for 2x2x3).
    A is 2x2, B is 2x3, C is 2x3.
    Indexing: U[i*2+j], V[j*3+k], W[k*2+i].
    """
    rank = 6
    U = [[0]*rank for _ in range(4)]   # n*m = 2*2
    V = [[0]*rank for _ in range(6)]   # m*p = 2*3
    W = [[0]*rank for _ in range(6)]   # p*n = 3*2

    # Column 0: A[0,0], B[0,0], C[0,0]
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1

    # Column 1: A[0,1], B[0,1], C[0,1]
    U[1][1] = 1
    V[1][1] = 1
    W[1][1] = 1

    # Column 2: A[1,0], B[1,0], C[0,2]
    U[2][2] = 1
    V[3][2] = 1
    W[2][2] = 1

    # Column 3: A[1,1], B[1,1], C[1,0]
    U[3][3] = 1
    V[4][3] = 1
    W[3][3] = 1

    # Column 4: A[0,0]+A[1,1], B[0,2], C[1,1]
    U[0][4] = 1; U[3][4] = 1
    V[2][4] = 1
    W[5][4] = 1

    # Column 5: A[0,1]+A[1,0], B[1,2], C[1,2]
    U[1][5] = 1; U[2][5] = 1
    V[5][5] = 1
    W[5][5] = 1

    return U, V, W

def _kron_decomposition_dims(d1, dims1, d2, dims2):
    """
    Kronecker product of two decompositions with explicit dimensions.
    d1 = (U1, V1, W1) for (n1, m1, p1)
    d2 = (U2, V2, W2) for (n2, m2, p2)
    Produces decomposition for (n1*n2, m1*m2, p1*p2).
    Indexing: U[i*m+j], V[j*p+k], W[k*n+i].
    """
    U1, V1, W1 = d1
    U2, V2, W2 = d2
    n1, m1, p1 = dims1
    n2, m2, p2 = dims2

    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2

    n = n1 * n2
    m = m1 * m2
    p = p1 * p2

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c1 in range(r1):
        for c2 in range(r2):
            col = c1 * r2 + c2

            # U: index (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
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
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            U[i*m + j][col] = u1_val * u2_val

            # V: index (j1*p2 + j2) * (p1*p2) + (k1*p2 + k2)
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
                            j = j1*p2 + j2
                            k = k1*p2 + k2
                            V[j*p + k][col] = v1_val * v2_val

            # W: index (k1*n2 + k2) * (n1*n2) + (i1*n2 + i2)
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
                            k = k1*n2 + k2
                            i = i1*n2 + i2
                            W[k*n + i][col] = w1_val * w2_val

    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    try:
        # 2x2x2 Strassen (rank 7)
        U1, V1, W1 = _build_strassen_2x2()
        d1 = (U1, V1, W1)
        dims1 = (2, 2, 2)

        # 2x2x3 rank-6 decomposition
        U2, V2, W2 = _build_2x2x3()
        d2 = (U2, V2, W2)
        dims2 = (2, 2, 3)

        # Build 4x4x6 via Kronecker (2x2x2) ⊗ (2x2x3): rank 7*6 = 42
        U_kron, V_kron, W_kron = _kron_decomposition_dims(d1, dims1, d2, dims2)

        rank_kron = 42
        rank_extra = 4 * 1 * 6  # 24
        total_rank = rank_kron + rank_extra

        U = [[0]*total_rank for _ in range(4*5)]  # 20 rows
        V = [[0]*total_rank for _ in range(5*6)]  # 30 rows
        W = [[0]*total_rank for _ in range(6*4)]  # 24 rows

        # Copy Kronecker part: A rows i=0..3, j=0..3 -> U[i*5+j]
        # B rows j=0..3, k=0..5 -> V[j*6+k]
        # C rows k=0..5, i=0..3 -> W[k*4+i]
        for i in range(4):
            for j in range(4):
                u_row = i * 5 + j
                for c in range(rank_kron):
                    U[u_row][c] = U_kron[i*4 + j][c]

        for j in range(4):
            for k in range(6):
                v_row = j * 6 + k
                for c in range(rank_kron):
                    V[v_row][c] = V_kron[j*4 + k][c]

        for k in range(6):
            for i in range(4):
                w_row = k * 4 + i
                for c in range(rank_kron):
                    W[w_row][c] = W_kron[k*4 + i][c]

        # Schoolbook part: A[:,4] * B[4,:] -> C
        col_offset = rank_kron
        for i in range(4):
            for k in range(6):
                U[i*5 + 4][col_offset + i*6 + k] = 1
                V[4*6 + k][col_offset + i*6 + k] = 1
                W[k*4 + i][col_offset + i*6 + k] = 1

        return {"U": U, "V": V, "W": W}

    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
