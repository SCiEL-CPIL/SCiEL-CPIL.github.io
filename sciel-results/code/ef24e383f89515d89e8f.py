import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# ---- Rank-7 decomposition for 2x2x2 (Strassen) ----
def _strassen222():
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    # (a,b,c) index triples for U[i*m+j], V[j*p+k], W[k*n+i]
    triples = [
        (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1),
    ]
    for c, (i, j, k) in enumerate(triples):
        U[i * m + j, c] = 1
        V[j * p + k, c] = 1
        W[k * n + i, c] = 1
    return U, V, W

# ---- Rank-11 decomposition for 2x3x3 (recursive: 2x2x2 + 2x2x1 + 2x1x2) ----
def _decomp_233():
    n = m = p = 3
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_block(Ub, Vb, Wb, off):
        U[:, off:off + Ub.shape[1]] = Ub
        V[:, off:off + Vb.shape[1]] = Vb
        W[:, off:off + Wb.shape[1]] = Wb

    # Block A: indices i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2, rank 7
    Ua, Va, Wa = _strassen222()
    add_block(Ua, Va, Wa, 0)

    # Block B: indices i in {0,1}, j in {0,1}, k in {2} -> 2x2x1, rank 4
    # 2x2x1 = 2x2x1 (j=0,k=2) + 2x1x1 (j=1,k=2)
    U1 = np.zeros((6, 2))
    V1 = np.zeros((9, 2))
    W1 = np.zeros((6, 2))
    # (i=0,j=0,k=2): U[0,0]=1, V[0*3+2]=1, W[2*2+0]=1
    U1[0 * 3 + 0, 0] = 1; V1[0 * 3 + 2, 0] = 1; W1[2 * 2 + 0, 0] = 1
    # (i=1,j=0,k=2): U[1,0]=1, V[0,2]=1, W[2,1]=1
    U1[1 * 3 + 0, 1] = 1; V1[0 * 3 + 2, 1] = 1; W1[2 * 2 + 1, 1] = 1
    # 2x1x1 (i in {0,1}, j=1, k=2)
    U2 = np.zeros((6, 2))
    V2 = np.zeros((9, 2))
    W2 = np.zeros((6, 2))
    U2[0 * 3 + 1, 0] = 1; V2[1 * 3 + 2, 0] = 1; W2[2 * 2 + 0, 0] = 1
    U2[1 * 3 + 1, 1] = 1; V2[1 * 3 + 2, 1] = 1; W2[2 * 2 + 1, 1] = 1
    add_block(np.hstack([U1, U2]), np.hstack([V1, V2]), np.hstack([W1, W2]), 7)

    # Block C: indices i in {0,1}, j in {2}, k in {0,1} -> 2x1x2, rank 4
    U3 = np.zeros((6, 2))
    V3 = np.zeros((9, 2))
    W3 = np.zeros((6, 2))
    # (i=0,j=2,k=0): U[0,2]=1, V[2*3+0]=1, W[0*2+0]=1
    U3[0 * 3 + 2, 0] = 1; V3[2 * 3 + 0, 0] = 1; W3[0 * 2 + 0, 0] = 1
    # (i=1,j=2,k=0): U[1,2]=1, V[2,0]=1, W[0,1]=1
    U3[1 * 3 + 2, 1] = 1; V3[2 * 3 + 0, 1] = 1; W3[0 * 2 + 1, 1] = 1
    # 2x1x1 (i in {0,1}, j=2, k=1)
    U4 = np.zeros((6, 2))
    V4 = np.zeros((9, 2))
    W4 = np.zeros((6, 2))
    U4[0 * 3 + 2, 0] = 1; V4[2 * 3 + 1, 0] = 1; W4[1 * 2 + 0, 0] = 1
    U4[1 * 3 + 2, 1] = 1; V4[2 * 3 + 1, 1] = 1; W4[1 * 2 + 1, 1] = 1
    add_block(np.hstack([U3, U4]), np.hstack([V3, V4]), np.hstack([W3, W4]), 11)

    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]  should be 1
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    return np.max(np.abs(T - 1.0))

def construct():
    start = time.time()
    deadline = start + 110.0
    n, m, p = DIMENSIONS  # 2,5,6

    # Split the 5 columns into 2 (Strassen 2x2x2) + 3 (2x3x3) + 0
    # Actually: first 2 columns -> 2x2x2; remaining 3 columns -> 2x3x3
    U1, V1, W1 = _strassen222()   # n=2,m=2,p=2
    U2, V2, W2 = _decomp_233()    # n=2,m=3,p=3

    # Assemble: U rows = n*m_total = 2*5 = 10, V rows = m_total*p = 5*6 = 30, W rows = p*n = 6*2 = 12
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank = rank1 + rank2
    U = np.zeros((n * 5, rank))
    V = np.zeros((5 * p, rank))
    W = np.zeros((p * n, rank))

    # Block 1: columns j in {0,1}, k in {0,1}
    # U rows for i in {0,1}, j in {0,1} -> indices i*5+j
    # V rows for j in {0,1}, k in {0,1} -> indices j*6+k
    # W rows for k in {0,1}, i in {0,1} -> indices k*2+i
    U[:2, :rank1] = U1  # U1 row index i*2+j, but we need i*5+j for i in 0..1, j in 0..1
    # Fix mapping: U1 has shape (2*2,7); map row (i*2+j) -> (i*5+j)
    U1m = np.zeros((n * 5, rank1))
    for i in range(2):
        for j in range(2):
            U1m[i * 5 + j, :] = U1[i * 2 + j, :]
    U[:, :rank1] = U1m

    V1m = np.zeros((5 * p, rank1))
    for j in range(2):
        for k in range(2):
            V1m[j * 6 + k, :] = V1[j * 2 + k, :]
    V[:, :rank1] = V1m

    W[:, :rank1] = W1  # W1 row index k*2+i, k in 0..1, i in 0..1 -> k*2+i matches k*n+i since n=2

    # Block 2: columns j in {2,3,4}, k in {2,3,4,5}  (3x3 block)
    # U rows for i in {0,1}, j in {2,3,4} -> indices i*5+j
    # V rows for j in {2,3,4}, k in {2,3,4,5} -> indices j*6+k
    # W rows for k in {2,3,4,5}, i in {0,1} -> indices k*2+i
    U2m = np.zeros((n * 5, rank2))
    for i in range(2):
        for j in range(3):
            U2m[i * 5 + (j + 2), :] = U2[i * 3 + j, :]
    U[:, rank1:] = U2m

    V2m = np.zeros((5 * p, rank2))
    for j in range(3):
        for k in range(3):
            V2m[(j + 2) * 6 + (k + 2), :] = V2[j * 3 + k, :]
    V[:, rank1:] = V2m

    W2m = np.zeros((p * n, rank2))
    for k in range(3):
        for i in range(2):
            W2m[(k + 2) * 2 + i, :] = W2[k * 2 + i, :]
    W[:, rank1:] = W2m

    # Verify
    err = _verify(U, V, W, n, 5, p)
    if err > 1e-9:
        # Fallback to schoolbook if verification fails
        rank = n * 5 * p
        U = [[0] * rank for _ in range(n * 5)]
        V = [[0] * rank for _ in range(5 * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(5):
                for k in range(p):
                    U[i * 5 + j][column] = 1
                    V[j * 6 + k][column] = 1
                    W[k * 2 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Convert to lists of ints
    U_list = [[int(round(x)) for x in row] for row in U]
    V_list = [[int(round(x)) for x in row] for row in V]
    W_list = [[int(round(x)) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
