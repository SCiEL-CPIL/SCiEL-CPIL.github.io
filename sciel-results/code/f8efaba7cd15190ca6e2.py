import time
import numpy as np

DIMENSIONS = [2, 3, 6]


def _strassen222():
    """Return (U,V,W) each 4x7 for 2x2x2 Strassen. U idx a=i*2+j, V idx b=j*2+k, W idx c=k*2+i."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1   # m0 = a00*b00 -> c00
    U[1, 1] = 1; V[2, 1] = 1; W[1, 1] = 1   # m1 = a01*b10 -> c01
    U[2, 2] = 1; V[0, 2] = 1; W[2, 2] = 1   # m2 = a10*b01 -> c10
    U[3, 3] = 1; V[2, 3] = 1; W[3, 3] = 1   # m3 = a11*b11 -> c11
    U[0, 4] = 1; U[3, 4] = 1
    V[2, 4] = 1; V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = 1; W[1, 4] = -1; W[2, 4] = -1
    U[0, 5] = 1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1; W[1, 5] = -1; W[3, 5] = -1
    U[2, 6] = 1; U[3, 6] = 1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1; W[2, 6] = -1; W[3, 6] = -1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 on support, 0 off support."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    # For each r, contribution tensor T_r = outer(U_r, V_r, W_r)
    # sum over r of T_r should equal target tensor T where T[i*m+j, j*p+k, k*n+i]=1
    acc = np.zeros((n * m, m * p, p * n))
    for r in range(R):
        acc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    err = np.max(np.abs(acc - target))
    return err < tol


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS  # 2, 3, 6

    U, V, W = _strassen222()

    # ---- Block 1: A1 (2x2) * B1 (2x6), three independent 2x2x2 Strassen (cols k=0..5) ----
    # A1 row-index i in {0,1}, col j in {0,1}  -> A global index i*m + j = i*3 + j
    # B1 row-index j in {0,1}, col k in {0..5} -> B global index j*p + k = j*6 + k
    # C col-index k in {0..5}, row i in {0,1}  -> W global index k*n + i = k*2 + i
    b1_U = np.zeros((n * m, 21))
    b1_V = np.zeros((m * p, 21))
    b1_W = np.zeros((p * n, 21))
    for c in range(3):
        for r in range(7):
            col = c * 7 + r
            # U
            for a in range(4):
                b1_U[a, col] = U[a, r]
            # V
            for b in range(4):
                b1_V[b, col] = V[b, r]
            # W: shift k by c*2
            for w in range(4):
                b1_W[w + c * 2, col] = W[w, r]

    # ---- Block 2: A2 (2x1) * B2 (1x6) = (a0*b0 + a0*b1 + a1*b2) * 1^T ----
    # A2: i in {0,1}, j=2 -> A index i*3 + 2
    # B2: j=2, k in {0..5} -> B index 2*6 + k = 12 + k
    # C: k in {0..5}, i in {0,1} -> W index k*2 + i
    b2_U = np.zeros((n * m, 12))
    b2_V = np.zeros((m * p, 12))
    b2_W = np.zeros((p * n, 12))
    for k in range(6):
        # term a0*b_k
        col = k * 2
        b2_U[0 * 3 + 2, col] = 1   # A[0,2]
        b2_V[2 * 6 + k, col] = 1   # B[2,k]
        b2_W[k * 2 + 0, col] = 1   # C[k,0]
        b2_W[k * 2 + 1, col] = 1   # C[k,1]
        # term a1*b_k
        col = k * 2 + 1
        b2_U[1 * 3 + 2, col] = 1   # A[1,2]
        b2_V[2 * 6 + k, col] = 1   # B[2,k]
        b2_W[k * 2 + 0, col] = 1
        b2_W[k * 2 + 1, col] = 1

    U_full = np.concatenate([b1_U, b2_U], axis=1)
    V_full = np.concatenate([b1_V, b2_V], axis=1)
    W_full = np.concatenate([b1_W, b2_W], axis=1)

    # Verify exactness
    if not _verify(U_full, V_full, W_full, n, m, p):
        # Fallback: guaranteed-valid schoolbook rank 36
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

    if time.time() > deadline:
        # Shouldn't happen; return schoolbook for safety
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

    # Convert to integer lists (entries are integers)
    U_list = [[int(round(x)) for x in row] for row in U_full]
    V_list = [[int(round(x)) for x in row] for row in V_full]
    W_list = [[int(round(x)) for x in row] for row in W_full]
    return {"U": U_list, "V": V_list, "W": W_list}
