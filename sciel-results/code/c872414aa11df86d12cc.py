import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 8

    # Build the target tensor for verification
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1

    # Strategy: split A (2x3) into A[:,0] (2x1) and A[:,1:3] (2x2).
    # The 8 columns of B split into 4 independent (k0,k1) pairs.
    # For each pair, C[:, k0:k1] = A[:,0]*B[0,k0:k1] + A[:,1:3]*B[1:3,k0:k1]
    #   outer part  : rank 4
    #   Strassen    : rank 7
    #   block total : rank 10
    # Total rank = 4 * 10 = 40.

    rank_total = 40
    U = np.zeros((n*m, rank_total), dtype=np.int64)
    V = np.zeros((m*p, rank_total), dtype=np.int64)
    W = np.zeros((p*n, rank_total), dtype=np.int64)

    col_offset = 0

    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1

        # A indices: a11=A[0,1]=1, a12=A[0,2]=2, a21=A[1,1]=4, a22=A[1,2]=5
        # B indices: b11=B[1,k0]=8+k0, b12=B[1,k1]=8+k1,
        #            b21=B[2,k0]=16+k0, b22=B[2,k1]=16+k1
        # C indices: C[0,k0]=k0*2+0, C[0,k1]=k1*2+0,
        #            C[1,k0]=k0*2+1, C[1,k1]=k1*2+1
        c11 = k0*2 + 0
        c12 = k1*2 + 0
        c21 = k0*2 + 1
        c22 = k1*2 + 1

        # --- Outer product part: A[:,0] * B[0,k0:k1] (rank 4) ---
        # A[0,0]=0, A[1,0]=3 ; B[0,k0]=k0, B[0,k1]=k1
        U[0, col_offset] = 1; V[k0, col_offset] = 1; W[c11, col_offset] = 1; col_offset += 1
        U[0, col_offset] = 1; V[k1, col_offset] = 1; W[c12, col_offset] = 1; col_offset += 1
        U[3, col_offset] = 1; V[k0, col_offset] = 1; W[c21, col_offset] = 1; col_offset += 1
        U[3, col_offset] = 1; V[k1, col_offset] = 1; W[c22, col_offset] = 1; col_offset += 1

        # --- Strassen part: A[:,1:3] * B[1:3,k0:k1] (rank 7) ---
        # p1 = (a11+a22)(b11+b22)
        U[1, col_offset] = 1; U[5, col_offset] = 1
        V[8+k0, col_offset] = 1; V[16+k1, col_offset] = 1
        W[c11, col_offset] = 1; W[c12, col_offset] = -1; W[c21, col_offset] = -1; W[c22, col_offset] = 1
        col_offset += 1
        # p2 = (a21+a22)(b11+b12)
        U[4, col_offset] = 1; U[5, col_offset] = 1
        V[8+k0, col_offset] = 1; V[8+k1, col_offset] = 1
        W[c12, col_offset] = -1; W[c21, col_offset] = 1; W[c22, col_offset] = -1
        col_offset += 1
        # p3 = a11(b21-b11)
        U[1, col_offset] = 1
        V[16+k0, col_offset] = 1; V[8+k0, col_offset] = -1
        W[c11, col_offset] = 1; W[c21, col_offset] = -1
        col_offset += 1
        # p4 = a22(b22-b12)
        U[5, col_offset] = 1
        V[16+k1, col_offset] = 1; V[8+k1, col_offset] = -1
        W[c11, col_offset] = -1; W[c12, col_offset] = 1; W[c22, col_offset] = -1
        col_offset += 1
        # p5 = (a11+a12)(b12+b22)
        U[1, col_offset] = 1; U[2, col_offset] = 1
        V[8+k1, col_offset] = 1; V[16+k1, col_offset] = 1
        W[c12, col_offset] = 1; W[c21, col_offset] = -1
        col_offset += 1
        # p6 = (a21-a11)(b11+b21)
        U[4, col_offset] = 1; U[1, col_offset] = -1
        V[8+k0, col_offset] = 1; V[16+k0, col_offset] = 1
        W[c12, col_offset] = -1; W[c21, col_offset] = 1
        col_offset += 1
        # p7 = (a22-a12)(b12+b22)
        U[5, col_offset] = 1; U[2, col_offset] = -1
        V[8+k1, col_offset] = 1; V[16+k1, col_offset] = 1
        W[c11, col_offset] = -1; W[c22, col_offset] = 1
        col_offset += 1

    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.array_equal(T_check, T_exp)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook (rank 48)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.int64)
    V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.int64)
    W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
