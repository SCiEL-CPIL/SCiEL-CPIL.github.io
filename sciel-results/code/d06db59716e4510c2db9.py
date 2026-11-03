import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build the 10x3x2 tensor (transposed view)
    # T_trans[k*3+j, j*2+i, i*10+k] = 1
    # This is equivalent to: 10x3 * 3x2 -> 10x2
    # U_trans: (10*3, rank), V_trans: (3*2, rank), W_trans: (2*10, rank)
    # Decompose as:
    #   j=0: 2x1x2 block (k=0,1; i=0,1) -> rank 2
    #   j=1,2: 2x2x2 blocks (k=0,1; i=0,1) for 5 k-pairs -> rank 7 each
    # Total rank = 2 + 5*7 = 37

    rank = 37
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    col = 0

    # Part A: j=0 block (2x1x2): A[:,0] * B[0,:] -> C[:,0]
    # In transposed view: U_trans rows (k*3+0), V_trans row (0*2+i), W_trans col (i*10+k)
    # Original: U[i*3+0, col], V[0*10+k, col], W[k*2+i, col]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0  # i*3+0
            v_idx = 0 * p + k  # 0*10+k
            w_idx = k * n + i  # k*2+i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    # col should be 2 here

    # Part B: j=1,2 blocks (5 x 2x2x2 Strassen)
    # For each k-pair (k0, k1), the 2x2x2 block involves:
    #   A submatrix: rows i=0,1; cols j=1,2  -> U indices: 0*3+1=1, 0*3+2=2, 1*3+1=4, 1*3+2=5
    #   B submatrix: rows j=1,2; cols k=k0,k1 -> V indices: 1*10+k0=10+k0, 1*10+k1=10+k1, 2*10+k0=20+k0, 2*10+k1=20+k1
    #   C submatrix: rows k=k0,k1; cols i=0,1 -> W indices: k0*2+0, k0*2+1, k1*2+0, k1*2+1

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # U indices (A submatrix entries)
        u_11 = 0 * m + 1  # A[0,1]
        u_12 = 0 * m + 2  # A[0,2]
        u_21 = 1 * m + 1  # A[1,1]
        u_22 = 1 * m + 2  # A[1,2]

        # V indices (B submatrix entries)
        v_11 = 1 * p + k0  # B[1,k0]
        v_12 = 1 * p + k1  # B[1,k1]
        v_21 = 2 * p + k0  # B[2,k0]
        v_22 = 2 * p + k1  # B[2,k1]

        # W indices (C submatrix entries)
        w_00 = k0 * n + 0  # C[k0,0]
        w_01 = k0 * n + 1  # C[k0,1]
        w_10 = k1 * n + 0  # C[k1,0]
        w_11 = k1 * n + 1  # C[k1,1]

        # Strassen's 7 products for 2x2x2:
        # M1 = (a11 + a22)(b11 + b22)
        U[u_11, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1; V[v_22, col] += 1
        W[w_00, col] += 1; W[w_11, col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_01, col] += 1; W[w_11, col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11, col] += 1
        V[v_12, col] += 1; V[v_22, col] += -1
        W[w_10, col] += 1; W[w_11, col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22, col] += 1
        V[v_21, col] += 1; V[v_11, col] += -1
        W[w_00, col] += 1; W[w_01, col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11, col] += 1; U[u_12, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += -1; W[w_10, col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21, col] += 1; U[u_11, col] += -1
        V[v_11, col] += 1; V[v_12, col] += 1
        W[w_11, col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12, col] += 1; U[u_22, col] += -1
        V[v_21, col] += 1; V[v_22, col] += 1
        W[w_00, col] += 1
        col += 1

    # Verify the decomposition
    def verify(U, V, W, r, tol=1e-9):
        # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        err_max = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j, :]
                    v_row = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err_max = max(err_max, abs(val - 1.0))
        return err_max < tol, err_max

    valid, err = verify(U, V, W, rank)

    if not valid:
        # Fallback: return rank-55 solution if rank-37 fails verification
        rank_fb = 55
        U_fb = np.zeros((nm, rank_fb))
        V_fb = np.zeros((mp, rank_fb))
        W_fb = np.zeros((pn, rank_fb))
        col = 0
        for i in range(n):
            for k in range(p):
                U_fb[i * m + 0, col] = 1
                V_fb[0 * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            u_11 = 0 * m + 1; u_12 = 0 * m + 2
            u_21 = 1 * m + 1; u_22 = 1 * m + 2
            v_11 = 1 * p + k0; v_12 = 1 * p + k1
            v_21 = 2 * p + k0; v_22 = 2 * p + k1
            w_00 = k0 * n + 0; w_01 = k1 * n + 0
            w_10 = k0 * n + 1; w_11 = k1 * n + 1

            U_fb[u_11, col] += 1; U_fb[u_22, col] += 1
            V_fb[v_11, col] += 1; V_fb[v_22, col] += 1
            W_fb[w_00, col] += 1; W_fb[w_11, col] += 1
            col += 1

            U_fb[u_21, col] += 1; U_fb[u_22, col] += 1
            V_fb[v_11, col] += 1
            W_fb[w_10, col] += 1; W_fb[w_11, col] += -1
            col += 1

            U_fb[u_11, col] += 1
            V_fb[v_12, col] += 1; V_fb[v_22, col] += -1
            W_fb[w_01, col] += 1; W_fb[w_11, col] += 1
            col += 1

            U_fb[u_22, col] += 1
            V_fb[v_21, col] += 1; V_fb[v_11, col] += -1
            W_fb[w_00, col] += 1; W_fb[w_10, col] += 1
            col += 1

            U_fb[u_11, col] += 1; U_fb[u_12, col] += 1
            V_fb[v_22, col] += 1
            W_fb[w_00, col] += -1; W_fb[w_01, col] += 1
            col += 1

            U_fb[u_21, col] += 1; U_fb[u_11, col] += -1
            V_fb[v_11, col] += 1; V_fb[v_12, col] += 1
            W_fb[w_11, col] += 1
            col += 1

            U_fb[u_12, col] += 1; U_fb[u_22, col] += -1
            V_fb[v_21, col] += 1; V_fb[v_22, col] += 1
            W_fb[w_00, col] += 1
            col += 1

        U, V, W, rank = U_fb, V_fb, W_fb, rank_fb

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
