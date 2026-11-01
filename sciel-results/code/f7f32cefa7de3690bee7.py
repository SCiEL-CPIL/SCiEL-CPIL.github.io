import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 2, 2, 15
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # We use a block decomposition.
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) using Strassen (rank 7 each)
    # 1 single column (14) using trivial (rank 4)
    # Total rank = 7 * 7 + 4 = 53

    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53

    # A indices in U:
    # 0: A00, 1: A01, 2: A10, 3: A11

    # B indices in V:
    # j*p + k. j=0 -> 0..14, j=1 -> 15..29

    # C indices in W:
    # k*n + i. k=0 -> 0,1; k=1 -> 2,3; ... k=14 -> 28,29

    # Helper to set Strassen block for column pair (k0, k1)
    # Standard Strassen for C = A*B where A,B are 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a-d)*(g+h)
    # M2 = (a+b)*g
    # M3 = d*(f-h)
    # M4 = a*e
    # M5 = (b+d)*(e+h)
    # M6 = (a-b)*f
    # M7 = (c+a)*(h-e)
    # C11 = M4 + M5 - M1 - M7
    # C12 = M2 + M6
    # C21 = M3 + M5
    # C22 = M1 + M4 - M2 + M7

    def add_strassen_block(base_r, k0, k1):
        # k0, k1 are the column indices in B (and C)
        # e = B[0,k0], f = B[0,k1], g = B[1,k0], h = B[1,k1]
        # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
        
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6

        # M1 = (a-d)*(g+h)
        # U: a=1, d=-1
        U[0][r1] = 1
        U[3][r1] = -1
        # V: g=1, h=1
        V[1*p + k0][r1] = 1
        V[1*p + k1][r1] = 1
        # W: C11 gets -M1, C22 gets +M1
        # C11 is at index k0*n + 0 = 2*k0
        # C22 is at index k1*n + 1 = 2*k1 + 1
        W[2*k0][r1] = -1
        W[2*k1 + 1][r1] = 1

        # M2 = (a+b)*g
        # U: a=1, b=1
        U[0][r2] = 1
        U[1][r2] = 1
        # V: g=1
        V[1*p + k0][r2] = 1
        # W: C12 gets +M2, C22 gets -M2
        # C12 is at index k1*n + 0 = 2*k1
        # C22 is at index k1*n + 1 = 2*k1 + 1
        W[2*k1][r2] = 1
        W[2*k1 + 1][r2] = -1

        # M3 = d*(f-h)
        # U: d=1
        U[3][r3] = 1
        # V: f=1, h=-1
        V[0*p + k1][r3] = 1
        V[1*p + k1][r3] = -1
        # W: C21 gets +M3
        # C21 is at index k0*n + 1 = 2*k0 + 1
        W[2*k0 + 1][r3] = 1

        # M4 = a*e
        # U: a=1
        U[0][r4] = 1
        # V: e=1
        V[0*p + k0][r4] = 1
        # W: C11 gets +M4, C22 gets +M4
        W[2*k0][r4] = 1
        W[2*k1 + 1][r4] = 1

        # M5 = (b+d)*(e+h)
        # U: b=1, d=1
        U[1][r5] = 1
        U[3][r5] = 1
        # V: e=1, h=1
        V[0*p + k0][r5] = 1
        V[1*p + k1][r5] = 1
        # W: C11 gets +M5, C21 gets +M5
        W[2*k0][r5] = 1
        W[2*k0 + 1][r5] = 1

        # M6 = (a-b)*f
        # U: a=1, b=-1
        U[0][r6] = 1
        U[1][r6] = -1
        # V: f=1
        V[0*p + k1][r6] = 1
        # W: C12 gets +M6
        W[2*k1][r6] = 1

        # M7 = (c+a)*(h-e)
        # U: c=1, a=1
        U[2][r7] = 1
        U[0][r7] = 1
        # V: h=1, e=-1
        V[1*p + k1][r7] = 1
        V[0*p + k0][r7] = -1
        # W: C11 gets -M7, C22 gets +M7
        W[2*k0][r7] = -1
        W[2*k1 + 1][r7] = 1

    # Add 7 Strassen blocks for pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)

    # Add trivial block for column 14
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3

    # Q1 = A00 * B014 -> C014
    # C014 is at index 14*2 + 0 = 28
    U[0][r1] = 1
    V[0*p + 14][r1] = 1
    W[2*14][r1] = 1

    # Q2 = A01 * B114 -> C014
    # C014 is at index 14*2 + 0 = 28
    U[1][r2] = 1
    V[1*p + 14][r2] = 1
    W[2*14][r2] = 1

    # Q3 = A10 * B014 -> C114
    # C114 is at index 14*2 + 1 = 29
    U[2][r3] = 1
    V[0*p + 14][r3] = 1
    W[2*14 + 1][r3] = 1

    # Q4 = A11 * B114 -> C114
    # C114 is at index 14*2 + 1 = 29
    U[3][r4] = 1
    V[1*p + 14][r4] = 1
    W[2*14 + 1][r4] = 1

    # Verify numerically if time permits
    if time.time() < deadline - 5:
        T = np.zeros((4, 30, 30), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(15):
                    T[i*2+j, j*15+k, i*15+k] = 1

        # Reconstruct
        T_recon = np.zeros((4, 30, 30), dtype=float)
        for r in range(rank):
            ua = np.array([U[a][r] for a in range(4)])
            vb = np.array([V[b][r] for b in range(30)])
            wc = np.array([W[c][r] for c in range(30)])
            T_recon += np.einsum('a,b,c->abc', ua, vb, wc)

        max_err = np.max(np.abs(T - T_recon))
        if max_err > 1e-10:
            # Fallback to schoolbook
            rank = 60
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(30)]
            W = [[0]*rank for _ in range(30)]
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(15):
                        U[i*2+j][col] = 1
                        V[j*15+k][col] = 1
                        W[k*2+i][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}
