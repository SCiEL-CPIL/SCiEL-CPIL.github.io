import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_blocks = p // 2  # 6 blocks of 2 columns each

    # Strassen 7-product decomposition for 2x2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[C00,C01],[C10,C11]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6

    # U coefficients for each Strassen term (A-side): index is A[i*m+j] = A[i,j]
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    U_strassen = [
        {0: 1, 3: 1},       # M1 = (a+d)(e+h)
        {2: 1, 3: 1},       # M2 = (c+d)e
        {0: 1},             # M3 = a(f-h)
        {3: 1},             # M4 = d(g-e)
        {0: 1, 1: 1},       # M5 = (a+b)h
        {0: -1, 2: 1},      # M6 = (c-a)(e+f)
        {1: 1, 3: -1},      # M7 = (b-d)(g+h)
    ]

    # V coefficients for each Strassen term (B-side, local indices)
    # B indices: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
    V_strassen = [
        {0: 1, 3: 1},       # M1 = (a+d)(e+h)
        {0: 1},             # M2 = (c+d)e
        {1: 1, 3: -1},      # M3 = a(f-h)
        {2: 1, 0: -1},      # M4 = d(g-e)
        {3: 1},             # M5 = (a+b)h
        {0: 1, 1: 1},       # M6 = (c-a)(e+f)
        {2: 1, 3: 1},       # M7 = (b-d)(g+h)
    ]

    # W coefficients for each Strassen term (C-side, local indices)
    # C indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    W_strassen = [
        {0: 1, 3: 1},       # C00=M1+M4-M5+M7, C11=M1-M2+M3+M6
        {2: 1, 3: -1},      # C10=M2+M4, C11=M1-M2+M3+M6
        {1: 1, 3: 1},       # C01=M3+M5, C11=M1-M2+M3+M6
        {0: 1, 2: 1},       # C00=M1+M4-M5+M7, C10=M2+M4
        {0: -1, 1: 1},      # C00=M1+M4-M5+M7, C01=M3+M5
        {3: 1},             # C11=M1-M2+M3+M6
        {0: 1},             # C00=M1+M4-M5+M7
    ]

    # Build the rank-42 Strassen decomposition
    rank = num_blocks * 7  # 42

    U = [[0] * rank for _ in range(n * m)]  # (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (24, 42)

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s

            # U: A-side, same for all blocks
            for a_idx, val in U_strassen[s].items():
                U[a_idx][col] = val

            # V: B-side, local index maps to global
            # Local B index: 0=B[0,2t], 1=B[0,2t+1], 2=B[1,2t], 3=B[1,2t+1]
            # Global V row index: j*p + k where j is row of B, k is column
            for local_b, val in V_strassen[s].items():
                if local_b == 0:
                    global_b = 0 * p + 2 * t      # B[0, 2t]
                elif local_b == 1:
                    global_b = 0 * p + 2 * t + 1  # B[0, 2t+1]
                elif local_b == 2:
                    global_b = 1 * p + 2 * t      # B[1, 2t]
                else:
                    global_b = 1 * p + 2 * t + 1  # B[1, 2t+1]
                V[global_b][col] = val

            # W: C-side, local index maps to global
            # Local C index: 0=C[0,2t], 1=C[0,2t+1], 2=C[1,2t], 3=C[1,2t+1]
            # Global W row index: k*n + i where k is column of C, i is row
            for local_c, val in W_strassen[s].items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0      # C[0, 2t]
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0  # C[0, 2t+1]
                elif local_c == 2:
                    global_w = (2 * t) * n + 1      # C[1, 2t]
                else:
                    global_w = (2 * t + 1) * n + 1  # C[1, 2t+1]
                W[global_w][col] += val

    # Verify the decomposition is exact
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((n, m))
    B_test = rng.standard_normal((m, p))
    C_expected = A_test @ B_test

    C_computed = np.zeros((n, p))
    for r in range(rank):
        a_lin = U_arr[:, r]
        b_lin = V_arr[:, r]
        w_lin = W_arr[:, r]

        A_lin = a_lin.reshape(n, m)
        B_lin = b_lin.reshape(m, p)

        prod_a = np.sum(A_lin * A_test)
        prod_b = np.sum(B_lin * B_test)

        for k in range(p):
            for i in range(n):
                C_computed[i, k] += prod_a * prod_b * w_lin[k * n + i]

    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to naive rank-48 decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1

    return {"U": U, "V": V, "W": W}
