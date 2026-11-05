import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 115

    # Build the target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target[(a, b, c)] = 1.0

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(U_vec, V_vec, W_vec):
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)

    # Strassen 2x2x2 rank-7 decomposition
    # U: [a00, a01, a10, a11], V: [b00, b01, b10, b11], W: [c00, c01, c10, c11]
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 1, 0]),
        ([0, 0, 0, 1], [1, 1, 0, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([1, 0, -1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 1]),
    ]

    # Part 1: Rows {0,1}, j in {0,1}, k in pairs -> 2x2x2 Strassen (4 k-pairs)
    for k_pair in range(4):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        for (u_c, v_c, w_c) in strassen_terms:
            U_vec = [0.0] * (n*m)
            U_vec[0] = float(u_c[0])   # i=0,j=0
            U_vec[1] = float(u_c[1])   # i=0,j=1
            U_vec[4] = float(u_c[2])   # i=1,j=0
            U_vec[5] = float(u_c[3])   # i=1,j=1

            V_vec = [0.0] * (m*p)
            V_vec[k0] = float(v_c[0])       # j=0,k=k0
            V_vec[k1] = float(v_c[1])       # j=0,k=k1
            V_vec[8 + k0] = float(v_c[2])   # j=1,k=k0
            V_vec[8 + k1] = float(v_c[3])   # j=1,k=k1

            W_vec = [0.0] * (p*n)
            W_vec[3*k0] = float(w_c[0])       # k=k0,i=0
            W_vec[3*k1] = float(w_c[1])       # k=k1,i=0
            W_vec[3*k0 + 1] = float(w_c[2])   # k=k0,i=1
            W_vec[3*k1 + 1] = float(w_c[3])   # k=k1,i=1

            add_rank1(U_vec, V_vec, W_vec)

    # Part 2: Rows {0,1}, j in {2,3}, k in pairs -> 2x2x2 Strassen (4 k-pairs)
    for k_pair in range(4):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        for (u_c, v_c, w_c) in strassen_terms:
            U_vec = [0.0] * (n*m)
            U_vec[2] = float(u_c[0])   # i=0,j=2
            U_vec[3] = float(u_c[1])   # i=0,j=3
            U_vec[6] = float(u_c[2])   # i=1,j=2
            U_vec[7] = float(u_c[3])   # i=1,j=3

            V_vec = [0.0] * (m*p)
            V_vec[16 + k0] = float(v_c[0])   # j=2,k=k0
            V_vec[16 + k1] = float(v_c[1])   # j=2,k=k1
            V_vec[24 + k0] = float(v_c[2])   # j=3,k=k0
            V_vec[24 + k1] = float(v_c[3])   # j=3,k=k1

            W_vec = [0.0] * (p*n)
            W_vec[3*k0] = float(w_c[0])
            W_vec[3*k1] = float(w_c[1])
            W_vec[3*k0 + 1] = float(w_c[2])
            W_vec[3*k1 + 1] = float(w_c[3])

            add_rank1(U_vec, V_vec, W_vec)

    # Part 3: Row {2}, all j in {0,1,2,3}, k in {0..7} -> schoolbook 1x4x8
    for j in range(4):
        for k in range(8):
            U_vec = [0.0] * (n*m)
            U_vec[2*4 + j] = 1.0

            V_vec = [0.0] * (m*p)
            V_vec[j*8 + k] = 1.0

            W_vec = [0.0] * (p*n)
            W_vec[k*3 + 2] = 1.0

            add_rank1(U_vec, V_vec, W_vec)

    rank = len(U_cols)

    # Convert to arrays
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T

    # Verify correctness
    error = 0.0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = np.dot(U[a] * V[b], W[c])
                true_val = target.get((a, b, c), 0.0)
                error += (val - true_val)**2

    if error > 1e-6:
        # Fallback to schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_vec = [0.0] * (n*m)
                    U_vec[i*m+j] = 1.0
                    V_vec = [0.0] * (m*p)
                    V_vec[j*p+k] = 1.0
                    W_vec = [0.0] * (p*n)
                    W_vec[k*n+i] = 1.0
                    add_rank1(U_vec, V_vec, W_vec)
        rank = len(U_cols)
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
