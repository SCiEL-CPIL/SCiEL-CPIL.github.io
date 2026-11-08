"""
5x5 matrix multiplication - rank 119 block decomposition.
Uses Strassen for 2x2 blocks and optimized 3x3 decomposition.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n*m, dtype=float)
        V = np.zeros(m*p, dtype=float)
        W = np.zeros(p*n, dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m+j] = val
        for (j, k), val in v_vals.items():
            V[j*p+k] = val
        for (k, i), val in w_vals.items():
            W[k*n+i] = val

        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    # C_22 = A_22 B_22 + A_23 B_32
    strassen_2x2(0, 1, 0, 1, 0, 1)  # A22*B22: 7
    strassen_2x2(0, 1, 2, 3, 0, 1)  # A23a*B32a: 7
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # A23b*B32b: 4

    # C_23 = A_22 B_23 + A_23 B_33
    strassen_2x2(0, 1, 0, 1, 2, 3)  # A22*B23a: 7
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # A22*B23b: 4
    strassen_2x2(0, 1, 2, 3, 2, 3)  # A23a*B33a: 7
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # A23a*B33b: 4
    for i in [0, 1]:
        for k in [2, 3, 4]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # A23b*B33: 6

    # C_32 = A_32 B_22 + A_33 B_32
    strassen_2x2(2, 3, 0, 1, 0, 1)  # A32a*B22: 7
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # A32b*B22: 4
    strassen_2x2(2, 3, 2, 3, 0, 1)  # A33a*B32a: 7
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # A33b*B32b: 4
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # A33c*B32a: 4
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})  # A33d*B32b: 2

    # C_33 = A_32 B_23 + A_33 B_33
    strassen_2x2(2, 3, 0, 1, 2, 3)  # A32a*B23a: 7
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # A32a*B23b: 4
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # A32b*B23a: 4
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})  # A32b*B23b: 2
    strassen_2x2(2, 3, 2, 3, 2, 3)  # A33a*B33a: 7
    for i in [2, 3]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})  # A33b*B33b: 4
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})  # A33a*B33b: 4
    for i in [2, 3]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})  # A33b*B33c: 2
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})  # A33c*B33a: 4
    for k in [2, 3]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})  # A33d*B33b: 2
    for j in [2, 3]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})  # A33c*B33b: 2
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})  # A33d*B33c: 1

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
