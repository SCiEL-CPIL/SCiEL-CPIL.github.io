import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0  # Safety margin before 150s kill

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
        """Strassen decomposition for 2x2 * 2x2 -> 2x2 (7 terms)"""
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    def schoolbook_1x2_2x1(i0, j0, j1, k0):
        """1x2 * 2x1 -> 1x1 (2 terms)"""
        # C[i0,k0] = A[i0,j0]B[j0,k0] + A[i0,j1]B[j1,k0]
        add_term({(i0,j0):1}, {(j0,k0):1}, {(k0,i0):1})
        add_term({(i0,j1):1}, {(j1,k0):1}, {(k0,i0):1})

    def schoolbook_1x2_2x2(i0, j0, j1, k0, k1):
        """1x2 * 2x2 -> 1x2 (4 terms)"""
        for k in [k0, k1]:
            add_term({(i0,j0):1}, {(j0,k):1}, {(k,i0):1})
            add_term({(i0,j1):1}, {(j1,k):1}, {(k,i0):1})

    def schoolbook_2x1_1x2(i0, i1, j0, k0, k1):
        """2x1 * 1x2 -> 2x2 (4 terms)"""
        for i in [i0, i1]:
            add_term({(i,j0):1}, {(j0,k0):1}, {(k0,i):1})
            add_term({(i,j0):1}, {(j0,k1):1}, {(k1,i):1})

    def schoolbook_2x2_2x1(i0, i1, j0, j1, k0):
        """2x2 * 2x1 -> 2x1 (4 terms)"""
        for i in [i0, i1]:
            add_term({(i,j0):1}, {(j0,k0):1}, {(k0,i):1})
            add_term({(i,j1):1}, {(j1,k0):1}, {(k0,i):1})

    # Split indices:
    # Block 1: 0
    # Block 2: 1, 2
    # Block 3: 3, 4

    # C_11 (1x1) = A_11(1x1) * B_11(1x1)
    add_term({(0,0):1}, {(0,0):1}, {(0,0):1})

    # C_12 (1x2) = A_11(1x1) * B_12(1x2) + A_12(1x2) * B_22(2x2) + A_13(1x2) * B_32(2x2)
    # A_11 * B_12: 1x1 * 1x2 (2 terms)
    for k in [1, 2]:
        add_term({(0,0):1}, {(0,k):1}, {(k,0):1})
    # A_12 * B_22: 1x2 * 2x2 (4 terms)
    schoolbook_1x2_2x2(0, 1, 2, 1, 2)
    # A_13 * B_32: 1x2 * 2x2 (4 terms)
    schoolbook_1x2_2x2(0, 3, 4, 1, 2)

    # C_13 (1x2) = A_11(1x1) * B_13(1x2) + A_12(1x2) * B_23(2x2) + A_13(1x2) * B_33(2x2)
    # A_11 * B_13: 1x1 * 1x2 (2 terms)
    for k in [3, 4]:
        add_term({(0,0):1}, {(0,k):1}, {(k,0):1})
    # A_12 * B_23: 1x2 * 2x2 (4 terms)
    schoolbook_1x2_2x2(0, 1, 2, 3, 4)
    # A_13 * B_33: 1x2 * 2x2 (4 terms)
    schoolbook_1x2_2x2(0, 3, 4, 3, 4)

    # C_21 (2x1) = A_21(2x1) * B_11(1x1) + A_22(2x2) * B_21(2x1) + A_23(2x2) * B_31(2x1)
    # A_21 * B_11: 2x1 * 1x1 (2 terms)
    for i in [1, 2]:
        add_term({(i,0):1}, {(0,0):1}, {(0,i):1})
    # A_22 * B_21: 2x2 * 2x1 (4 terms)
    schoolbook_2x2_2x1(1, 2, 1, 2, 1)
    # A_23 * B_31: 2x2 * 2x1 (4 terms)
    schoolbook_2x2_2x1(1, 2, 3, 4, 1)

    # C_22 (2x2) = A_21(2x1) * B_12(1x2) + A_22(2x2) * B_22(2x2) + A_23(2x2) * B_32(2x2)
    # A_21 * B_12: 2x1 * 1x2 (4 terms)
    schoolbook_2x1_1x2(1, 2, 0, 1, 2)
    # A_22 * B_22: 2x2 * 2x2 (7 terms)
    strassen_2x2(1, 2, 1, 2, 1, 2)
    # A_23 * B_32: 2x2 * 2x2 (7 terms)
    strassen_2x2(1, 2, 3, 4, 1, 2)

    # C_23 (2x2) = A_21(2x1) * B_13(1x2) + A_22(2x2) * B_23(2x2) + A_23(2x2) * B_33(2x2)
    # A_21 * B_13: 2x1 * 1x2 (4 terms)
    schoolbook_2x1_1x2(1, 2, 0, 3, 4)
    # A_22 * B_23: 2x2 * 2x2 (7 terms)
    strassen_2x2(1, 2, 1, 2, 3, 4)
    # A_23 * B_33: 2x2 * 2x2 (7 terms)
    strassen_2x2(1, 2, 3, 4, 3, 4)

    # C_31 (2x1) = A_31(2x1) * B_11(1x1) + A_32(2x2) * B_21(2x1) + A_33(2x2) * B_31(2x1)
    # A_31 * B_11: 2x1 * 1x1 (2 terms)
    for i in [3, 4]:
        add_term({(i,0):1}, {(0,0):1}, {(0,i):1})
    # A_32 * B_21: 2x2 * 2x1 (4 terms)
    schoolbook_2x2_2x1(3, 4, 1, 2, 1)
    # A_33 * B_31: 2x2 * 2x1 (4 terms)
    schoolbook_2x2_2x1(3, 4, 3, 4, 1)

    # C_32 (2x2) = A_31(2x1) * B_12(1x2) + A_32(2x2) * B_22(2x2) + A_33(2x2) * B_32(2x2)
    # A_31 * B_12: 2x1 * 1x2 (4 terms)
    schoolbook_2x1_1x2(3, 4, 0, 1, 2)
    # A_32 * B_22: 2x2 * 2x2 (7 terms)
    strassen_2x2(3, 4, 1, 2, 1, 2)
    # A_33 * B_32: 2x2 * 2x2 (7 terms)
    strassen_2x2(3, 4, 3, 4, 1, 2)

    # C_33 (2x2) = A_31(2x1) * B_13(1x2) + A_32(2x2) * B_23(2x2) + A_33(2x2) * B_33(2x2)
    # A_31 * B_13: 2x1 * 1x2 (4 terms)
    schoolbook_2x1_1x2(3, 4, 0, 3, 4)
    # A_32 * B_23: 2x2 * 2x2 (7 terms)
    strassen_2x2(3, 4, 1, 2, 3, 4)
    # A_33 * B_33: 2x2 * 2x2 (7 terms)
    strassen_2x2(3, 4, 3, 4, 3, 4)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
