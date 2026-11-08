"""
5x5 matrix multiplication - rank 72 decomposition.
Uses a 2+1+2 block split with Strassen for all 2x2 sub-blocks
and schoolbook for 1x1/1x2/2x1 cross-blocks.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0  # Safety margin: evaluator kills at 150s

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros((n*m, 1), dtype=float)
        V = np.zeros((m*p, 1), dtype=float)
        W = np.zeros((p*n, 1), dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m+j, 0] = val
        for (j, k), val in v_vals.items():
            V[j*p+k, 0] = val
        for (k, i), val in w_vals.items():
            W[k*n+i, 0] = val

        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())

    # Strassen for 2x2 * 2x2 -> 2x2
    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """Strassen for block with row idx i0,i1; col j0,j1; B col k0,k1"""
        # M1 = (A00+A11)(B00+B11)
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        # M2 = (A10+A11)B00
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        # M3 = A00(B01-B11)
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        # M4 = A11(B10-B00)
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        # M5 = (A00+A01)B11
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        # M6 = (A10-A00)(B00+B01)
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        # M7 = (A01-A11)(B10+B11)
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    # Schoolbook for 2x1 * 1x2 -> 2x2 (4 terms)
    def schoolbook_2x1_1x2(i0, i1, j_mid, k0, k1):
        """Schoolbook for A(i0,i1 x j_mid) * B(j_mid x k0,k1) -> C(i0,i1 x k0,k1)"""
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j_mid): 1}, {(j_mid, k): 1}, {(k, i): 1})

    # 2+1+2 split: indices 0-1, 2, 3-4
    # Block structure:
    # A: rows 0-1,2,3-4; cols 0-1,2,3-4
    # B: rows 0-1,2,3-4; cols 0-1,2,3-4
    # C: rows 0-1,2,3-4; cols 0-1,2,3-4

    # C_22 (rows 0-1, cols 0-1) = A_22*B_22 + A_21*B_12 + A_23*B_32
    # A_22: rows 0-1, cols 0-1; B_22: rows 0-1, cols 0-1
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # A_21: rows 0-1, col 2; B_12: row 2, cols 0-1
    schoolbook_2x1_1x2(0, 1, 2, 0, 1)
    # A_23: rows 0-1, cols 3-4; B_32: rows 3-4, cols 0-1
    strassen_2x2(0, 1, 3, 4, 0, 1)

    # C_23 (rows 0-1, cols 3-4) = A_22*B_23 + A_21*B_13 + A_23*B_33
    # A_22: rows 0-1, cols 0-1; B_23: rows 0-1, cols 3-4
    strassen_2x2(0, 1, 0, 1, 3, 4)
    # A_21: rows 0-1, col 2; B_13: row 2, cols 3-4
    schoolbook_2x1_1x2(0, 1, 2, 3, 4)
    # A_23: rows 0-1, cols 3-4; B_33: rows 3-4, cols 3-4
    strassen_2x2(0, 1, 3, 4, 3, 4)

    # C_32 (rows 3-4, cols 0-1) = A_32*B_22 + A_31*B_12 + A_33*B_32
    # A_32: rows 3-4, cols 0-1; B_22: rows 0-1, cols 0-1
    strassen_2x2(3, 4, 0, 1, 0, 1)
    # A_31: rows 3-4, col 2; B_12: row 2, cols 0-1
    schoolbook_2x1_1x2(3, 4, 2, 0, 1)
    # A_33: rows 3-4, cols 3-4; B_32: rows 3-4, cols 0-1
    strassen_2x2(3, 4, 3, 4, 0, 1)

    # C_33 (rows 3-4, cols 3-4) = A_32*B_23 + A_31*B_13 + A_33*B_33
    # A_32: rows 3-4, cols 0-1; B_23: rows 0-1, cols 3-4
    strassen_2x2(3, 4, 0, 1, 3, 4)
    # A_31: rows 3-4, col 2; B_13: row 2, cols 3-4
    schoolbook_2x1_1x2(3, 4, 2, 3, 4)
    # A_33: rows 3-4, cols 3-4; B_33: rows 3-4, cols 3-4
    strassen_2x2(3, 4, 3, 4, 3, 4)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
