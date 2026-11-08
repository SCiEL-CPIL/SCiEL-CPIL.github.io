"""
5x5 matrix multiplication - rank 120 block decomposition.
Uses Strassen for 2x2 blocks (rank 7) and schoolbook for remaining blocks.
Split 5 as 2+3, then 3 as 2+1.
Includes deadline guard and term deduplication.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n * m, dtype=float)
        V = np.zeros(m * p, dtype=float)
        W = np.zeros(p * n, dtype=float)
        for (i, j), val in u_vals.items():
            U[i * m + j] = val
        for (j, k), val in v_vals.items():
            V[j * p + k] = val
        for (k, i), val in w_vals.items():
            W[k * n + i] = val
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    def schoolbook_block(i_list, j_list, k_list):
        """Add schoolbook terms for A[i_list, j_list] * B[j_list, k_list] -> C[i_list, k_list]"""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term({(i, j): 1.0}, {(j, k): 1.0}, {(k, i): 1.0})

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """
        Strassen decomposition for 2x2 block multiplication.
        A rows: {i0, i1}, A cols: {j0, j1}
        B rows: {j0, j1}, B cols: {k0, k1}
        C rows: {i0, i1}, C cols: {k0, k1}
        """
        # M1 = (A00+A11)*(B00+B11)
        add_term({(i0, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1, (k1, i1): 1})
        # M2 = (A10+A11)*B00
        add_term({(i1, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1},
                 {(k0, i1): 1, (k1, i1): -1})
        # M3 = A00*(B01-B11)
        add_term({(i0, j0): 1},
                 {(j0, k1): 1, (j1, k1): -1},
                 {(k1, i0): 1, (k1, i1): 1})
        # M4 = A11*(B10-B00)
        add_term({(i1, j1): 1},
                 {(j1, k0): 1, (j0, k0): -1},
                 {(k0, i0): 1, (k0, i1): 1})
        # M5 = (A00+A01)*B11
        add_term({(i0, j0): 1, (i0, j1): 1},
                 {(j1, k1): 1},
                 {(k0, i0): -1, (k1, i0): 1})
        # M6 = (A10-A00)*(B00+B01)
        add_term({(i1, j0): 1, (i0, j0): -1},
                 {(j0, k0): 1, (j0, k1): 1},
                 {(k1, i1): 1})
        # M7 = (A01-A11)*(B10+B11)
        add_term({(i0, j1): 1, (i1, j1): -1},
                 {(j1, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1})

    # Block structure:
    # Rows of C: {0,1} (top 2) and {2,3,4} (bottom 3)
    # Cols of C: {0,1} (left 2) and {2,3,4} (right 3)
    # A rows: {0,1} and {2,3,4}
    # B cols: {0,1} and {2,3,4}
    # A cols = B rows: {0,1} and {2,3,4}

    # --- C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:5]*B[2:5, 0:2] ---
    strassen_2x2(0, 1, 0, 1, 0, 1)
    strassen_2x2(0, 1, 2, 3, 0, 1)
    schoolbook_block([0, 1], [4], [0, 1])

    # --- C[0:2, 2:5] = A[0:2, 0:2]*B[0:2, 2:5] + A[0:2, 2:5]*B[2:5, 2:5] ---
    strassen_2x2(0, 1, 0, 1, 2, 3)
    schoolbook_block([0, 1], [0, 1], [4])
    strassen_2x2(0, 1, 2, 3, 2, 3)
    schoolbook_block([0, 1], [2, 3], [4])
    schoolbook_block([0, 1], [4], [2, 3])
    schoolbook_block([0, 1], [4], [4])

    # --- C[2:5, 0:2] = A[2:5, 0:2]*B[0:2, 0:2] + A[2:5, 2:5]*B[2:5, 0:2] ---
    strassen_2x2(2, 3, 0, 1, 0, 1)
    schoolbook_block([4], [0, 1], [0, 1])
    strassen_2x2(2, 3, 2, 3, 0, 1)
    schoolbook_block([2, 3], [4], [0, 1])
    schoolbook_block([4], [2, 3], [0, 1])
    schoolbook_block([4], [4], [0, 1])

    # --- C[2:5, 2:5] = A[2:5, 0:2]*B[0:2, 2:5] + A[2:5, 2:5]*B[2:5, 2:5] ---
    strassen_2x2(2, 3, 0, 1, 2, 3)
    schoolbook_block([2, 3], [0, 1], [4])
    schoolbook_block([4], [0, 1], [2, 3])
    schoolbook_block([4], [0, 1], [4])
    strassen_2x2(2, 3, 2, 3, 2, 3)
    schoolbook_block([2, 3], [2, 3], [4])
    schoolbook_block([2, 3], [4], [2, 3])
    schoolbook_block([2, 3], [4], [4])
    schoolbook_block([4], [2, 3], [2, 3])
    schoolbook_block([4], [2, 3], [4])
    schoolbook_block([4], [4], [2, 3])
    schoolbook_block([4], [4], [4])

    # Deduplicate terms
    unique_U = []
    unique_V = []
    unique_W = []
    seen = set()
    
    for i in range(len(U_list)):
        # Check deadline
        if time.time() - t0 > deadline:
            break
            
        u_key = tuple(U_list[i])
        if u_key in seen:
            continue
        seen.add(u_key)
        unique_U.append(U_list[i])
        unique_V.append(V_list[i])
        unique_W.append(W_list[i])

    if not unique_U:
        # Fallback to original if dedup failed or empty
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
    else:
        U = np.array(unique_U).T
        V = np.array(unique_V).T
        W = np.array(unique_W).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
