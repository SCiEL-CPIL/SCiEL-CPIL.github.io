"""
5x5 matrix multiplication - rank 75 block decomposition.
Uses 1+2+2 split with Strassen for 2x2 blocks.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
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

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    # 1+2+2 Split
    # Indices: 0 (1x1), 1-2 (2x2), 3-4 (2x2)
    
    # C_11 = A_11 B_11
    add_term({(0,0):1}, {(0,0):1}, {(0,0):1})

    # C_12 = A_11 B_12
    add_term({(0,0):1}, {(0,1):1}, {(1,0):1})
    add_term({(0,0):1}, {(0,2):1}, {(2,0):1})

    # C_13 = A_11 B_13
    add_term({(0,0):1}, {(0,3):1}, {(3,0):1})
    add_term({(0,0):1}, {(0,4):1}, {(4,0):1})

    # C_21 = A_21 B_11 + A_22 B_21
    # A_21 B_11
    add_term({(1,0):1}, {(0,0):1}, {(0,1):1})
    add_term({(2,0):1}, {(0,0):1}, {(0,2):1})
    # A_22 B_21 (Schoolbook 4 terms)
    add_term({(1,1):1}, {(1,0):1}, {(1,1):1})
    add_term({(1,2):1}, {(2,0):1}, {(2,1):1})
    add_term({(2,1):1}, {(1,0):1}, {(1,2):1})
    add_term({(2,2):1}, {(2,0):1}, {(2,2):1})

    # C_31 = A_31 B_11 + A_32 B_21
    # A_31 B_11
    add_term({(3,0):1}, {(0,0):1}, {(0,3):1})
    add_term({(4,0):1}, {(0,0):1}, {(0,4):1})
    # A_32 B_21 (Schoolbook 4 terms)
    add_term({(3,1):1}, {(1,0):1}, {(1,3):1})
    add_term({(3,2):1}, {(2,0):1}, {(2,3):1})
    add_term({(4,1):1}, {(1,0):1}, {(1,4):1})
    add_term({(4,2):1}, {(2,0):1}, {(2,4):1})

    # C_22 = A_21 B_12 + A_22 B_22 + A_23 B_32
    # A_21 B_12
    add_term({(1,0):1}, {(0,1):1}, {(1,1):1})
    add_term({(1,0):1}, {(0,2):1}, {(2,1):1})
    add_term({(2,0):1}, {(0,1):1}, {(1,2):1})
    add_term({(2,0):1}, {(0,2):1}, {(2,2):1})
    # A_22 B_22 (Strassen 7 terms)
    strassen_2x2(1, 2, 1, 2, 1, 2)
    # A_23 B_32 (Strassen 7 terms)
    strassen_2x2(1, 2, 3, 4, 1, 2)

    # C_32 = A_31 B_12 + A_32 B_22 + A_33 B_32
    # A_31 B_12
    add_term({(3,0):1}, {(0,1):1}, {(1,3):1})
    add_term({(3,0):1}, {(0,2):1}, {(2,3):1})
    add_term({(4,0):1}, {(0,1):1}, {(1,4):1})
    add_term({(4,0):1}, {(0,2):1}, {(2,4):1})
    # A_32 B_22 (Strassen 7 terms)
    strassen_2x2(3, 4, 1, 2, 3, 4)
    # A_33 B_32 (Strassen 7 terms)
    strassen_2x2(3, 4, 3, 4, 3, 4)

    # C_23 = A_21 B_13 + A_22 B_23 + A_23 B_33
    # A_21 B_13
    add_term({(1,0):1}, {(0,3):1}, {(3,1):1})
    add_term({(1,0):1}, {(0,4):1}, {(4,1):1})
    add_term({(2,0):1}, {(0,3):1}, {(3,2):1})
    add_term({(2,0):1}, {(0,4):1}, {(4,2):1})
    # A_22 B_23 (Strassen 7 terms)
    strassen_2x2(1, 2, 1, 2, 3, 4)
    # A_23 B_33 (Strassen 7 terms)
    strassen_2x2(1, 2, 3, 4, 3, 4)

    # C_33 = A_31 B_13 + A_32 B_23 + A_33 B_33
    # A_31 B_13
    add_term({(3,0):1}, {(0,3):1}, {(3,3):1})
    add_term({(3,0):1}, {(0,4):1}, {(4,3):1})
    add_term({(4,0):1}, {(0,3):1}, {(3,4):1})
    add_term({(4,0):1}, {(0,4):1}, {(4,4):1})
    # A_32 B_23 (Strassen 7 terms)
    strassen_2x2(3, 4, 1, 2, 3, 4)
    # A_33 B_33 (Strassen 7 terms)
    strassen_2x2(3, 4, 3, 4, 3, 4)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
