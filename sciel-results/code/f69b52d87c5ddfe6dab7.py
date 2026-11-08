import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0

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

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        add_term({(i0, j0): 1, (i1, j1): 1}, {(j0, k0): 1, (j1, k1): 1}, {(k0, i0): 1, (k1, i1): 1})
        add_term({(i1, j0): 1, (i1, j1): 1}, {(j0, k0): 1}, {(k0, i1): 1, (k1, i1): -1})
        add_term({(i0, j0): 1}, {(j0, k1): 1, (j1, k1): -1}, {(k1, i0): 1, (k1, i1): 1})
        add_term({(i1, j1): 1}, {(j1, k0): 1, (j0, k0): -1}, {(k0, i0): 1, (k0, i1): 1})
        add_term({(i0, j0): 1, (i0, j1): 1}, {(j1, k1): 1}, {(k0, i0): -1, (k1, i0): 1})
        add_term({(i1, j0): 1, (i0, j0): -1}, {(j0, k0): 1, (j0, k1): 1}, {(k1, i1): 1})
        add_term({(i0, j1): 1, (i1, j1): -1}, {(j1, k0): 1, (j1, k1): 1}, {(k0, i0): 1})

    def strassen_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        # i0,i1 are the 2x2 block rows, i2 is the 1x1 row
        # j0,j1 are the 2x2 block cols, j2 is the 1x1 col
        # k0,k1 are the 2x2 block cols of B, k2 is the 1x1 col of B
        
        # 1. A[0,1; 0,1] * B[0,1; 0,1] (7 terms)
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        
        # 2. A[0,1; 2] * B[2; 0,1] (4 terms)
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})
                
        # 3. A[2; 0,1] * B[0,1; 2] (2 terms)
        for j in [j0, j1]:
            add_term({(i2, j): 1}, {(j, k2): 1}, {(k2, i2): 1})
            
        # 4. A[2; 2] * B[2; 2] (1 term)
        add_term({(i2, j2): 1}, {(j2, k2): 1}, {(k2, i2): 1})

    # 1. C_22 = A_22 B_22 + A_23 B_32
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # A_23 (2x3) * B_32 (3x2)
    strassen_2x2(0, 1, 2, 3, 0, 1)
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i, 4): 1}, {(4, k): 1}, {(k, i): 1})

    # 2. C_23 = A_22 B_23 + A_23 B_33
    strassen_2x2(0, 1, 0, 1, 2, 3)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i, j): 1}, {(j, 4): 1}, {(4, i): 1})
    strassen_2x2(0, 1, 2, 3, 2, 3)
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i, j): 1}, {(j, 4): 1}, {(4, i): 1})
    for i in [0, 1]:
        for k in [2, 3, 4]:
            add_term({(i, 4): 1}, {(4, k): 1}, {(k, i): 1})

    # 3. C_32 = A_32 B_22 + A_33 B_32
    strassen_2x2(2, 3, 0, 1, 0, 1)
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4, j): 1}, {(j, k): 1}, {(k, 4): 1})
    strassen_2x2(2, 3, 2, 3, 0, 1)
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i, 4): 1}, {(4, k): 1}, {(k, i): 1})
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4, j): 1}, {(j, k): 1}, {(k, 4): 1})
    for k in [0, 1]:
        add_term({(4, 4): 1}, {(4, k): 1}, {(k, 4): 1})

    # 4. C_33 = A_32 B_23 + A_33 B_33
    strassen_2x2(2, 3, 0, 1, 2, 3)
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i, j): 1}, {(j, 4): 1}, {(4, i): 1})
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4, j): 1}, {(j, k): 1}, {(k, 4): 1})
    for j in [0, 1]:
        add_term({(4, j): 1}, {(j, 4): 1}, {(4, 4): 1})
    
    # A_33 B_33 (3x3 * 3x3) with optimized 2+1 split (14 terms)
    strassen_3x3(2, 3, 4, 2, 3, 4, 2, 3, 4)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    rank = U.shape[1]

    if time.time() - t0 < deadline - 5:
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                    if abs(val - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            U = np.zeros((n*m, n*m*p))
            V = np.zeros((m*p, n*m*p))
            W = np.zeros((p*n, n*m*p))
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        idx = i2*m*p + j2*p + k2
                        U[i2*m+j2, idx] = 1
                        V[j2*p+k2, idx] = 1
                        W[k2*n+i2, idx] = 1
            rank = n*m*p

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
