import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vals, v_vals, w_vals):
        U_vec = np.zeros(n * m)
        V_vec = np.zeros(m * p)
        W_vec = np.zeros(p * n)
        for (i, j), val in u_vals.items():
            U_vec[i * m + j] = val
        for (j, k), val in v_vals.items():
            V_vec[j * p + k] = val
        for (k, i), val in w_vals.items():
            W_vec[k * n + i] = val
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        add_term({(i0, j0): 1, (i1, j1): 1}, {(j0, k0): 1, (j1, k1): 1}, {(k0, i0): 1, (k1, i1): 1})
        add_term({(i1, j0): 1, (i1, j1): 1}, {(j0, k0): 1}, {(k0, i1): 1, (k1, i1): -1})
        add_term({(i0, j0): 1}, {(j0, k1): 1, (j1, k1): -1}, {(k1, i0): 1, (k1, i1): 1})
        add_term({(i1, j1): 1}, {(j1, k0): 1, (j0, k0): -1}, {(k0, i0): 1, (k0, i1): 1})
        add_term({(i0, j0): 1, (i0, j1): 1}, {(j1, k1): 1}, {(k0, i0): -1, (k1, i0): 1})
        add_term({(i1, j0): 1, (i0, j0): -1}, {(j0, k0): 1, (j0, k1): 1}, {(k1, i1): 1})
        add_term({(i0, j1): 1, (i1, j1): -1}, {(j1, k0): 1, (j1, k1): 1}, {(k0, i0): 1})

    def schoolbook_block(rows, cols_a, cols_b):
        for i in rows:
            for j in cols_a:
                for k in cols_b:
                    add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    # Block decomposition 2+3
    # A is 5x5, B is 5x5, C is 5x5
    # Rows of A: 0,1 (top 2), 2,3,4 (bottom 3)
    # Cols of A: 0,1 (left 2), 2,3,4 (right 3)
    # Rows of B: 0,1 (top 2), 2,3,4 (bottom 3)
    # Cols of B: 0,1 (left 2), 2,3,4 (right 3)

    # C[0:2, 0:2] = A[0:2, 0:2]B[0:2, 0:2] + A[0:2, 2:5]B[2:5, 0:2]
    # 1. A[0:2, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> Strassen (7)
    strassen_2x2(0, 1, 0, 1, 0, 1)
    
    # 2. A[0:2, 2:5] (2x3) * B[2:5, 0:2] (3x2)
    # Split B[2:5, 0:2] into B[2:4, 0:2] (2x2) and B[4, 0:2] (1x2)
    # A[0:2, 2:4] (2x2) * B[2:4, 0:2] (2x2) -> Strassen (7)
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # A[0:2, 4] (2x1) * B[4, 0:2] (1x2) -> Schoolbook (2*1*2 = 4)
    schoolbook_block([0, 1], [4], [0, 1])

    # C[0:2, 2:5] = A[0:2, 0:2]B[0:2, 2:5] + A[0:2, 2:5]B[2:5, 2:5]
    # 3. A[0:2, 0:2] (2x2) * B[0:2, 2:5] (2x3)
    # Split B[0:2, 2:5] into B[0:2, 2:4] (2x2) and B[0:2, 4] (2x1)
    # A[0:2, 0:2] (2x2) * B[0:2, 2:4] (2x2) -> Strassen (7)
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A[0:2, 0:2] (2x2) * B[0:2, 4] (2x1) -> Schoolbook (2*2*1 = 4)
    schoolbook_block([0, 1], [0, 1], [4])

    # 4. A[0:2, 2:5] (2x3) * B[2:5, 2:5] (3x3)
    # Split B[2:5, 2:5] into B[2:4, 2:4] (2x2), B[2:4, 4] (2x1), B[4, 2:4] (1x2), B[4, 4] (1x1)
    # A[0:2, 2:4] (2x2) * B[2:4, 2:4] (2x2) -> Strassen (7)
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A[0:2, 2:4] (2x2) * B[2:4, 4] (2x1) -> Schoolbook (2*2*1 = 4)
    schoolbook_block([0, 1], [2, 3], [4])
    # A[0:2, 4] (2x1) * B[4, 2:5] (1x3) -> Schoolbook (2*1*3 = 6)
    schoolbook_block([0, 1], [4], [2, 3, 4])

    # C[2:5, 0:2] = A[2:5, 0:2]B[0:2, 0:2] + A[2:5, 2:5]B[2:5, 0:2]
    # 5. A[2:5, 0:2] (3x2) * B[0:2, 0:2] (2x2)
    # Split A[2:5, 0:2] into A[2:4, 0:2] (2x2) and A[4, 0:2] (1x2)
    # A[2:4, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> Strassen (7)
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A[4, 0:2] (1x2) * B[0:2, 0:2] (2x2) -> Schoolbook (1*2*2 = 4)
    schoolbook_block([4], [0, 1], [0, 1])

    # 6. A[2:5, 2:5] (3x3) * B[2:5, 0:2] (3x2)
    # Split A[2:5, 2:5] into A[2:4, 2:4] (2x2), A[2:4, 4] (2x1), A[4, 2:4] (1x2), A[4, 4] (1x1)
    # Split B[2:5, 0:2] into B[2:4, 0:2] (2x2) and B[4, 0:2] (1x2)
    # A[2:4, 2:4] (2x2) * B[2:4, 0:2] (2x2) -> Strassen (7)
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A[2:4, 4] (2x1) * B[4, 0:2] (1x2) -> Schoolbook (2*1*2 = 4)
    schoolbook_block([2, 3], [4], [0, 1])
    # A[4, 2:4] (1x2) * B[2:4, 0:2] (2x2) -> Schoolbook (1*2*2 = 4)
    schoolbook_block([4], [2, 3], [0, 1])
    # A[4, 4] (1x1) * B[4, 0:2] (1x2) -> Schoolbook (1*1*2 = 2)
    schoolbook_block([4], [4], [0, 1])

    # C[2:5, 2:5] = A[2:5, 0:2]B[0:2, 2:5] + A[2:5, 2:5]B[2:5, 2:5]
    # 7. A[2:5, 0:2] (3x2) * B[0:2, 2:5] (2x3)
    # Split A[2:5, 0:2] into A[2:4, 0:2] (2x2) and A[4, 0:2] (1x2)
    # Split B[0:2, 2:5] into B[0:2, 2:4] (2x2) and B[0:2, 4] (2x1)
    # A[2:4, 0:2] (2x2) * B[0:2, 2:4] (2x2) -> Strassen (7)
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A[2:4, 0:2] (2x2) * B[0:2, 4] (2x1) -> Schoolbook (2*2*1 = 4)
    schoolbook_block([2, 3], [0, 1], [4])
    # A[4, 0:2] (1x2) * B[0:2, 2:4] (2x2) -> Schoolbook (1*2*2 = 4)
    schoolbook_block([4], [0, 1], [2, 3])
    # A[4, 0:2] (1x2) * B[0:2, 4] (2x1) -> Schoolbook (1*2*1 = 2)
    schoolbook_block([4], [0, 1], [4])

    # 8. A[2:5, 2:5] (3x3) * B[2:5, 2:5] (3x3)
    # Split A[2:5, 2:5] into A[2:4, 2:4] (2x2), A[2:4, 4] (2x1), A[4, 2:4] (1x2), A[4, 4] (1x1)
    # Split B[2:5, 2:5] into B[2:4, 2:4] (2x2), B[2:4, 4] (2x1), B[4, 2:4] (1x2), B[4, 4] (1x1)
    # A[2:4, 2:4] (2x2) * B[2:4, 2:4] (2x2) -> Strassen (7)
    strassen_2x2(2, 3, 2, 3, 2, 3)
    # A[2:4, 2:4] (2x2) * B[2:4, 4] (2x1) -> Schoolbook (2*2*1 = 4)
    schoolbook_block([2, 3], [2, 3], [4])
    # A[2:4, 4] (2x1) * B[4, 2:5] (1x3) -> Schoolbook (2*1*3 = 6)
    schoolbook_block([2, 3], [4], [2, 3, 4])
    # A[4, 2:4] (1x2) * B[2:4, 2:5] (2x3) -> Schoolbook (1*2*3 = 6)
    schoolbook_block([4], [2, 3], [2, 3, 4])
    # A[4, 4] (1x1) * B[4, 2:5] (1x3) -> Schoolbook (1*1*3 = 3)
    schoolbook_block([4], [4], [2, 3, 4])

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    rank = U.shape[1]

    # Quick verification
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
            # Fallback to naive
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
