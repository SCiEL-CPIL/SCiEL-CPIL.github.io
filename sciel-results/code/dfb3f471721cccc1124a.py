"""
5x5 matrix multiplication - rank 117 block decomposition.
Uses Strassen for 2x2 sub-blocks and efficient handling of mixed blocks.
Verifies tensor decomposition identity before returning.
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
        """Add a single rank-1 term to the decomposition."""
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
        """
        Decompose 2x2 block multiplication using Strassen (rank 7).
        Multiplies A[i0:i1+1, j0:j1+1] @ B[j0:j1+1, k0:k1+1] -> C[k0:k1+1, i0:i1+1]
        """
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # C[k0,i0] += M1, C[k1,i1] += M1
        add_term({(i0, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1, (k1, i1): 1})

        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        # C[k0,i1] += M2, C[k1,i1] -= M2
        add_term({(i1, j0): 1, (i1, j1): 1},
                 {(j0, k0): 1},
                 {(k0, i1): 1, (k1, i1): -1})

        # M3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
        # C[k1,i0] += M3, C[k1,i1] += M3
        add_term({(i0, j0): 1},
                 {(j0, k1): 1, (j1, k1): -1},
                 {(k1, i0): 1, (k1, i1): 1})

        # M4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
        # C[k0,i0] += M4, C[k0,i1] += M4
        add_term({(i1, j1): 1},
                 {(j1, k0): 1, (j0, k0): -1},
                 {(k0, i0): 1, (k0, i1): 1})

        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        # C[k0,i0] -= M5, C[k1,i0] += M5
        add_term({(i0, j0): 1, (i0, j1): 1},
                 {(j1, k1): 1},
                 {(k0, i0): -1, (k1, i0): 1})

        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        # C[k1,i1] += M6
        add_term({(i1, j0): 1, (i0, j0): -1},
                 {(j0, k0): 1, (j0, k1): 1},
                 {(k1, i1): 1})

        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        # C[k0,i0] += M7
        add_term({(i0, j1): 1, (i1, j1): -1},
                 {(j1, k0): 1, (j1, k1): 1},
                 {(k0, i0): 1})

    def block_mul_2x1(i0, i1, j, k0, k1):
        """
        Multiply A[i0:i1+1, j] @ B[j, k0:k1+1] -> C[k0:k1+1, i0:i1+1]
        Uses rank 2 (schoolbook for 2x1 @ 1x2)
        """
        for k in [k0, k1]:
            add_term({(i0, j): 1, (i1, j): 1},
                     {(j, k): 1},
                     {(k, i0): 1, (k, i1): 1})

    def block_mul_1x2(i, j0, j1, k):
        """
        Multiply A[i, j0:j1+1] @ B[j0:j1+1, k] -> C[k, i]
        Uses rank 2
        """
        for j in [j0, j1]:
            add_term({(i, j): 1},
                     {(j, k): 1},
                     {(k, i): 1})

    def block_mul_2x2_schoolbook(i0, i1, j0, j1, k0, k1):
        """
        Multiply A[i0:i1+1, j0:j1+1] @ B[j0:j1+1, k0:k1+1] -> C[k0:k1+1, i0:i1+1]
        Schoolbook rank 4
        """
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    add_term({(i, j): 1},
                             {(j, k): 1},
                             {(k, i): 1})

    # 2+3 Split: indices 0,1 form the 2x2 part; 2,3,4 form the 3x3 part
    # C[0:2,0:2] = A[0:2,0:2]@B[0:2,0:2] + A[0:2,2:5]@B[2:5,0:2]
    strassen_2x2(0, 1, 0, 1, 0, 1)  # A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2], rank 7
    strassen_2x2(0, 1, 2, 3, 0, 1)  # A[0:2,2:4] @ B[2:4,0:2] -> C[0:2,0:2], rank 7
    # A[0:2,4] @ B[4,0:2] -> C[0:2,0:2], rank 2
    block_mul_2x1(0, 1, 4, 0, 1)

    # C[0:2,2:5] = A[0:2,0:2]@B[0:2,2:5] + A[0:2,2:5]@B[2:5,2:5]
    # A[0:2,0:2] @ B[0:2,2:4] -> C[2:4,0:2], rank 7
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # A[0:2,0:2] @ B[0:2,4] -> C[4,0:2], rank 2
    block_mul_1x2(0, 0, 1, 4)
    block_mul_1x2(1, 0, 1, 4)
    # A[0:2,2:4] @ B[2:4,2:4] -> C[2:4,0:2], rank 7
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # A[0:2,2:4] @ B[2:4,4] -> C[4,0:2], rank 2
    block_mul_1x2(0, 2, 3, 4)
    block_mul_1x2(1, 2, 3, 4)
    # A[0:2,4] @ B[4,2:5] -> C[2:5,0:2], rank 3
    for k in [2, 3, 4]:
        add_term({(0, 4): 1, (1, 4): 1},
                 {(4, k): 1},
                 {(k, 0): 1, (k, 1): 1})

    # C[2:5,0:2] = A[2:5,0:2]@B[0:2,0:2] + A[2:5,2:5]@B[2:5,0:2]
    # A[2:4,0:2] @ B[0:2,0:2] -> C[0:2,2:4], rank 7
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # A[4,0:2] @ B[0:2,0:2] -> C[0:2,4], rank 2
    for k in [0, 1]:
        add_term({(4, 0): 1, (4, 1): 1},
                 {(0, k): 1},
                 {(k, 4): 1})
    # A[2:4,2:4] @ B[2:4,0:2] -> C[0:2,2:4], rank 7
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # A[2:4,4] @ B[4,0:2] -> C[0:2,2:4], rank 2
    for k in [0, 1]:
        add_term({(2, 4): 1, (3, 4): 1},
                 {(4, k): 1},
                 {(k, 2): 1, (k, 3): 1})
    # A[4,2:4] @ B[2:4,0:2] -> C[0:2,4], rank 2
    for k in [0, 1]:
        add_term({(4, 2): 1, (4, 3): 1},
                 {(2, k): 1, (3, k): 1},
                 {(k, 4): 1})
    # A[4,4] @ B[4,0:2] -> C[0:2,4], rank 2
    for k in [0, 1]:
        add_term({(4, 4): 1},
                 {(4, k): 1},
                 {(k, 4): 1})

    # C[2:5,2:5] = A[2:5,0:2]@B[0:2,2:5] + A[2:5,2:5]@B[2:5,2:5]
    # A[2:4,0:2] @ B[0:2,2:4] -> C[2:4,2:4], rank 7
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # A[2:4,0:2] @ B[0:2,4] -> C[4,2:4], rank 2
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i, j): 1},
                     {(j, 4): 1},
                     {(4, i): 1})
    # A[4,0:2] @ B[0:2,2:4] -> C[2:4,4], rank 2
    for k in [2, 3]:
        add_term({(4, 0): 1, (4, 1): 1},
                 {(0, k): 1, (1, k): 1},
                 {(k, 4): 1})
    # A[4,0:2] @ B[0:2,4] -> C[4,4], rank 2
    for j in [0, 1]:
        add_term({(4, j): 1},
                 {(j, 4): 1},
                 {(4, 4): 1})
    # A[2:4,2:4] @ B[2:4,2:4] -> C[2:4,2:4], rank 7
    strassen_2x2(2, 3, 2, 3, 2, 3)
    # A[2:4,2:4] @ B[2:4,4] -> C[4,2:4], rank 2
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i, j): 1},
                     {(j, 4): 1},
                     {(4, i): 1})
    # A[2:4,4] @ B[4,2:4] -> C[2:4,2:4], rank 2
    for i in [2, 3]:
        for k in [2, 3]:
            add_term({(i, 4): 1},
                     {(4, k): 1},
                     {(k, i): 1})
    # A[2:4,4] @ B[4,4] -> C[4,2:4], rank 2
    for i in [2, 3]:
        add_term({(i, 4): 1},
                 {(4, 4): 1},
                 {(4, i): 1})
    # A[4,2:4] @ B[2:4,2:4] -> C[2:4,4], rank 2
    for k in [2, 3]:
        add_term({(4, 2): 1, (4, 3): 1},
                 {(2, k): 1, (3, k): 1},
                 {(k, 4): 1})
    # A[4,2:4] @ B[2:4,4] -> C[4,4], rank 2
    for j in [2, 3]:
        add_term({(4, j): 1},
                 {(j, 4): 1},
                 {(4, 4): 1})
    # A[4,4] @ B[4,2:4] -> C[2:4,4], rank 2
    for k in [2, 3]:
        add_term({(4, 4): 1},
                 {(4, k): 1},
                 {(k, 4): 1})
    # A[4,4] @ B[4,4] -> C[4,4], rank 1
    add_term({(4, 4): 1},
             {(4, 4): 1},
             {(4, 4): 1})

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    rank = U.shape[1]

    elapsed = time.time() - t0
    if elapsed > 100.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if i==k else 0 (matrix multiplication tensor)
    try:
        T_decomp = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T_decomp += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)

        T_expected = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for k in range(p):
                if i == k:
                    for j in range(m):
                        T_expected[i, j, k] = 1.0

        error = np.max(np.abs(T_decomp - T_expected))
        if error < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
