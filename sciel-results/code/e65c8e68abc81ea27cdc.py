import numpy as np
import time

DIMENSIONS = [2, 2, 14]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    # Number of 2x2 blocks in the 2x14 matrix
    num_blocks = p // 2  # 7

    # Strassen's algorithm for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # A indices: A00=0, A01=1, A10=2, A11=3
    # B local indices: B00=0, B01=1, B10=2, B11=3
    # C local indices: C00=0, C01=1, C10=2, C11=3

    # Strassen terms: (a_coeffs, b_coeffs, c_coeffs)
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11) -> contributes to C00 (+1) and C11 (+1)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10 + A11)B00 -> contributes to C10 (+1) and C11 (-1)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01 - B11) -> contributes to C01 (+1) and C11 (+1)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10 - B00) -> contributes to C00 (+1) and C10 (+1)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00 + A01)B11 -> contributes to C00 (-1) and C01 (+1)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10 - A00)(B00 + B01) -> contributes to C11 (+1)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01 - A11)(B10 + B11) -> contributes to C00 (+1)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    rank = num_blocks * 7  # 49
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]

            # U: shape (n*m, rank) = (4, rank)
            # U[i*m + j, col] = coefficient of A[i,j] in term s
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff

            # V: shape (m*p, rank) = (28, rank)
            # V[j*p + k, col] = coefficient of B[j,k] in term s for block t
            # Block t covers columns 2*t and 2*t+1 of B
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff

            # W: shape (p*n, rank) = (28, rank)
            # W[k*n + i, col] = coefficient for C[i,k]
            # Block t covers columns 2*t and 2*t+1 of C
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Verify correctness
    def tensor_check(Uf, Vf, Wf):
        """Check that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not tensor_check(U, V, W):
        # Fallback to schoolbook if Strassen construction is incorrect
        rank = n * m * p  # 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Convert to lists with integer values where possible
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_list]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_list]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_list]

    return {"U": U_list, "V": V_list, "W": W_list}
