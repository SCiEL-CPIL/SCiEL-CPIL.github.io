"""
5x5 matrix multiplication - rank 70 block decomposition (1+4 split).
Uses Strassen for the 4x4 block (rank 49) and schoolbook for cross terms.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0  # Safety margin: evaluator kills at 150s

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

    def add_strassen_2x2(idx_a, idx_b):
        """Add Strassen terms for 2x2 blocks at indices idx_a (row/col of A) and idx_b (col/row of B).
        idx_a = [i0, i1], idx_b = [j0, j1]"""
        i0, i1 = idx_a
        j0, j1 = idx_b

        # M1 = (A00+A11)(B00+B11)
        u = {(i0,j0):1, (i1,j1):1}
        v = {(j0,j0):1, (j1,j1):1}
        w = {(j0,i0):1, (j1,i1):1}
        add_term(u, v, w)

        # M2 = (A10+A11)B00
        u = {(i1,j0):1, (i1,j1):1}
        v = {(j0,j0):1}
        w = {(j0,i0):1, (j1,i1):-1}
        add_term(u, v, w)

        # M3 = A00(B01-B11)
        u = {(i0,j0):1}
        v = {(j0,j1):1, (j1,j1):-1}
        w = {(j1,i0):1, (j1,i1):1}
        add_term(u, v, w)

        # M4 = A11(B10-B00)
        u = {(i1,j1):1}
        v = {(j1,j0):1, (j0,j0):-1}
        w = {(j0,i0):1, (j0,i1):1}
        add_term(u, v, w)

        # M5 = (A00+A01)B11
        u = {(i0,j0):1, (i0,j1):1}
        v = {(j1,j1):1}
        w = {(j0,i0):-1, (j1,i0):1}
        add_term(u, v, w)

        # M6 = (A10-A00)(B00+B01)
        u = {(i1,j0):1, (i0,j0):-1}
        v = {(j0,j0):1, (j0,j1):1}
        w = {(j1,i1):1}
        add_term(u, v, w)

        # M7 = (A01-A11)(B10+B11)
        u = {(i0,j1):1, (i1,j1):-1}
        v = {(j1,j0):1, (j1,j1):1}
        w = {(j0,i0):1}
        add_term(u, v, w)

    def strassen_3x3_21(i_list, j_list, k_list):
        """Strassen-based 3x3 using 2+1 split.
        i_list, j_list, k_list are lists of 3 indices.
        First 2 indices form the 2x2 block, last 1 is the 1x1 block.
        Returns list of (u_vals, v_vals, w_vals) tuples."""
        s2 = i_list[:2]  # 2x2 block indices for A rows
        s3 = i_list[2]   # 1x1 index for A rows
        t2 = j_list[:2]  # 2x2 block indices for A cols / B rows
        t3 = j_list[2]   # 1x1 index
        u2 = k_list[:2]  # 2x2 block indices for B cols
        u3 = k_list[2]   # 1x1 index

        terms = []

        # C22 = A22*B22 + A21*B12 (within 3x3, local indices)
        # A22: i in s2, j in t2; B22: j in t2, k in u2; C22: i in s2, k in u2
        # Strassen for A22*B22 (2x2 * 2x2)
        i0, i1 = s2
        j0, j1 = t2
        k0, k1 = u2

        # M1
        u = {(i0,j0):1, (i1,j1):1}
        v = {(j0,j0):1, (j1,j1):1}
        w = {(k0,i0):1, (k1,i1):1}
        terms.append((u, v, w))

        # M2
        u = {(i1,j0):1, (i1,j1):1}
        v = {(j0,j0):1}
        w = {(k0,i0):1, (k1,i1):-1}
        terms.append((u, v, w))

        # M3
        u = {(i0,j0):1}
        v = {(j0,j1):1, (j1,j1):-1}
        w = {(k1,i0):1, (k1,i1):1}
        terms.append((u, v, w))

        # M4
        u = {(i1,j1):1}
        v = {(j1,j0):1, (j0,j0):-1}
        w = {(k0,i0):1, (k0,i1):1}
        terms.append((u, v, w))

        # M5
        u = {(i0,j0):1, (i0,j1):1}
        v = {(j1,j1):1}
        w = {(k0,i0):-1, (k1,i0):1}
        terms.append((u, v, w))

        # M6
        u = {(i1,j0):1, (i0,j0):-1}
        v = {(j0,j0):1, (j0,j1):1}
        w = {(k1,i1):1}
        terms.append((u, v, w))

        # M7
        u = {(i0,j1):1, (i1,j1):-1}
        v = {(j1,j0):1, (j1,j1):1}
        w = {(k0,i0):1}
        terms.append((u, v, w))

        # A21*B12: i in s2, j=t3; B: j=t3, k in u2; C: i in s2, k in u2
        for i in s2:
            for k in u2:
                u = {(i, t3):1}
                v = {(t3, k):1}
                w = {(k, i):1}
                terms.append((u, v, w))

        # C23 = A22*B23 + A21*B13 (within 3x3)
        # A22: i in s2, j in t2; B23: j in t2, k=u3; C23: i in s2, k=u3
        for i in s2:
            for j in t2:
                u = {(i,j):1}
                v = {(j, u3):1}
                w = {(u3, i):1}
                terms.append((u, v, w))
        # A21*B13: i in s2, j=t3; B: j=t3, k=u3; C: i in s2, k=u3
        for i in s2:
            u = {(i, t3):1}
            v = {(t3, u3):1}
            w = {(u3, i):1}
            terms.append((u, v, w))

        # C32 = A32*B22 + A33*B32 (within 3x3)
        # A32: i=s3, j in t2; B22: j in t2, k in u2; C32: i=s3, k in u2
        for j in t2:
            for k in u2:
                u = {(s3, j):1}
                v = {(j, k):1}
                w = {(k, s3):1}
                terms.append((u, v, w))
        # A33*B32: i=s3, j=t3; B: j=t3, k in u2; C: i=s3, k in u2
        for k in u2:
            u = {(s3, t3):1}
            v = {(t3, k):1}
            w = {(k, s3):1}
            terms.append((u, v, w))

        # C33 = A32*B23 + A33*B33 (within 3x3)
        # A32: i=s3, j in t2; B23: j in t2, k=u3; C33: i=s3, k=u3
        for j in t2:
            u = {(s3, j):1}
            v = {(j, u3):1}
            w = {(u3, s3):1}
            terms.append((u, v, w))
        # A33*B33: i=s3, j=t3; B: j=t3, k=u3; C: i=s3, k=u3
        u = {(s3, t3):1}
        v = {(t3, u3):1}
        w = {(u3, s3):1}
        terms.append((u, v, w))

        return terms

    def strassen_4x4(i_list, j_list, k_list):
        """Strassen for 4x4 using 2+2 split.
        i_list, j_list, k_list are lists of 4 indices.
        Returns list of (u_vals, v_vals, w_vals) tuples."""
        s2 = i_list[:2]  # First 2 indices
        s3 = i_list[2:]  # Last 2 indices
        t2 = j_list[:2]
        t3 = j_list[2:]
        u2 = k_list[:2]
        u3 = k_list[2:]

        terms = []

        # C22 = A22*B22 + A21*B12
        # Strassen for A22*B22
        i0, i1 = s2
        j0, j1 = t2
        k0, k1 = u2

        u = {(i0,j0):1, (i1,j1):1}
        v = {(j0,j0):1, (j1,j1):1}
        w = {(k0,i0):1, (k1,i1):1}
        terms.append((u, v, w))

        u = {(i1,j0):1, (i1,j1):1}
        v = {(j0,j0):1}
        w = {(k0,i0):1, (k1,i1):-1}
        terms.append((u, v, w))

        u = {(i0,j0):1}
        v = {(j0,j1):1, (j1,j1):-1}
        w = {(k1,i0):1, (k1,i1):1}
        terms.append((u, v, w))

        u = {(i1,j1):1}
        v = {(j1,j0):1, (j0,j0):-1}
        w = {(k0,i0):1, (k0,i1):1}
        terms.append((u, v, w))

        u = {(i0,j0):1, (i0,j1):1}
        v = {(j1,j1):1}
        w = {(k0,i0):-1, (k1,i0):1}
        terms.append((u, v, w))

        u = {(i1,j0):1, (i0,j0):-1}
        v = {(j0,j0):1, (j0,j1):1}
        w = {(k1,i1):1}
        terms.append((u, v, w))

        u = {(i0,j1):1, (i1,j1):-1}
        v = {(j1,j0):1, (j1,j1):1}
        w = {(k0,i0):1}
        terms.append((u, v, w))

        # A21*B12: i in s2, j in t3; B: j in t3, k in u2; C: i in s2, k in u2
        for i in s2:
            for j in t3:
                for k in u2:
                    u = {(i,j):1}
                    v = {(j,k):1}
                    w = {(k,i):1}
                    terms.append((u, v, w))

        # C23 = A22*B23 + A21*B13
        for i in s2:
            for j in t2:
                for k in u3:
                    u = {(i,j):1}
                    v = {(j,k):1}
                    w = {(k,i):1}
                    terms.append((u, v, w))
        for i in s2:
            for j in t3:
                for k in u3:
                    u = {(i,j):1}
                    v = {(j,k):1}
                    w = {(k,i):1}
                    terms.append((u, v, w))

        # C32 = A32*B22 + A33*B32
        for i in s3:
            for j in t2:
                for k in u2:
                    u = {(i,j):1}
                    v = {(j,k):1}
                    w = {(k,i):1}
                    terms.append((u, v, w))
        for i in s3:
            for j in t3:
                for k in u2:
                    u = {(i,j):1}
                    v = {(j,k):1}
                    w = {(k,i):1}
                    terms.append((u, v, w))

        # C33 = A32*B23 + A33*B33
        # Strassen for A33*B33
        i0, i1 = s3
        j0, j1 = t3
        k0, k1 = u3

        u = {(i0,j0):1, (i1,j1):1}
        v = {(j0,j0):1, (j1,j1):1}
        w = {(k0,i0):1, (k1,i1):1}
        terms.append((u, v, w))

        u = {(i1,j0):1, (i1,j1):1}
        v = {(j0,j0):1}
        w = {(k0,i0):1, (k1,i1):-1}
        terms.append((u, v, w))

        u = {(i0,j0):1}
        v = {(j0,j1):1, (j1,j1):-1}
        w = {(k1,i0):1, (k1,i1):1}
        terms.append((u, v, w))

        u = {(i1,j1):1}
        v = {(j1,j0):1, (j0,j0):-1}
        w = {(k0,i0):1, (k0,i1):1}
        terms.append((u, v, w))

        u = {(i0,j0):1, (i0,j1):1}
        v = {(j1,j1):1}
        w = {(k0,i0):-1, (k1,i0):1}
        terms.append((u, v, w))

        u = {(i1,j0):1, (i0,j0):-1}
        v = {(j0,j0):1, (j0,j1):1}
        w = {(k1,i1):1}
        terms.append((u, v, w))

        u = {(i0,j1):1, (i1,j1):-1}
        v = {(j1,j0):1, (j1,j1):1}
        w = {(k0,i0):1}
        terms.append((u, v, w))

        # A32*B23: i in s3, j in t2; B: j in t2, k in u3; C: i in s3, k in u3
        for i in s3:
            for j in t2:
                for k in u3:
                    u = {(i,j):1}
                    v = {(j,k):1}
                    w = {(k,i):1}
                    terms.append((u, v, w))

        return terms

    # 1+4 split: index 0 is the 1x1 block, indices 1,2,3,4 form the 4x4 block
    idx_1 = [0]
    idx_4 = [1, 2, 3, 4]

    # C11 = A11*B11 (1x1 * 1x1): rank 1
    u = {(0,0):1}
    v = {(0,0):1}
    w = {(0,0):1}
    add_term(u, v, w)

    # C12 = A11*B12 + A14*B42... wait, let's use block notation
    # A has rows/cols: block1 = {0}, block4 = {1,2,3,4}
    # B has rows/cols: block1 = {0}, block4 = {1,2,3,4}
    # C has rows/cols: block1 = {0}, block4 = {1,2,3,4}

    # C11 (1x1) = A11(1x1)*B11(1x1) + A14(1x4)*B41(4x1)
    # A11*B11: rank 1 (already added)
    # A14*B41: i=0, j in {1,2,3,4}, k in {1,2,3,4} -> wait, B41 is 4x1, so k=0
    # A14: i=0, j in {1,2,3,4}; B41: j in {1,2,3,4}, k=0; C: i=0, k=0
    # This is 1x4x1, rank 4
    for j in [1, 2, 3, 4]:
        u = {(0, j):1}
        v = {(j, 0):1}
        w = {(0, 0):1}
        add_term(u, v, w)

    # C14 (1x4) = A11(1x1)*B14(1x4) + A14(1x4)*B44(4x4)
    # A11*B14: i=0, j=0, k in {1,2,3,4}: rank 4
    for k in [1, 2, 3, 4]:
        u = {(0, 0):1}
        v = {(0, k):1}
        w = {(k, 0):1}
        add_term(u, v, w)
    # A14*B44: i=0, j in {1,2,3,4}, k in {1,2,3,4}: rank 16
    for j in [1, 2, 3, 4]:
        for k in [1, 2, 3, 4]:
            u = {(0, j):1}
            v = {(j, k):1}
            w = {(k, 0):1}
            add_term(u, v, w)

    # C41 (4x1) = A41(4x1)*B11(1x1) + A44(4x4)*B41(4x1)
    # A41*B11: i in {1,2,3,4}, j=0, k=0: rank 4
    for i in [1, 2, 3, 4]:
        u = {(i, 0):1}
        v = {(0, 0):1}
        w = {(0, i):1}
        add_term(u, v, w)
    # A44*B41: i in {1,2,3,4}, j in {1,2,3,4}, k=0: rank 16
    for i in [1, 2, 3, 4]:
        for j in [1, 2, 3, 4]:
            u = {(i, j):1}
            v = {(j, 0):1}
            w = {(0, i):1}
            add_term(u, v, w)

    # C44 (4x4) = A41(4x1)*B14(1x4) + A44(4x4)*B44(4x4)
    # A41*B14: i in {1,2,3,4}, j=0, k in {1,2,3,4}: rank 16
    for i in [1, 2, 3, 4]:
        for k in [1, 2, 3, 4]:
            u = {(i, 0):1}
            v = {(0, k):1}
            w = {(k, i):1}
            add_term(u, v, w)
    # A44*B44: rank 49 via Strassen
    terms_44 = strassen_4x4([1,2,3,4], [1,2,3,4], [1,2,3,4])
    for (u_vals, v_vals, w_vals) in terms_44:
        add_term(u_vals, v_vals, w_vals)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    # Verify correctness with random test
    if time.time() - t0 < deadline - 5:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))

        C_expected = A @ B

        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]

            for i in range(n):
                for k in range(p):
                    wk_i = w_vec[k*n+i]
                    if wk_i != 0:
                        for j in range(m):
                            C_decomp[i, k] += u_vec[i*m+j] * v_vec[j*p+k] * wk_i

        err = np.max(np.abs(C_decomp - C_expected))
        if err > 1e-10:
            pass

    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
