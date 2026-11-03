"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 6

    terms = []  # Each term: (U_dict, V_dict, W_dict)

    def add_term(U_entries, V_entries, W_entries):
        U_d = {}
        for (i, j, c) in U_entries:
            U_d[(i, j)] = U_d.get((i, j), 0) + c
        V_d = {}
        for (j, k, c) in V_entries:
            V_d[(j, k)] = V_d.get((j, k), 0) + c
        W_d = {}
        for (k, i, c) in W_entries:
            W_d[(k, i)] = W_d.get((k, i), 0) + c
        terms.append((U_d, V_d, W_d))

    def strassen_terms(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        r0, r1 = A_rows
        c0, c1 = A_cols
        s0, s1 = B_rows
        t0, t1 = B_cols

        # M1 = (A[r0,c0]+A[r1,c1]) * (B[s0,t0]+B[s1,t1])
        add_term(
            [(r0, c0, 1), (r1, c1, 1)],
            [(s0, t0, 1), (s1, t1, 1)],
            [(t0, r0, 1), (t1, r1, 1)]
        )
        # M2 = (A[r1,c0]+A[r1,c1]) * B[s0,t0]
        add_term(
            [(r1, c0, 1), (r1, c1, 1)],
            [(s0, t0, 1)],
            [(t0, r1, 1), (t1, r1, -1)]
        )
        # M3 = A[r0,c0] * (B[s0,t1]-B[s1,t1])
        add_term(
            [(r0, c0, 1)],
            [(s0, t1, 1), (s1, t1, -1)],
            [(t1, r0, 1), (t1, r1, 1)]
        )
        # M4 = A[r1,c1] * (B[s1,t0]-B[s0,t0])
        add_term(
            [(r1, c1, 1)],
            [(s1, t0, 1), (s0, t0, -1)],
            [(t0, r1, 1), (t0, r0, 1)]
        )
        # M5 = (A[r0,c0]+A[r0,c1]) * B[s1,t1]
        add_term(
            [(r0, c0, 1), (r0, c1, 1)],
            [(s1, t1, 1)],
            [(t1, r1, 1), (t0, r0, -1)]
        )
        # M6 = (A[r1,c1]-A[r0,c1]) * (B[s0,t0]+B[s0,t1])
        add_term(
            [(r1, c1, 1), (r0, c1, -1)],
            [(s0, t0, 1), (s0, t1, 1)],
            [(t1, r1, 1)]
        )
        # M7 = (A[r0,c1]-A[r1,c1]) * (B[s1,t1]+B[s0,t1])
        add_term(
            [(r0, c1, 1), (r1, c1, -1)],
            [(s1, t1, 1), (s0, t1, 1)],
            [(t0, r0, 1)]
        )

    def direct_1x2_2x2_terms(A_row, A_cols, B_rows, B_cols, C_row, C_cols):
        r = A_row
        c0, c1 = A_cols
        s0, s1 = B_rows
        t0, t1 = B_cols
        add_term([(r, c0, 1)], [(s0, t0, 1)], [(t0, r, 1)])
        add_term([(r, c1, 1)], [(s1, t0, 1)], [(t0, r, 1)])
        add_term([(r, c0, 1)], [(s0, t1, 1)], [(t1, r, 1)])
        add_term([(r, c1, 1)], [(s1, t1, 1)], [(t1, r, 1)])

    col_groups = [(0, 1), (2, 3), (4, 5)]

    for ci, (t0, t1) in enumerate(col_groups):
        strassen_terms((0, 1), (0, 1), (0, 1), (t0, t1), (0, 1), (t0, t1))
        strassen_terms((0, 1), (2, 3), (2, 3), (t0, t1), (0, 1), (t0, t1))
        direct_1x2_2x2_terms(2, (0, 1), (0, 1), (t0, t1), 2, (t0, t1))
        direct_1x2_2x2_terms(2, (2, 3), (2, 3), (t0, t1), 2, (t0, t1))

    def build_matrices(term_list):
        rank = len(term_list)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (U_d, V_d, W_d) in enumerate(term_list):
            for (i, j), coeff in U_d.items():
                U[i * m + j, c] = coeff
            for (j, k), coeff in V_d.items():
                V[j * p + k, c] = coeff
            for (k, i), coeff in W_d.items():
                W[k * n + i, c] = coeff
        return U, V, W

    def tensor_from(U, V, W):
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += u[:, :, None] * v[None, :, :] * w[None, None, :]
        return T

    def verify(U, V, W):
        T = tensor_from(U, V, W)
        return bool(np.allclose(T, np.ones((n, m, p)), atol=1e-9))

    U, V, W = build_matrices(terms)
    base_rank = len(terms)
    if not verify(U, V, W):
        # Fall back to schoolbook
        rs = n * m * p
        U = np.zeros((n * m, rs)); V = np.zeros((m * p, rs)); W = np.zeros((p * n, rs))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1; V[j * p + k, col] = 1; W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the 72xR coefficient matrix: each tensor entry is a linear form in the
    # term coefficients (1 if the term contributes to that entry, else 0).
    def coef_matrix(term_list):
        R = len(term_list)
        M = np.zeros((n * m * p, R))
        for c, (U_d, V_d, W_d) in enumerate(term_list):
            for (i, j) in U_d:
                for (k) in range(p):
                    if (j, k) in V_d and (k, i) in W_d:
                        M[i * m * p + j * p + k, c] = 1.0
        return M

    def try_reduce(term_list):
        # Greedy: find a column that is a linear combination of the others in the
        # tensor space (i.e., dropping it still allows a completion). We test by
        # checking whether the column space of the remaining columns still spans
        # the target tensor. Equivalent: is the target in colspace(M[:, drop])?
        target = np.ones(n * m * p)
        M = coef_matrix(term_list)
        R = len(term_list)
        removed = []
        # Try removing columns one at a time; keep if target remains in span.
        # Use least squares on the reduced matrix; residual ~0 means removable.
        order = list(range(R))
        for c in range(R - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            keep = [x for x in order if x != c]
            if not keep:
                break
            Msub = M[:, keep]
            # Solve Msub @ x = target
            try:
                x, res, rank_, sv = np.linalg.lstsq(Msub, target, rcond=None)
            except np.linalg.LinAlgError:
                continue
            residual = np.linalg.norm(Msub @ x - target)
            if residual < 1e-8:
                # column c is redundant; drop it
                order.remove(c)
                removed.append(c)
        return order, removed

    elapsed = time.time() - start_time
    if elapsed < deadline - 15 and base_rank > 60:
        order, removed = try_reduce(terms)
        if removed:
            new_terms = [terms[i] for i in order]
            U2, V2, W2 = build_matrices(new_terms)
            if verify(U2, V2, W2):
                U, V, W = U2, V2, W2

    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
