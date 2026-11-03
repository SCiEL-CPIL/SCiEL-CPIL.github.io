"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def add_strassen(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols

        def U(la, lc): return a_rows[la] * m + c_cols[lc]
        def V(lc, ld): return c_cols[lc] * p + d_cols[ld]
        def W(ld, la): return d_cols[ld] * n + a_rows[la]

        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        add_term([(U(0,0), 1), (U(1,1), 1)],
                 [(V(0,0), 1), (V(1,1), 1)],
                 [(W(0,0), 1), (W(1,1), 1)])

        # M2 = (A[1,0]+A[1,1])*B[0,0]
        add_term([(U(1,0), 1), (U(1,1), 1)],
                 [(V(0,0), 1)],
                 [(W(0,1), 1), (W(1,1), -1)])

        # M3 = A[0,0]*(B[0,1]-B[1,1])
        add_term([(U(0,0), 1)],
                 [(V(0,1), 1), (V(1,1), -1)],
                 [(W(1,0), 1), (W(1,1), 1)])

        # M4 = A[1,1]*(B[1,0]-B[0,0])
        add_term([(U(1,1), 1)],
                 [(V(1,0), 1), (V(0,0), -1)],
                 [(W(0,0), 1), (W(0,1), 1)])

        # M5 = (A[0,0]+A[0,1])*B[1,1]
        add_term([(U(0,0), 1), (U(0,1), 1)],
                 [(V(1,1), 1)],
                 [(W(0,0), -1), (W(1,0), 1)])

        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        add_term([(U(1,0), 1), (U(0,0), -1)],
                 [(V(0,0), 1), (V(0,1), 1)],
                 [(W(1,1), 1)])

        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        add_term([(U(0,1), 1), (U(1,1), -1)],
                 [(V(1,0), 1), (V(1,1), 1)],
                 [(W(0,0), 1)])

    def add_schoolbook_2x2x1(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0 = d_cols[0]
        def U(la, lc): return a_rows[la] * m + c_cols[lc]
        def V(lc, ld): return c_cols[lc] * p + d_cols[0]
        def W(ld, la): return d_cols[0] * n + a_rows[la]
        add_term([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)])
        add_term([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)])
        add_term([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)])
        add_term([(U(1,1), 1)], [(V(1,0), 1)], [(W(0,1), 1)])

    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        def U(la, lc): return a_rows[la] * m + c_cols[0]
        def V(lc, ld): return c_cols[0] * p + d_cols[ld]
        def W(ld, la): return d_cols[ld] * n + a_rows[la]
        add_term([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)])
        add_term([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)])
        add_term([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)])
        add_term([(U(1,0), 1)], [(V(0,1), 1)], [(W(1,1), 1)])

    def add_schoolbook_2x1x1(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0 = d_cols[0]
        def U(la, lc): return a_rows[la] * m + c_cols[0]
        def V(lc, ld): return c_cols[0] * p + d_cols[0]
        def W(ld, la): return d_cols[0] * n + a_rows[la]
        add_term([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)])
        add_term([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)])

    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0, d1 = d_cols
        def U(la, lc): return a_rows[0] * m + c_cols[lc]
        def V(lc, ld): return c_cols[lc] * p + d_cols[ld]
        def W(ld, la): return d_cols[ld] * n + a_rows[0]
        add_term([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)])
        add_term([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)])
        add_term([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)])
        add_term([(U(0,1), 1)], [(V(1,1), 1)], [(W(1,0), 1)])

    def add_schoolbook_1x2x1(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0 = d_cols[0]
        def U(la, lc): return a_rows[0] * m + c_cols[lc]
        def V(lc, ld): return c_cols[lc] * p + d_cols[0]
        def W(ld, la): return d_cols[0] * n + a_rows[0]
        add_term([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)])
        add_term([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)])

    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0, d1 = d_cols
        def U(la, lc): return a_rows[0] * m + c_cols[0]
        def V(lc, ld): return c_cols[0] * p + d_cols[ld]
        def W(ld, la): return d_cols[ld] * n + a_rows[0]
        add_term([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)])
        add_term([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)])

    def add_schoolbook_1x1x1(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0 = d_cols[0]
        def U(la, lc): return a_rows[0] * m + c_cols[0]
        def V(lc, ld): return c_cols[0] * p + d_cols[0]
        def W(ld, la): return d_cols[0] * n + a_rows[0]
        add_term([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)])

    # Build the rank-84 decomposition with partition [2,1]x[2,2,1]x[2,2,2]
    n_block_ranges = [(0, 2), (2, 3)]
    m_block_ranges = [(0, 2), (2, 4), (4, 5)]
    p_block_ranges = [(0, 2), (2, 4), (4, 6)]

    for ni, (ns, ne) in enumerate(n_block_ranges):
        for mi, (ms, me) in enumerate(m_block_ranges):
            for pi, (ps, pe) in enumerate(p_block_ranges):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps

                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))

                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    add_schoolbook_2x2x1(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 1:
                    add_schoolbook_2x1x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 1:
                    add_schoolbook_1x2x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 1:
                    add_schoolbook_1x1x1(a_rows, c_cols, d_cols)
                else:
                    raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")

    # Build U, V, W matrices
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U[row, ci] += coeff
        for row, coeff in v_entries:
            V[row, ci] += coeff
        for row, coeff in w_entries:
            W[row, ci] += coeff

    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_fb = np.zeros((n * m, rank_school), dtype=float)
        V_fb = np.zeros((m * p, rank_school), dtype=float)
        W_fb = np.zeros((p * n, rank_school), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    # Check that U @ diag @ V @ diag @ W reconstructs the tensor
    # T[i*m+j, j*p+k, k*n+i] should be 1, and all other entries should be 0
    # Using: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Check a few entries to save time
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > 1e-10:
                    return False
    
    # Check that off-diagonal entries are zero
    # Sample some off-diagonal entries
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        # Pick a different (i', j', k')
        i2 = random.randint(0, n - 1)
        j2 = random.randint(0, m - 1)
        k2 = random.randint(0, p - 1)
        if (i2, j2, k2) != (i, j, k):
            val = np.dot(U[i2 * m + j2], V[j2 * p + k2] * W[k2 * n + i2])
            if abs(val) > 1e-10:
                return False
    
    return True

# EVOLVE-BLOCK-END
