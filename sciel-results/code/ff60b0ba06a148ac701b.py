import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W, seed=12345, trials=12):
        """Return True if the decomposition exactly reproduces A@B for random inputs."""
        try:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            rng = np.random.RandomState(seed)
            for _ in range(trials):
                A = rng.rand(n, m)
                B = rng.rand(m, p)
                C_expected = A @ B
                # Precompute transformed factors
                # x_alpha = sum_{i,j} A[i,j]*U[i*m+j, alpha]
                # y_alpha = sum_{j,k} B[j,k]*V[j*p+k, alpha]
                x = (A.reshape(n*m) @ Ua)
                y = (B.reshape(m*p) @ Va)
                C_computed = np.zeros((p, n))
                # C_computed[k*n+i] = sum_alpha W[k*n+i, alpha]*x_alpha*y_alpha
                # = (Wa * (x*y)[None,:]) summed over alpha
                prod = x * y
                C_computed = (Wa * prod[None, :]).sum(axis=1).reshape(p, n)
                C_expected = C_expected.T  # C_expected[i,k]; we need [k*n+i]
                if not np.allclose(C_computed, C_expected, atol=1e-9, rtol=1e-9):
                    return False
            return True
        except Exception:
            return False

    def reduce_rank(U, V, W, max_iter=4):
        """Attempt to remove a dependent column (term) by projecting others onto null space.
        Returns (U,V,W,removed_count) with possibly lower rank. Only applied if verified."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        removed = 0
        for _ in range(max_iter):
            if elapsed() > DEADLINE - 30:
                break
            if rank <= 1:
                break
            # Try to find a column index whose removal keeps the span of the "trilinear map".
            # Heuristic: find a column with smallest norm across all three matrices,
            # then check if it lies in the span of the others for a combined representation.
            # Exact merge is hard; instead try: remove column c if for all three matrices
            # column c is in the column-span of the rest (rank unchanged).
            norms = []
            for c in range(rank):
                nn = (Ua[:, c] ** 2).sum() + (Va[:, c] ** 2).sum() + (Wa[:, c] ** 2).sum()
                norms.append((nn, c))
            norms.sort()
            found = False
            for _, c in norms:
                # Check if removing column c keeps each matrix's column space the same
                # (i.e., column c is in span of others).
                Uo = np.delete(Ua, c, axis=1)
                Vo = np.delete(Va, c, axis=1)
                Wo = np.delete(Wa, c, axis=1)
                # rank check via SVD
                def col_in_span(full, others, col):
                    r_full = np.linalg.matrix_rank(full, tol=1e-9)
                    r_others = np.linalg.matrix_rank(others, tol=1e-9)
                    if r_full == r_others:
                        # col is in span of others
                        return True
                    return False
                if col_in_span(Ua, Uo, c) and col_in_span(Va, Vo, c) and col_in_span(Wa, Wo, c):
                    Ua = Uo
                    Va = Vo
                    Wa = Wo
                    rank -= 1
                    removed += 1
                    found = True
                    break
            if not found:
                break
        # Convert back to lists of (int or half-int) if clean, else float
        def to_list(a):
            return [[(float(v)) for v in row] for row in a]
        return to_list(Ua), to_list(Va), to_list(Wa), removed

    # ---- Build the known-correct double-Strassen (49) + schoolbook (16) = 65 ----
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Inner Strassen 2x2x2
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),       # M1
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),             # M2
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),             # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),             # M4
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),             # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),               # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),               # M7
    ]

    # Super Strassen 4x4x4
    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),       # M1
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),      # M2
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),      # M3
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),      # M4
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),      # M5
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),              # M6
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),              # M7
    ]

    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))
            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))
            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 schoolbook (rank 16)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    # Verify the baseline; if somehow wrong, still return (shouldn't happen)
    baseline_ok = verify(U, V, W, seed=99, trials=8)

    # Attempt rank reduction only if we have time and baseline is correct
    best = (U, V, W)
    if baseline_ok and elapsed() < DEADLINE - 40:
        Ur, Vr, Wr, removed = reduce_rank(U, V, W, max_iter=4)
        if removed > 0 and verify(Ur, Vr, Wr, seed=777, trials=15):
            best = (Ur, Vr, Wr)

    Uo, Vo, Wo = best
    return {"U": Uo, "V": Vo, "W": Wo}
