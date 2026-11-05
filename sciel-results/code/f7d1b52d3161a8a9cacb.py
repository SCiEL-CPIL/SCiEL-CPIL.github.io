"""Matrix multiplication 4,4,9 tensor decomposition.

Family: Strassen block construction (rank-114 nominal) + rank-reduction search.

Design (combined):
  * Structural idea: build the 4x4*4x9 product as two rank-49 Strassen 4x4*4x4
    blocks (columns 0..8) plus a rank-16 schoolbook block (column 8). Nominal rank 114.
  * Refinement: detect null columns (linear dependencies among the 114 rank-1 terms)
    numerically and drop them. Then search for additional independent null vectors
    with a smooth least-squares objective. Each independent null vector lowers rank by one.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]


def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS  # 4, 4, 9

    n_dim = n * m      # 16
    m_dim = m * p      # 36
    p_dim = p * n      # 36

    # ---- Target tensor (the matrix-multiplication tensor) --------------------
    target = np.zeros(n_dim * m_dim * p_dim, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx * (m_dim * p_dim) + b_idx * p_dim + c_idx] = 1.0
    T = target.reshape(n_dim * m_dim, p_dim)  # shape (576, 36)

    def verify_decomposition(U, V, W):
        return np.allclose(np.einsum('ar,br,cr->abc', U, V, W).flatten(),
                           target, atol=1e-9)

    # ---- Strassen 2x2x2 base: 7 rank-1 terms ---------------------------------
    strassen_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]

    def build_strassen_4x4(A_i_base, A_j_base, B_j_base, B_k_base,
                           C_i_base, C_k_base):
        """7x7 = 49 terms for one 4x4*4x4 subproblem (two-level Strassen)."""
        terms = []

        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                for (loc_i, loc_j), coeff in u_c.items():
                    for (br, bc, bc_coeff) in A_blocks:
                        gi = A_i_base + br * 2 + loc_i
                        gj = A_j_base + bc * 2 + loc_j
                        u_vec[gi * m + gj] += coeff * bc_coeff
                for (loc_j, loc_k), coeff in v_c.items():
                    for (br, bc, bc_coeff) in B_blocks:
                        gj = B_j_base + br * 2 + loc_j
                        gk = B_k_base + bc * 2 + loc_k
                        v_vec[gj * p + gk] += coeff * bc_coeff
                for (loc_i, loc_k), coeff in w_c.items():
                    for (br, bc, bc_coeff) in C_blocks:
                        gi = C_i_base + br * 2 + loc_i
                        gk = C_k_base + bc * 2 + loc_k
                        w_vec[gk * n + gi] += coeff * bc_coeff
                terms.append((u_vec, v_vec, w_vec))

        add_inner([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)])
        add_inner([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)])
        add_inner([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)])
        add_inner([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)])
        add_inner([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)])
        add_inner([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)])
        add_inner([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)])
        return terms

    # ---- Assemble the nominal rank-114 solution ------------------------------
    all_terms = []
    all_terms.extend(build_strassen_4x4(0, 0, 0, 0, 0, 0))   # cols 0..3
    all_terms.extend(build_strassen_4x4(0, 0, 0, 4, 0, 4))   # cols 4..8
    for i in range(n):
        for j in range(m):
            u_vec = [0] * (n * m); v_vec = [0] * (m * p); w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            all_terms.append((u_vec, v_vec, w_vec))

    N = len(all_terms)
    U_full = np.zeros((n_dim, N))
    V_full = np.zeros((m_dim, N))
    W_full = np.zeros((p_dim, N))
    for r, (u, v, w) in enumerate(all_terms):
        U_full[:, r] = u
        V_full[:, r] = v
        W_full[:, r] = w

    def make_result(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Null-vector detection (smooth, numerical) ---------------------------
    # A null vector c satisfies [UV] c = 0 and [UW] c = 0 (then the rank-1 term
    # sum_r c_r U[:,r] V[:,r] W[:,r] == 0, so dropping it is exact).
    UV = np.einsum('ar,br->abr', U_full, V_full).reshape(n_dim * m_dim, N)
    UW = np.einsum('ar,cr->acr', U_full, W_full).reshape(n_dim * p_dim, N)
    M = np.vstack([UV, UW])  # (936, 114)

    def null_candidates(mat, tol):
        """Return indices whose unit column is (near-)null, in ascending-norm order."""
        nrm = np.linalg.norm(mat, axis=0)
        order = np.argsort(nrm)
        cands = []
        for j in order:
            if nrm[j] <= tol:
                cands.append(int(j))
        return cands

    # Detect null columns with a conservative tolerance.
    tol0 = 1e-6 * max(1.0, np.max(np.linalg.norm(M, axis=0)))
    null_cols = null_candidates(M, tol0)
    # Keep a greedy linearly-independent subset of the null vectors.
    keep = []
    kept_mat = None
    for j in null_cols:
        col = M[:, j].copy()
        if kept_mat is None:
            keep.append(j); kept_mat = col.reshape(-1, 1)
            continue
        # project out
        proj = kept_mat @ (kept_mat.T @ col) / (kept_mat.T @ kept_mat).diagonal()
        res = col - proj
        if np.linalg.norm(res) > 1e-4:
            keep.append(j); kept_mat = np.hstack([kept_mat, res.reshape(-1, 1)])

    if len(keep) >= 2:  # at least a couple of nulls -> construction is dependent
        idx = [r for r in range(N) if r not in keep]
        U0, V0, W0 = U_full[:, idx], V_full[:, idx], W_full[:, idx]
        if verify_decomposition(U0, V0, W0):
            base_rank = len(idx)
        else:
            base_rank = N
            U0, V0, W0 = U_full.copy(), V_full.copy(), W_full.copy()
    else:
        base_rank = N
        U0, V0, W0 = U_full.copy(), V_full.copy(), W_full.copy()

    best = (U0.copy(), V0.copy(), W0.copy(), base_rank)

    # ---- Search for additional independent null vectors (smooth refinement) ---
    # For a set S of columns, find a non-trivial c in span(S) with M[:,S] c ~ 0.
    # Minimise ||M[:,S] c|| (smooth) subject to ||c||=1 via the smallest
    # singular vector; accept if the residual is below tolerance.
    def refine_nulls(U, V, W, rank, max_extra):
        UV = np.einsum('ar,br->abr', U, V).reshape(n_dim * m_dim, rank)
        UW = np.einsum('ar,cr->acr', U, W).reshape(n_dim * p_dim, rank)
        M2 = np.vstack([UV, UW])
        scale = max(1.0, np.max(np.linalg.norm(M2, axis=0)))
        tol = 1e-8 * scale
        accepted = []
        acc_mat = None
        # greedy: repeatedly find the most-null direction among columns
        remaining = list(range(rank))
        for _ in range(max_extra):
            if time.time() > deadline - 20:
                break
            best_j, best_res, best_vec = None, None, None
            for j in remaining:
                sub = M2[:, [j] + accepted] if accepted else M2[:, [j]]
                # smallest singular vector of sub (direction in its column span)
                try:
                    _, s, vh = np.linalg.svd(sub, full_matrices=False)
                    c = vh[-1]
                except Exception:
                    continue
                res = np.linalg.norm(M2[:, [j] + accepted] @ c) if accepted \
                    else np.linalg.norm(M2[:, j] * c)
                if best_res is None or res < best_res:
                    best_res, best_j, best_vec = res, j, c
            if best_j is None:
                break
            if best_res <= tol:
                # add the new null column j to the accepted set
                newc = best_vec.copy()
                if acc_mat is None:
                    accepted.append(best_j); acc_mat = newc.reshape(-1, 1)
                else:
                    proj = acc_mat @ (acc_mat.T @ newc) / (acc_mat.T @ acc_mat).diagonal()
                    resv = newc - proj
                    if np.linalg.norm(resv) > 1e-4:
                        accepted.append(best_j); acc_mat = np.hstack([acc_mat, resv.reshape(-1, 1)])
            else:
                break
        if not accepted:
            return U, V, W, rank
        idx = [r for r in range(rank) if r not in accepted]
        Un, Vn, Wn = U[:, idx], V[:, idx], W[:, idx]
        if verify_decomposition(Un, Vn, Wn):
            return Un, Vn, Wn, len(idx)
        return U, V, W, rank

    # Try to push the rank below the base value.
    if base_rank < N and time.time() < deadline - 25:
        U1, V1, W1, r1 = refine_nulls(U0, V0, W0, base_rank, max_extra=12)
        if r1 < best[3]:
            best = (U1, V1, W1, r1)

    # ---- Final safety: always return a valid decomposition --------------------
    Uf, Vf, Wf, rf = best
    if not verify_decomposition(Uf, Vf, Wf):
        # fall back to the nominal construction, then to pure schoolbook
        if verify_decomposition(U_full, V_full, W_full):
            return make_result(U_full, V_full, W_full)
        rank_sb = n * m * p
        U_sb = np.zeros((n_dim, rank_sb)); V_sb = np.zeros((m_dim, rank_sb)); W_sb = np.zeros((p_dim, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return make_result(U_sb, V_sb, W_sb)

    return make_result(Uf, Vf, Wf)
