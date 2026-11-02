"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]


# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.

    Strategy:
      1. Build a correct base decomposition (block-Strassen, rank 36).
      2. Verify correctness (sum of rank-1 tensor vectors == all-ones tensor).
      3. Exploit linear dependencies among the 36 rank-1 tensor vectors:
         if the 36 vectors are linearly dependent, we can rescale a subset
         (absorbing the scale into U/V/W) to drop one term. Repeat greedily.
      4. Try a "cross-block" merge: combine two terms that share structure
         across the left/right A-block split to see if a rank reduction is
         possible.
      5. Deadline guard: return best-so-far before the 120s limit.
    """
    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin

    n, m, p = DIMENSIONS  # 2, 4, 5
    N_A = n * m  # 8
    N_B = m * p  # 20
    N_C = p * n  # 10
    T_DIM = n * m * p  # 40

    def time_left():
        return DEADLINE - time.time()

    # ---------- Build base decomposition (block-Strassen, rank 36) ----------
    rank_terms = []  # list of (u_dict, v_dict, w_dict)

    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        terms = [
            ({(i0, j0): 1, (i0 + 1, j0 + 1): 1},
             {(j0, k0): 1, (j0 + 1, k0 + 1): 1},
             {(i0, k0): 1, (i0 + 1, k0 + 1): 1}),
            ({(i0 + 1, j0): 1, (i0 + 1, j0 + 1): 1},
             {(j0, k0): 1},
             {(i0 + 1, k0): 1, (i0 + 1, k0 + 1): -1}),
            ({(i0, j0): 1},
             {(j0, k0 + 1): 1, (j0 + 1, k0 + 1): -1},
             {(i0, k0 + 1): 1, (i0 + 1, k0 + 1): 1}),
            ({(i0 + 1, j0 + 1): 1},
             {(j0 + 1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0 + 1, k0): 1}),
            ({(i0, j0): 1, (i0, j0 + 1): 1},
             {(j0 + 1, k0 + 1): 1},
             {(i0, k0): -1, (i0, k0 + 1): 1}),
            ({(i0 + 1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0 + 1): 1},
             {(i0 + 1, k0 + 1): 1}),
            ({(i0, j0 + 1): 1, (i0 + 1, j0 + 1): -1},
             {(j0 + 1, k0): 1, (j0 + 1, k0 + 1): 1},
             {(i0, k0): 1}),
        ]
        for u, v, w in terms:
            rank_terms.append((
                {k: val * sign for k, val in u.items()},
                {k: val * sign for k, val in v.items()},
                {k: val * sign for k, val in w.items()},
            ))

    def add_schoolbook_2x2x1(i0, j0, k0, sign=1):
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0 + 1): 1}, {(j0 + 1, k0): 1}, {(i0, k0): 1}),
            ({(i0 + 1, j0): 1}, {(j0, k0): 1}, {(i0 + 1, k0): 1}),
            ({(i0 + 1, j0 + 1): 1}, {(j0 + 1, k0): 1}, {(i0 + 1, k0): 1}),
        ]
        for u, v, w in terms:
            rank_terms.append((
                {k: val * sign for k, val in u.items()},
                {k: val * sign for k, val in v.items()},
                {k: val * sign for k, val in w.items()},
            ))

    # A = [A1(2x2) | A2(2x2)], B = [Btop(2x5); Bbot(2x5)]
    # C = A1*Btop + A2*Bbot
    add_strassen_2x2x2(i0=0, j0=0, k0=0, sign=1)
    add_strassen_2x2x2(i0=0, j0=0, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=0, k0=4, sign=1)
    add_strassen_2x2x2(i0=0, j0=2, k0=0, sign=1)
    add_strassen_2x2x2(i0=0, j0=2, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=2, k0=4, sign=1)

    def build_matrices(terms):
        r = len(terms)
        U = np.zeros((N_A, r), dtype=float)
        V = np.zeros((N_B, r), dtype=float)
        W = np.zeros((N_C, r), dtype=float)
        for c, (ud, vd, wd) in enumerate(terms):
            for (i, j), coeff in ud.items():
                U[i * m + j, c] = coeff
            for (j, k), coeff in vd.items():
                V[j * p + k, c] = coeff
            for (i, k), coeff in wd.items():
                W[k * n + i, c] = coeff
        return U, V, W

    U, V, W = build_matrices(rank_terms)

    # ---------- Helpers ----------
    def tensor_vecs(U, V, W):
        r = U.shape[1]
        T = np.zeros((r, T_DIM), dtype=float)
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n):
                for j in range(m):
                    a = uc[i * m + j]
                    if a == 0:
                        continue
                    for k in range(p):
                        b = vc[j * p + k]
                        if b == 0:
                            continue
                        d = wc[k * n + i]
                        if d == 0:
                            continue
                        T[c, i * (m * p) + j * p + k] = a * b * d
        return T

    def verify(U, V, W, tol=1e-7):
        T = tensor_vecs(U, V, W)
        return np.allclose(T.sum(axis=0), np.ones(T_DIM), atol=tol)

    # ---------- Verify base ----------
    if not verify(U, V, W):
        # Fallback: pure schoolbook rank 40 (guaranteed correct)
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((
                        {(i, j): 1}, {(j, k): 1}, {(i, k): 1},
                    ))
        U, V, W = build_matrices(terms)
        if not verify(U, V, W):
            # Should never happen; return what we have
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---------- Linear-dependency reduction (enhanced) ----------
    def try_reduce(U, V, W, max_iters=60):
        """Greedily drop terms by rescaling a subset to satisfy a null vector.
        Enhanced: try multiple pivot choices and verify with tighter tolerance."""
        cur_U, cur_V, cur_W = U, V, W
        for _ in range(max_iters):
            if time_left() < 5.0:
                break
            T = tensor_vecs(cur_U, cur_V, cur_W)
            r = T.shape[0]
            if r <= 1:
                break
            # Null space of T (r x T_DIM). Null vectors have length r.
            try:
                _, svals, Vt = np.linalg.svd(T, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            tol = 1e-9 * (svals[0] if svals.size else 1.0)
            rank_T = int(np.sum(svals > tol))
            if rank_T >= r:
                break  # independent, no more reductions
            null_space = Vt[rank_T:, :]  # (dim, r)
            found = False
            # Try all null vectors, and for each, try the top-3 pivot choices
            for nv in null_space:
                if time_left() < 5.0:
                    break
                # Get top-3 pivot candidates
                abs_nv = np.abs(nv)
                top_pivots = np.argsort(abs_nv)[::-1][:3]
                for idx in top_pivots:
                    idx = int(idx)
                    if abs_nv[idx] < 1e-12:
                        continue
                    # Build new scale factors
                    new_U = cur_U.copy()
                    new_V = cur_V.copy()
                    new_W = cur_W.copy()
                    for c in range(r):
                        if c == idx:
                            continue
                        s = 1.0 + nv[c] / nv[idx]
                        if abs(s) < 1e-12:
                            continue
                        new_U[:, c] *= s
                        new_V[:, c] *= s
                        new_W[:, c] *= s
                    # Drop column idx
                    keep = [c for c in range(r) if c != idx]
                    new_U = new_U[:, keep]
                    new_V = new_V[:, keep]
                    new_W = new_W[:, keep]
                    if verify(new_U, new_V, new_W, tol=1e-8):
                        cur_U, cur_V, cur_W = new_U, new_V, new_W
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return cur_U, cur_V, cur_W

    U, V, W = try_reduce(U, V, W, max_iters=60)

    # ---------- Winograd FTT over Z5: exploit output-column structure ----------
    # For p=5, use the 5-point cyclic structure. The key idea: instead of treating
    # each of the 5 output columns independently, use a transform to share work.
    # 
    # The 5-point DFT has the property that omega^5 = 1 where omega = e^(2*pi*i/5).
    # We can't use complex numbers, but we can use the rational factorization of
    # x^5 - 1 = (x-1)(x^4+x^3+x^2+x+1).
    # 
    # For the 4x5 B matrix, decompose B = B_0 * 1 + B_1 * x + B_2 * x^2 + B_3 * x^3 + B_4 * x^4
    # where x is a formal variable. Then C = A * B mod (x^5 - 1).
    # 
    # Using the factorization x^5 - 1 = (x-1)(x^4+x^3+x^2+x+1), we can write:
    # B = B_eval(1) * 1 + R(x) * (x^4+x^3+x^2+x+1) where R is degree < 4.
    # But this doesn't directly help with rank reduction.
    #
    # Alternative: use the Chinese Remainder Theorem on Z[x]/(x^5-1).
    # x^5 - 1 = (x-1)(x^4+x^3+x^2+x+1) over Q.
    # The factor x^4+x^3+x^2+x+1 is irreducible over Q (cyclotomic polynomial Phi_5).
    # So Z[x]/(x^5-1) ~= Q x Z[x]/(Phi_5(x)).
    # 
    # This means we can evaluate at x=1 (giving sum of columns) and in the
    # 4-dimensional field Q[x]/(Phi_5(x)). The latter requires working with
    # elements of the form a0 + a1*x + a2*x^2 + a3*x^3 mod Phi_5(x).
    # 
    # For our purposes, the key observation: the multiplication A*B can be
    # decomposed into:
    #   C = C_eval1 * 1 + C_field * Phi_5(x)  (in the polynomial ring)
    # where C_eval1 = A * (sum of B columns) is a 2x1 vector, and C_field
    # involves the 4-dimensional field multiplication.
    #
    # This gives a decomposition where the "x=1" part is rank 4 (2x4 * 4x1)
    # and the field part has some rank. But the field part is over a 4-dim
    # extension, so it's not directly rank-reducing.
    #
    # However, we can use this structure to find linear dependencies. Let's
    # try a more direct approach: use the fact that the 5 output columns
    # satisfy a linear recurrence (since they come from a 4x5 matrix times
    # a 2x4 matrix, the output has at most 4 degrees of freedom per row).
    #
    # For now, let's do an enhanced null-space search with the specific
    # block structure in mind.
    if time_left() > 15.0:
        T = tensor_vecs(U, V, W)
        r = T.shape[0]
        if r > 1:
            try:
                _, svals, Vt = np.linalg.svd(T, full_matrices=False)
            except np.linalg.LinAlgError:
                svals, Vt = None, None
            if svals is not None:
                tol = 1e-9 * (svals[0] if svals.size else 1.0)
                rank_T = int(np.sum(svals > tol))
                if rank_T < r:
                    null_space = Vt[rank_T:, :]
                    # Try each null vector; prefer ones with large entries in both
                    # the left-block terms (indices 0..10) and right-block terms (indices 14..24)
                    for nv in null_space:
                        if time_left() < 5.0:
                            break
                        # Check if this null vector spans both blocks
                        left_support = np.sum(np.abs(nv[:11]) > 1e-10)
                        right_support = np.sum(np.abs(nv[14:25]) > 1e-10)
                        if left_support < 2 or right_support < 2:
                            continue
                        # Try top-3 pivot choices
                        abs_nv = np.abs(nv)
                        top_pivots = np.argsort(abs_nv)[::-1][:3]
                        for idx in top_pivots:
                            idx = int(idx)
                            if abs_nv[idx] < 1e-12:
                                continue
                            new_U = U.copy()
                            new_V = V.copy()
                            new_W = W.copy()
                            for c in range(r):
                                if c == idx:
                                    continue
                                s = 1.0 + nv[c] / nv[idx]
                                if abs(s) < 1e-12:
                                    continue
                                new_U[:, c] *= s
                                new_V[:, c] *= s
                                new_W[:, c] *= s
                            keep = [c for c in range(r) if c != idx]
                            new_U = new_U[:, keep]
                            new_V = new_V[:, keep]
                            new_W = new_W[:, keep]
                            if verify(new_U, new_V, new_W, tol=1e-8):
                                U, V, W = new_U, new_V, new_W
                                break
                        else:
                            continue
                        break

    # ---------- Final verify & return ----------
    if not verify(U, V, W):
        # Rebuild a guaranteed-correct schoolbook fallback
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        U, V, W = build_matrices(terms)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
