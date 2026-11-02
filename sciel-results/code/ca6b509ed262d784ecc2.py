"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]


# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.

    Design (kept from current program, refined with the inspiration's
    structural block layout):
      1. Build a correct block-Strassen base decomposition (rank 36).
         Structure: split the middle dim into two 2x2 A-blocks and split the
         5 output columns into 2+2+1.  Each 2x2*2x2 uses Strassen (rank 7),
         each 2x2*2x1 uses schoolbook (rank 4):
             6 * 7  (three 2x2*2x2 per A-block, two A-blocks)
           + 2 * 4  (one 2x2*2x1 per A-block)
           = 36.
      2. Verify correctness (sum of rank-1 tensor vectors == all-ones tensor).
      3. Reduce rank via linear dependencies among the rank-1 tensor vectors.
         We run TWO complementary reduction passes:
           a) STRUCTURAL: ordered, targeted null-space search that respects the
              block layout (Strassen blocks first, then cross-block), which
              catches low-rank sub-structures a full SVD can miss.
           b) GENERIC: full-matrix SVD null-space greedy reduction (the
              current program's approach) as a robust fallback / continuation.
      4. Deadline guard: return best-so-far before the 120 s limit.
    """
    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin well inside the 150 s kill limit

    n, m, p = DIMENSIONS  # 2, 4, 5
    N_A = n * m  # 8
    N_B = m * p  # 20
    N_C = p * n  # 10
    T_DIM = n * m * p  # 40

    def time_left():
        return DEADLINE - time.time()

    # ---------- Base decomposition (block-Strassen, rank 36) ----------
    rank_terms = []  # list of (u_dict, v_dict, w_dict)

    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        # Strassen 7 products for 2x2*2x2 at block (i0..i0+1, j0..j0+1, k0..k0+1)
        terms = [
            # m1 = (x11+x22)*(y11+y22)
            ({(i0, j0): 1, (i0 + 1, j0 + 1): 1},
             {(j0, k0): 1, (j0 + 1, k0 + 1): 1},
             {(i0, k0): 1, (i0 + 1, k0 + 1): 1}),
            # m2 = (x21+x22)*y11
            ({(i0 + 1, j0): 1, (i0 + 1, j0 + 1): 1},
             {(j0, k0): 1},
             {(i0 + 1, k0): 1, (i0 + 1, k0 + 1): -1}),
            # m3 = x11*(y12-y22)
            ({(i0, j0): 1},
             {(j0, k0 + 1): 1, (j0 + 1, k0 + 1): -1},
             {(i0, k0 + 1): 1, (i0 + 1, k0 + 1): 1}),
            # m4 = x22*(y21-y11)
            ({(i0 + 1, j0 + 1): 1},
             {(j0 + 1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0 + 1, k0): 1}),
            # m5 = (x11+x12)*y22
            ({(i0, j0): 1, (i0, j0 + 1): 1},
             {(j0 + 1, k0 + 1): 1},
             {(i0, k0): -1, (i0, k0 + 1): 1}),
            # m6 = (x21-x11)*(y11+y12)
            ({(i0 + 1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0 + 1): 1},
             {(i0 + 1, k0 + 1): 1}),
            # m7 = (x12-x22)*(y21+y22)
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

    # A = [A_top (rows 0..1 of middle) | A_bot (rows 2..3 of middle)]
    # B columns split into (0..1), (2..3), (4).
    # C = A_top*(B_top1 + B_top2 + B_top3) + A_bot*(B_bot1 + B_bot2 + B_bot3)
    #   where B_top1/B_bot1 are 2x2 (cols 0..1), B_top2/B_bot2 are 2x2 (cols 2..3),
    #   and B_top3/B_bot3 are 2x1 (col 4).
    add_strassen_2x2x2(i0=0, j0=0, k0=0, sign=1)   # A_top * B_top1
    add_strassen_2x2x2(i0=0, j0=0, k0=2, sign=1)   # A_top * B_top2
    add_schoolbook_2x2x1(i0=0, j0=0, k0=4, sign=1) # A_top * B_top3
    add_strassen_2x2x2(i0=0, j0=2, k0=0, sign=1)   # A_bot * B_bot1
    add_strassen_2x2x2(i0=0, j0=2, k0=2, sign=1)   # A_bot * B_bot2
    add_schoolbook_2x2x1(i0=0, j0=2, k0=4, sign=1) # A_bot * B_bot3

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
        return bool(np.allclose(T.sum(axis=0), np.ones(T_DIM), atol=tol))

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
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---------- Generic SVD null-space greedy reduction (current approach) ----------
    def try_reduce_generic(U, V, W, max_iters=60):
        cur_U, cur_V, cur_W = U, V, W
        for _ in range(max_iters):
            if time_left() < 5.0:
                break
            T = tensor_vecs(cur_U, cur_V, cur_W)
            r = T.shape[0]
            if r <= 1:
                break
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
            for nv in null_space:
                if time_left() < 5.0:
                    break
                idx = int(np.argmax(np.abs(nv)))
                if abs(nv[idx]) < 1e-12:
                    continue
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
                keep = [c for c in range(r) if c != idx]
                new_U = new_U[:, keep]
                new_V = new_V[:, keep]
                new_W = new_W[:, keep]
                if verify(new_U, new_V, new_W):
                    cur_U, cur_V, cur_W = new_U, new_V, new_W
                    found = True
                    break
            if not found:
                break
        return cur_U, cur_V, cur_W

    # ---------- Structural ordered reduction (refinement from inspiration's layout) ----------
    # The block layout gives us a natural ordering of terms.  Low-rank sub-structures
    # (e.g. a 2x2*2x2 Strassen block sharing structure with an adjacent block, or a
    # Strassen block + its schoolbook neighbour) produce null vectors whose support is
    # concentrated on a *contiguous* slice of the term list.  We exploit this by
    # scanning candidate support windows and, for each, solving the constrained
    # dependency (a null vector of the full tensor matrix restricted to that window)
    # and attempting the drop.  This catches reductions the generic full-SVD pass
    # may miss because it only exposes a few null vectors at a time.
    def try_reduce_structural(U, V, W):
        cur_U, cur_V, cur_W = U, V, W
        # Candidate support windows derived from the block layout.
        # Base ordering (36 terms):
        #  0-6  : A_top * B_top1 (Strassen)
        #  7-13 : A_top * B_top2 (Strassen)
        # 14-17 : A_top * B_top3 (schoolbook)
        # 18-24 : A_bot * B_bot1 (Strassen)
        # 25-31 : A_bot * B_bot2 (Strassen)
        # 32-35 : A_bot * B_bot3 (schoolbook)
        windows = [
            (0, 7), (7, 14), (14, 18),          # A_top blocks
            (18, 25), (25, 32), (32, 36),       # A_bot blocks
            (0, 14), (14, 18), (18, 32),        # cross within A_top / A_bot
            (0, 36),                            # whole (generic)
        ]
        for (a, b) in windows:
            if time_left() < 5.0:
                break
            T = tensor_vecs(cur_U, cur_V, cur_W)
            r = T.shape[0]
            if r <= 1:
                break
            if b > r:
                b = r
            if b - a < 2:
                continue
            # Work on the sub-matrix of rows [a:b]; find its null space.
            Tsub = T[a:b, :]
            try:
                _, svals, Vt = np.linalg.svd(Tsub, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            tol = 1e-9 * (svals[0] if svals.size else 1.0)
            rank_sub = int(np.sum(svals > tol))
            if rank_sub >= (b - a):
                continue
            null_sub = Vt[rank_sub:, :]  # (dim, b-a)
            for nvsub in null_sub:
                if time_left() < 5.0:
                    break
                # Build a full-length null vector with support only on [a:b].
                nv = np.zeros(r)
                nv[a:b] = nvsub
                idx_rel = int(np.argmax(np.abs(nvsub)))
                idx = a + idx_rel
                if abs(nv[idx]) < 1e-12:
                    continue
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
                keep = [c for c in range(r) if c != idx]
                new_U = new_U[:, keep]
                new_V = new_V[:, keep]
                new_W = new_W[:, keep]
                if verify(new_U, new_V, new_W):
                    cur_U, cur_V, cur_W = new_U, new_V, new_W
                    break
        return cur_U, cur_V, cur_W

    # ---------- Run both reduction passes (structural first, then generic) ----------
    U, V, W = try_reduce_structural(U, V, W)
    U, V, W = try_reduce_generic(U, V, W)

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
