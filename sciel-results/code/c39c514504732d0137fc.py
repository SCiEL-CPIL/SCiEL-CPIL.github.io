"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]


# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Strategy:
      1. Build multiple candidate base decompositions (different block splits).
      2. For each, verify correctness.
      3. Aggressively reduce rank via null-space analysis with multiple strategies.
      4. Return the best (lowest rank) verified decomposition.
    """
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 5
    N_A = n * m  # 8
    N_B = m * p  # 20
    N_C = p * n  # 10
    T_DIM = n * m * p  # 40

    def time_left():
        return DEADLINE - time.time()

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

    # ---------- Strassen 2x2x2 helper ----------
    def strassen_2x2x2_terms(i0, j0, k0, sign=1):
        """Return list of (u_dict, v_dict, w_dict) for Strassen 2x2*2x2."""
        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        return [({k: v*sign for k,v in u.items()},
                 {k: v*sign for k,v in v.items()},
                 {k: v*sign for k,v in w.items()}) for u,v,w in terms]

    def schoolbook_2x2x1_terms(i0, j0, k0, sign=1):
        """Return list of (u_dict, v_dict, w_dict) for schoolbook 2x2*2x1."""
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        return [({k: v*sign for k,v in u.items()},
                 {k: v*sign for k,v in v.items()},
                 {k: v*sign for k,v in w.items()}) for u,v,w in terms]

    # ---------- Build candidate decompositions ----------
    candidates = []

    # Candidate 1: Split columns as 2+2+1, rows as 2+2
    # C[:,0:2] = A_top*B_top[0:2] + A_bot*B_bot[0:2]  (2 Strassen = 14)
    # C[:,2:4] = A_top*B_top[2:4] + A_bot*B_bot[2:4]  (2 Strassen = 14)
    # C[:,4:5] = A_top*B_top[4] + A_bot*B_bot[4]       (2 schoolbook 2x2x1 = 8)
    terms = []
    terms += strassen_2x2x2_terms(0, 0, 0)
    terms += strassen_2x2x2_terms(0, 0, 2)
    terms += schoolbook_2x2x1_terms(0, 0, 4)
    terms += strassen_2x2x2_terms(0, 2, 0)
    terms += strassen_2x2x2_terms(0, 2, 2)
    terms += schoolbook_2x2x1_terms(0, 2, 4)
    U, V, W = build_matrices(terms)
    if verify(U, V, W):
        candidates.append((U, V, W))

    # Candidate 2: Split columns as 3+2, rows as 2+2
    # C[:,0:3] = A_top*B_top[0:3] + A_bot*B_bot[0:3]
    #   B_top[0:3] is 2x3 = 2x2 + 2x1, so Strassen(7) + schoolbook(4) = 11 per row-block
    #   Total for C[:,0:3]: 2*11 = 22
    # C[:,3:5] = A_top*B_top[3:5] + A_bot*B_bot[3:5]
    #   B_top[3:5] is 2x2, Strassen = 7 per row-block
    #   Total for C[:,3:5]: 2*7 = 14
    # Grand total: 22 + 14 = 36
    terms = []
    # C[:,0:3]: use Strassen for cols 0-1, schoolbook for col 2
    terms += strassen_2x2x2_terms(0, 0, 0)
    terms += schoolbook_2x2x1_terms(0, 0, 2)
    terms += strassen_2x2x2_terms(0, 2, 0)
    terms += schoolbook_2x2x1_terms(0, 2, 2)
    # C[:,3:5]: use Strassen for cols 3-4
    terms += strassen_2x2x2_terms(0, 0, 3)
    terms += strassen_2x2x2_terms(0, 2, 3)
    U, V, W = build_matrices(terms)
    if verify(U, V, W):
        candidates.append((U, V, W))

    # Candidate 3: Pure schoolbook (rank 40) as fallback
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((
                    {(i, j): 1},
                    {(j, k): 1},
                    {(i, k): 1},
                ))
    U, V, W = build_matrices(terms)
    if verify(U, V, W):
        candidates.append((U, V, W))

    if not candidates:
        # Should never happen
        U, V, W = build_matrices([({(0,0):1},{(0,0):1},{(0,0):1})])
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---------- Aggressive rank reduction ----------
    def try_reduce(U, V, W, max_iters=100):
        """Greedily reduce rank using null-space analysis."""
        cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
        for it in range(max_iters):
            if time_left() < 8.0:
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
                break  # full rank, no more reductions possible
            
            null_space = Vt[rank_T:, :]  # shape (r - rank_T, r)
            found = False
            
            # Try each null vector
            for nv in null_space:
                if time_left() < 8.0:
                    break
                abs_nv = np.abs(nv)
                # Try multiple pivot choices (top 5)
                top_pivots = np.argsort(abs_nv)[::-1][:5]
                for idx in top_pivots:
                    idx = int(idx)
                    if abs_nv[idx] < 1e-12:
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
                    if verify(new_U, new_V, new_W, tol=1e-8):
                        cur_U, cur_V, cur_W = new_U, new_V, new_W
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return cur_U, cur_V, cur_W

    # Try to reduce each candidate
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    for U, V, W in candidates:
        if time_left() < 15.0:
            break
        r0 = U.shape[1]
        if r0 >= best_rank:
            continue
        U_r, V_r, W_r = try_reduce(U, V, W)
        r_final = U_r.shape[1]
        if r_final < best_rank:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = r_final
    
    # If no reduction found, use the smallest candidate
    if best_U is None:
        for U, V, W in sorted(candidates, key=lambda x: x[0].shape[1]):
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                break
    
    # ---------- Final verify & return ----------
    if best_U is None or not verify(best_U, best_V, best_W):
        # Ultimate fallback
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(i, k): 1},
                    ))
        best_U, best_V, best_W = build_matrices(terms)

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
