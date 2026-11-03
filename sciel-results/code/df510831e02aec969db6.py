import time
import numpy as np

DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 28, 14

    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # ------------------------------------------------------------------
    # Helper: verify exact decomposition (integer/half-integer safe)
    # ------------------------------------------------------------------
    def _verify(U, V, W, tol=1e-9):
        """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # ------------------------------------------------------------------
    # Helper: build schoolbook (rank 56)
    # ------------------------------------------------------------------
    def _build_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # ------------------------------------------------------------------
    # Helper: build Strassen block decomposition (rank 50)
    # 2x4x7 = 2 blocks of (2x2 x 2x7), each split into 3x(2x2x2) + 1x(2x2x1)
    # 2x2x2 Strassen rank 7, 2x2x1 schoolbook rank 2
    # Total: 2*(3*7 + 1*2) = 2*23 = 46? Wait: 3*7=21, +2=23, *2=46
    # Actually let me recount: 2 blocks, each has 3 sub-blocks of 2x2x2 (rank 7) 
    # and 1 sub-block of 2x2x1 (rank 2). So per block: 3*7+2 = 23. Total: 46.
    # ------------------------------------------------------------------
    def _build_strassen_block():
        # Strassen 2x2x2 (rank 7) - standard
        # U_s: 4x7, V_s: 4x7, W_s: 4x7
        # Indexing: U_s[i*2+j, r], V_s[j*2+k, r], W_s[k*2+i, r]
        U_s = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=np.int64)
        V_s = np.array([
            [0, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, -1, 0, 0, 1],
            [0, 0, -1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 1]
        ], dtype=np.int64)
        W_s = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=np.int64)

        # Verify Strassen block
        assert _verify(U_s, V_s, W_s, n=2, m=2, p=2), "Strassen 2x2x2 failed"

        # Schoolbook 2x2x1 (rank 2)
        # U: 4x2, V: 2x2, W: 2x2
        # T[i*2+j, j*1+0, 0*2+i] = 1 for i=0,1; j=0,1
        U_1 = np.zeros((4, 2), dtype=np.int64)
        V_1 = np.zeros((2, 2), dtype=np.int64)
        W_1 = np.zeros((2, 2), dtype=np.int64)
        for r in range(2):
            # i=r, j=0: U[r*2+0, r]=1, V[0, r]=1, W[0*2+r, r]=1
            # i=r, j=1: U[r*2+1, r]=1, V[1, r]=1, W[0*2+r, r]=1 (already set)
            U_1[r * 2 + 0, r] = 1
            U_1[r * 2 + 1, r] = 1
            V_1[0, r] = 1
            V_1[1, r] = 1
            W_1[r, r] = 1

        assert _verify(U_1, V_1, W_1, n=2, m=2, p=1), "Schoolbook 2x2x1 failed"

        total_rank = 2 * (3 * 7 + 1 * 2)  # 46
        U = np.zeros((nm, total_rank), dtype=np.int64)
        V = np.zeros((mp, total_rank), dtype=np.int64)
        W = np.zeros((pn, total_rank), dtype=np.int64)

        col = 0
        for block in range(2):
            # A cols: 2*block, 2*block+1
            # B rows: 2*block, 2*block+1
            # B cols: 0..6, split as 2,2,2,1

            # Three 2x2x2 Strassen blocks
            for sub in range(3):
                for r in range(7):
                    for i in range(2):
                        for j in range(2):
                            U[i * 4 + (2 * block + j), col] += U_s[i * 2 + j, r]
                    for j in range(2):
                        for k in range(2):
                            V[(2 * block + j) * 7 + (sub * 2 + k), col] += V_s[j * 2 + k, r]
                    for k in range(2):
                        for i in range(2):
                            W[(sub * 2 + k) * 2 + i, col] += W_s[k * 2 + i, r]
                    col += 1

            # One 2x2x1 schoolbook block (last column of B)
            for r in range(2):
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + (2 * block + j), col] += U_1[i * 2 + j, r]
                for j in range(2):
                    V[(2 * block + j) * 7 + 6, col] += V_1[j, r]
                for i in range(2):
                    W[6 * 2 + i, col] += W_1[i, r]
                col += 1

        assert _verify(U, V, W), "Strassen block decomposition failed"
        return U, V, W

    # ------------------------------------------------------------------
    # Build baseline decompositions
    # ------------------------------------------------------------------
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = n * m * p  # 56

    try:
        U_str, V_str, W_str = _build_strassen_block()
        if _verify(U_str, V_str, W_str):
            best_U, best_V, best_W = U_str.copy(), V_str.copy(), W_str.copy()
            best_rank = 46
    except Exception:
        pass

    # ------------------------------------------------------------------
    # NEW STRATEGY: Integer lattice enumeration seeded by rank-46 solution
    # For each target rank R < best_rank:
    #   1. Generate random integer projection matrices P_U (nm x R), P_V (mp x R), P_W (pn x R)
    #   2. Project the 46-term decomposition: for each term r46, compute scalar weights
    #      via least-squares in the projected space, then snap to half-integers
    #   3. More directly: use the 46-term tensor as a "high-rank" representation,
    #      then try to find R-term approximations via random linear combinations
    #      and snap to the half-integer lattice.
    #
    # Key insight: if a rank-R decomposition exists with half-integer entries,
    # then for generic random projections, the projected tensor will be close to
    # a rank-R tensor whose entries are half-integers. We enumerate nearby
    # half-integer points.
    # ------------------------------------------------------------------

    def _try_lattice_rank(R, n_trials=30, time_budget=8.0):
        """Try to find exact rank-R decomposition via lattice enumeration."""
        trial_start = time.time()
        for trial in range(n_trials):
            if time.time() > deadline - 3:
                return None
            if time.time() - trial_start > time_budget:
                break

            # Random integer projection matrices
            scale = 2  # entries in {-2,-1,0,1,2}
            P_U = np.random.randint(-scale, scale + 1, size=(nm, R)).astype(np.float64)
            P_V = np.random.randint(-scale, scale + 1, size=(mp, R)).astype(np.float64)
            P_W = np.random.randint(-scale, scale + 1, size=(pn, R)).astype(np.float64)

            # Project the target tensor: T_proj[a,b,c] = sum_{i,j,k} P_U[i,a] P_V[j,b] P_W[k,c] T[i,j,k]
            # This is a 3-tensor of shape (R, R, R)
            T_proj = np.einsum('ia,jb,kc,ijk->abc', P_U, P_V, P_W, T, optimize=True)

            # Now find U', V', W' of shape (R, R) each such that sum_r U'[a,r]V'[b,r]W'[c,r] = T_proj[a,b,c]
            # This is a rank-R decomposition of an R x R x R tensor.
            # If T_proj has rank <= R, we can find it.
            # Use CP decomposition via ALS on the projected tensor.
            # Initialize with random half-integers
            U_p = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=(R, R))
            V_p = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=(R, R))
            W_p = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=(R, R))

            # ALS iterations
            for it in range(50):
                # Update U_p: for each a, solve U_p[a,:] = (T_proj[a,:,:] * (V_p * W_p.T)) / (V_p * W_p.T).T
                # T_proj[a,b,c] = sum_r U_p[a,r] V_p[b,r] W_p[c,r]
                # Fix V_p, W_p: U_p[a,r] = sum_{b,c} T_proj[a,b,c] V_p[b,r] W_p[c,r] / sum_{b,c} V_p[b,r]^2 W_p[c,r]^2
                VW = V_p * W_p.T  # (R, R)
                denom = np.sum(VW * VW, axis=1)  # (R,)
                denom[denom < 1e-12] = 1e-12
                for a in range(R):
                    U_p[a, :] = np.sum(T_proj[a, :, :] * VW, axis=(0, 1)) / denom

                # Update V_p
                UW = U_p * W_p.T
                denom = np.sum(UW * UW, axis=1)
                denom[denom < 1e-12] = 1e-12
                for b in range(R):
                    V_p[b, :] = np.sum(T_proj[:, b, :] * UW, axis=(0, 2)) / denom

                # Update W_p
                UV = U_p * V_p.T
                denom = np.sum(UV * UV, axis=1)
                denom[denom < 1e-12] = 1e-12
                for c in range(R):
                    W_p[c, :] = np.sum(T_proj[:, :, c] * UV, axis=(0, 1)) / denom

            # Snap to half-integers
            U_snapped = np.round(U_p * 2) / 2
            V_snapped = np.round(V_p * 2) / 2
            W_snapped = np.round(W_p * 2) / 2

            # Check if snapped version is exact for T_proj
            T_proj_rec = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped, optimize=True)
            if np.max(np.abs(T_proj_rec - T_proj)) < 1e-8:
                # Lift back: U_full[i, r] = sum_a P_U[i, a] U_snapped[a, r]
                # But this doesn't directly give us the original U.
                # Instead, verify directly: construct U_full, V_full, W_full from the projection
                # Actually, the projection approach is indirect. Let's verify the original tensor.
                # The issue: P_U, P_V, P_W are not invertible generally.
                # Better approach: directly search for half-integer U, V, W of rank R.
                pass

            # Direct approach: random half-integer initialization + local search
            # Generate random half-integer U, V, W
            U_rand = np.random.choice([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], size=(nm, R))
            V_rand = np.random.choice([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], size=(mp, R))
            W_rand = np.random.choice([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], size=(pn, R))

            # Compute residual
            T_rec = np.einsum('ir,jr,kr->ijk', U_rand, V_rand, W_rand, optimize=True)
            residual = T_rec - T
            err = np.sum(residual * residual)
            if err < 1e-10:
                return (U_rand, V_rand, W_rand, R)

            # Local search: flip individual entries to reduce residual
            improved = True
            max_iters = 200
            iter_count = 0
            while improved and iter_count < max_iters:
                improved = False
                iter_count += 1
                if time.time() > deadline - 3:
                    return None

                # Try flipping U entries
                for _ in range(10):
                    idx_i = np.random.randint(nm)
                    idx_r = np.random.randint(R)
                    old_val = U_rand[idx_i, idx_r]
                    # Try nearby half-integers
                    candidates = [old_val + 0.5, old_val - 0.5, old_val + 1.0, old_val - 1.0, 0.0]
                    best_err = err
                    best_val = old_val
                    for cv in candidates:
                        if cv == old_val:
                            continue
                        U_rand[idx_i, idx_r] = cv
                        T_rec = np.einsum('ir,jr,kr->ijk', U_rand, V_rand, W_rand, optimize=True)
                        residual = T_rec - T
                        new_err = np.sum(residual * residual)
                        if new_err < best_err - 1e-12:
                            best_err = new_err
                            best_val = cv
                    U_rand[idx_i, idx_r] = best_val
                    if best_val != old_val:
                        err = best_err
                        improved = True

                # Try flipping V entries
                for _ in range(10):
                    idx_j = np.random.randint(mp)
                    idx_r = np.random.randint(R)
                    old_val = V_rand[idx_j, idx_r]
                    candidates = [old_val + 0.5, old_val - 0.5, old_val + 1.0, old_val - 1.0, 0.0]
                    best_err = err
                    best_val = old_val
                    for cv in candidates:
                        if cv == old_val:
                            continue
                        V_rand[idx_j, idx_r] = cv
                        T_rec = np.einsum('ir,jr,kr->ijk', U_rand, V_rand, W_rand, optimize=True)
                        residual = T_rec - T
                        new_err = np.sum(residual * residual)
                        if new_err < best_err - 1e-12:
                            best_err = new_err
                            best_val = cv
                    V_rand[idx_j, idx_r] = best_val
                    if best_val != old_val:
                        err = best_err
                        improved = True

                # Try flipping W entries
                for _ in range(10):
                    idx_k = np.random.randint(pn)
                    idx_r = np.random.randint(R)
                    old_val = W_rand[idx_k, idx_r]
                    candidates = [old_val + 0.5, old_val - 0.5, old_val + 1.0, old_val - 1.0, 0.0]
                    best_err = err
                    best_val = old_val
                    for cv in candidates:
                        if cv == old_val:
                            continue
                        W_rand[idx_k, idx_r] = cv
                        T_rec = np.einsum('ir,jr,kr->ijk', U_rand, V_rand, W_rand, optimize=True)
                        residual = T_rec - T
                        new_err = np.sum(residual * residual)
                        if new_err < best_err - 1e-12:
                            best_err = new_err
                            best_val = cv
                    W_rand[idx_k, idx_r] = best_val
                    if best_val != old_val:
                        err = best_err
                        improved = True

                if err < 1e-10:
                    return (U_rand, V_rand, W_rand, R)

        return None

    # Try ranks from best_rank-1 down to 25
    for R in range(best_rank - 1, 24, -1):
        if time.time() > deadline - 15:
            break
        result = _try_lattice_rank(R, n_trials=20, time_budget=6.0)
        if result is not None:
            U_new, V_new, W_new, rank_new = result
            if _verify(U_new, V_new, W_new):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = rank_new

    # ------------------------------------------------------------------
    # Final verification and return
    # ------------------------------------------------------------------
    if _verify(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback: schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
