import numpy as np
import time

def construct():
    n, m, p = 2, 2, 11
    deadline = time.time() + 110  # safety margin before 120s kill

    # Build the rank-39 Strassen construction
    U, V, W = _build_strassen(n, m, p)
    best_rank = 39
    best_U, best_V, best_W = np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    T_flat = T.reshape(-1)
    T_norm = np.linalg.norm(T_flat)

    def verify_fast(U_arr, V_arr, W_arr, tol=1e-8):
        """Check if U,V,W exactly decompose T using vectorized operations."""
        rank = U_arr.shape[1]
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # For each c: T[:,:,c] = sum_r W[c,r] * outer(U[:,r], V[:,r])
        # Vectorized: T[:,:,c] = (U * W[c,:][None,:]) @ V.T  -- no, that's not right
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = sum_r (U[a,r] * W[c,r]) * V[b,r]
        # For fixed c: T[:, :, c] = (U * W[c, :][None, :]) @ V.T
        # Wait: (U * W[c,:][None,:]) has shape (nm, rank), V.T has shape (rank, mp)
        # So (U * W[c,:][None,:]) @ V.T has shape (nm, mp)
        # Entry [a,b] = sum_r U[a,r]*W[c,r]*V[b,r] -- YES!
        for c in range(pn):
            computed = (U_arr * W_arr[c, :][None, :]) @ V_arr.T
            if np.max(np.abs(computed - T[:, :, c])) > tol:
                return False
        return True

    def build_flattened_columns(U_arr, V_arr, W_arr):
        """Build matrix where column r is vec(U[:,r] @ V[:,r].T @ W[:,r].T) as 3-tensor flattened."""
        rank = U_arr.shape[1]
        # T_r[a,b,c] = U[a,r]*V[b,r]*W[c,r]
        # Flatten index: a*mp*pn + b*pn + c
        # We can compute this as: for each r, the 3-tensor is U[:,r] @ V[:,r].T @ W[:,r].T (broadcasting)
        # Flattened: np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr).reshape(-1)
        # But that gives all columns at once. Let's do it column by column for the matrix form.
        # Actually, we can compute all at once:
        # M has shape (nm*mp*pn, rank)
        # M[a*mp*pn + b*pn + c, r] = U[a,r]*V[b,r]*W[c,r]
        # This is: np.einsum('ar,br,cr->abcr', U_arr, V_arr, W_arr).reshape(nm*mp*pn, rank)
        # Wait, that's not right either. Let me think...
        # einsum('ar,br,cr->abcr', U, V, W) gives shape (nm, mp, pn, rank)
        # Reshaping to (nm*mp*pn, rank) would interleave incorrectly.
        # We need: for each (a,b,c), the vector over r is U[a,:]*V[b,:]*W[c,:]
        # So M[a*mp*pn + b*pn + c, :] = U[a,:] * V[b,:] * W[c,:]
        # We can compute: M = np.einsum('ar,br,cr->abcr', U_arr, V_arr, W_arr).reshape(-1, rank)
        # Let's check: the last axis is r, and the first three axes are a,b,c
        # Reshaping (nm, mp, pn, rank) -> (nm*mp*pn, rank) keeps the last axis as columns
        # And the first three axes get flattened in C order: a varies slowest, c varies fastest
        # So index in flat array = a*mp*pn + b*pn + c -- YES, that's what we want!
        return np.einsum('ar,br,cr->abcr', U_arr, V_arr, W_arr).reshape(-1, rank)

    current_U = best_U.copy()
    current_V = best_V.copy()
    current_W = best_W.copy()
    current_rank = best_rank

    # Greedy rank reduction with least squares
    removed = set()
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        found_removal = False
        best_candidate = None
        best_candidate_rank = current_rank

        for r in range(current_rank):
            if r in removed:
                continue
            keep = [i for i in range(current_rank) if i != r and i not in removed]
            if len(keep) == 0:
                continue
            U_keep = current_U[:, keep]
            V_keep = current_V[:, keep]
            W_keep = current_W[:, keep]

            # Build flattened matrix and solve
            M = build_flattened_columns(U_keep, V_keep, W_keep)
            try:
                x, residuals, rank_m, sv = np.linalg.lstsq(M, T_flat, rcond=1e-12)
            except Exception:
                continue

            if len(residuals) == 0 or residuals[0] > 1e-7 * T_norm:
                continue

            # Construct new factor matrices
            new_U = U_keep * x[None, :]
            new_V = V_keep * x[None, :]
            new_W = W_keep * x[None, :]

            # Verify
            if verify_fast(new_U, new_V, new_W, tol=1e-8):
                if len(keep) < best_candidate_rank:
                    best_candidate_rank = len(keep)
                    best_candidate = (new_U, new_V, new_W, r)

        if best_candidate is not None:
            current_U, current_V, current_W, removed_idx = best_candidate
            current_rank = current_U.shape[1]
            removed.add(removed_idx)
            found_removal = True
            if current_rank < best_rank:
                best_rank = current_rank
                best_U = current_U.copy()
                best_V = current_V.copy()
                best_W = current_W.copy()
        else:
            break

    # Final verification
    if verify_fast(current_U, current_V, current_W, tol=1e-8):
        best_U = current_U
        best_V = current_V
        best_W = current_W
        best_rank = current_U.shape[1]

    # Convert back to list format
    U_out = [[float(best_U[i, r]) for r in range(best_rank)] for i in range(nm)]
    V_out = [[float(best_V[i, r]) for r in range(best_rank)] for i in range(mp)]
    W_out = [[float(best_W[i, r]) for r in range(best_rank)] for i in range(pn)]

    return {"U": U_out, "V": V_out, "W": W_out}


def _build_strassen(n, m, p):
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    U[0][r] += 1
    V[v_e2][r] += 1
    W[w_c02][r] += 1
    r += 1
    U[1][r] += 1
    V[v_g2][r] += 1
    W[w_c02][r] += 1
    r += 1
    U[2][r] += 1
    V[v_e2][r] += 1
    W[w_c12][r] += 1
    r += 1
    U[3][r] += 1
    V[v_g2][r] += 1
    W[w_c12][r] += 1
    r += 1
    return U, V, W
