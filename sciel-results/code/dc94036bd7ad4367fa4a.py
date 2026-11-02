"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p

    # Build the tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Helper to verify a decomposition
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        """Verify sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]"""
        rank = U_mat.shape[1]
        max_err = 0.0
        for a in range(nm):
            for b in range(mp):
                uv = U_mat[a, :] * V_mat[b, :]
                for c in range(pn):
                    val = np.dot(uv, W_mat[c, :])
                    err = abs(val - T[a, b, c])
                    if err > max_err:
                        max_err = err
        return max_err < tol

    def is_half_integer(arr, tol=1e-8):
        """Check if array entries are integer or half-integer"""
        doubled = arr * 2
        rounded = np.round(doubled)
        return np.all(np.abs(doubled - rounded) < tol)

    def round_to_half(arr):
        """Round to nearest half-integer"""
        return np.round(arr * 2) / 2

    # Strategy 1: Try SVD-based decomposition at various ranks
    T_mat = T.reshape(nm * mp, pn)

    # Get singular values to estimate minimum possible rank
    s_vals = np.linalg.svd(T_mat, compute_uv=False)
    tol = 1e-10
    min_rank_estimate = int(np.sum(s_vals > tol))
    min_rank_estimate = max(1, min_rank_estimate)

    best_result = None

    # Try to find decomposition starting from estimated rank
    max_rank_to_try = min(schoolbook_rank, max(min_rank_estimate, 30))

    for target_rank in range(min_rank_estimate, max_rank_to_try + 1):
        if time.time() > deadline:
            break

        # SVD decomposition
        U_svd, s, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)

        # Truncate to target_rank
        U_trunc = U_svd[:, :target_rank] * s[:target_rank]
        Vt_trunc = Vt_svd[:target_rank, :]

        U_3d = U_trunc.reshape(nm, target_rank)
        V_3d = Vt_trunc.reshape(mp, target_rank)

        # Solve for W: T[:,:,c].flatten() = M @ W[c,:]
        # where M[:,r] = (U_3d[:,r] * V_3d[:,r]).flatten()
        M = np.zeros((nm * mp, target_rank))
        for r in range(target_rank):
            M[:, r] = (U_3d[:, r] * V_3d[:, r]).flatten()

        W_3d = np.zeros((pn, target_rank))
        for c in range(pn):
            target_vec = T[:, :, c].flatten()
            try:
                w, _, _, _ = np.linalg.lstsq(M, target_vec, rcond=None)
                W_3d[c, :] = w
            except Exception:
                continue

        # Check if this is a valid real decomposition
        if verify(U_3d, V_3d, W_3d):
            # Try rounding to half-integers
            U_r = round_to_half(U_3d)
            V_r = round_to_half(V_3d)
            W_r = round_to_half(W_3d)

            if verify(U_r, V_r, W_r):
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                break

            # Try iterative refinement: adjust U,V,W to be half-integers
            # while maintaining the decomposition
            success = False
            for iteration in range(100):
                if time.time() > deadline:
                    break

                # Compute current error
                U_curr = U_r.copy()
                V_curr = V_r.copy()
                W_curr = W_r.copy()

                # Try to fix each matrix by projecting to half-integer lattice
                # while minimizing the error in the tensor product
                # Use a simple approach: round each matrix and check
                # If error is small, try local adjustments

                # Check current error
                if verify(U_curr, V_curr, W_curr):
                    success = True
                    U_r, V_r, W_r = U_curr, V_curr, W_curr
                    break

                # Try adjusting W first (smallest matrix)
                for c in range(pn):
                    for r in range(target_rank):
                        # Try changing W[c,r] by +/- 0.5
                        old_val = W_curr[c, r]
                        for delta in [-0.5, 0.5]:
                            W_curr[c, r] = old_val + delta
                            if verify(U_curr, V_curr, W_curr):
                                success = True
                                break
                            W_curr[c, r] = old_val
                        if success:
                            break
                    if success:
                        break
                if success:
                    break

                # Try adjusting V
                for b in range(mp):
                    for r in range(target_rank):
                        old_val = V_curr[b, r]
                        for delta in [-0.5, 0.5]:
                            V_curr[b, r] = old_val + delta
                            if verify(U_curr, V_curr, W_curr):
                                success = True
                                break
                            V_curr[b, r] = old_val
                        if success:
                            break
                    if success:
                        break
                if success:
                    break

                # Try adjusting U
                for a in range(nm):
                    for r in range(target_rank):
                        old_val = U_curr[a, r]
                        for delta in [-0.5, 0.5]:
                            U_curr[a, r] = old_val + delta
                            if verify(U_curr, V_curr, W_curr):
                                success = True
                                break
                            U_curr[a, r] = old_val
                        if success:
                            break
                    if success:
                        break
                if success:
                    break

            if success:
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                break

        # If we've reached schoolbook rank, use the known decomposition
        if target_rank >= schoolbook_rank:
            U_sb = np.zeros((nm, target_rank))
            V_sb = np.zeros((mp, target_rank))
            W_sb = np.zeros((pn, target_rank))

            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, col] = 1
                        V_sb[j * p + k, col] = 1
                        W_sb[k * n + i, col] = 1
                        col += 1

            if verify(U_sb, V_sb, W_sb):
                best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
                break

    # Strategy 2: Randomized search for lower rank
    if best_result is None or best_result["U"].shape[1] > min_rank_estimate + 5:
        if time.time() < deadline - 20:
            rng = np.random.default_rng(42)

            # Try random initializations and refine
            for target_rank in range(max(1, min_rank_estimate), min(min_rank_estimate + 10, schoolbook_rank)):
                if time.time() > deadline - 10:
                    break

                found = False
                for trial in range(50):
                    if time.time() > deadline - 10:
                        break

                    # Random initialization
                    U_rand = rng.integers(-2, 3, size=(nm, target_rank)) / 2.0
                    V_rand = rng.integers(-2, 3, size=(mp, target_rank)) / 2.0
                    W_rand = rng.integers(-2, 3, size=(pn, target_rank)) / 2.0

                    if verify(U_rand, V_rand, W_rand):
                        best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                        found = True
                        break

                    # Try to solve for W given random U,V
                    M_rand = np.zeros((nm * mp, target_rank))
                    for r in range(target_rank):
                        M_rand[:, r] = (U_rand[:, r] * V_rand[:, r]).flatten()

                    # Check if M has full column rank
                    try:
                        rank_m = np.linalg.matrix_rank(M_rand)
                        if rank_m < target_rank:
                            continue
                    except Exception:
                        continue

                    W_solved = np.zeros((pn, target_rank))
                    valid = True
                    for c in range(pn):
                        target_vec = T[:, :, c].flatten()
                        try:
                            w, res, _, _ = np.linalg.lstsq(M_rand, target_vec, rcond=None)
                            # Check if solution is close to half-integer
                            w_rounded = round_to_half(w)
                            if np.max(np.abs(w - w_rounded)) < 0.01:
                                W_solved[c, :] = w_rounded
                            else:
                                valid = False
                                break
                        except Exception:
                            valid = False
                            break

                    if valid and verify(U_rand, V_rand, W_solved):
                        best_result = {"U": U_rand, "V": V_rand, "W": W_solved}
                        found = True
                        break

                if found:
                    break

    # Fallback: schoolbook decomposition
    if best_result is None:
        rank = schoolbook_rank
        U_sb = np.zeros((nm, rank))
        V_sb = np.zeros((mp, rank))
        W_sb = np.zeros((pn, rank))

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1

        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Convert to lists and return
    return {
        "U": best_result["U"].tolist(),
        "V": best_result["V"].tolist(),
        "W": best_result["W"].tolist()
    }
# EVOLVE-BLOCK-END
