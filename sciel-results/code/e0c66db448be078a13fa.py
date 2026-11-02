"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # stay safely under 150s wall-clock

    n, m, p = 2, 3, 6
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12

    # Build the target multiplication tensor T[i,j,k] = 1 if all indices in range
    # T has shape (nm, mp, pn) = (6, 18, 12)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                # Check if this triple corresponds to a valid multiplication term
                # i = row_a * m + col_a  (row_a in [0,n), col_a in [0,m))
                # j = col_a * p + col_b  (col_a in [0,m), col_b in [0,p))
                # k = col_b * n + row_b  (col_b in [0,p), row_b in [0,n))
                row_a, col_a = divmod(i, m)
                col_a2, col_b = divmod(j, p)
                col_b2, row_b = divmod(k, n)
                if col_a == col_a2 and col_b == col_b2:
                    T[i, j, k] = 1.0

    # Try to find a low-rank exact decomposition
    # Known: rank(2x3x3) = 11, so rank(2x3x6) <= 22
    # We try ranks from 18 upward, using ALS optimization

    best_result = None
    best_rank = nm * mp * pn  # schoolbook fallback

    # Precompute schoolbook as fallback
    U_fb = np.zeros((nm, best_rank))
    V_fb = np.zeros((mp, best_rank))
    W_fb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1

    # Candidate ranks to try (must be >= known lower bounds)
    # Lower bound for matmul(2,3,6): at least max(n*m, m*p, p*n) = 18
    # Also need at least rank of any "slice"
    candidate_ranks = [18, 19, 20, 21, 22]

    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break

        success = False
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline - 15:
                break

            rng = np.random.default_rng(seed=42 + attempt * 1000 + rank)
            U = rng.standard_normal((nm, rank))
            V = rng.standard_normal((mp, rank))
            W = rng.standard_normal((pn, rank))

            # ALS iterations
            max_als = 300
            for it in range(max_als):
                if time.time() > deadline - 15:
                    break

                # Solve for U: T_jk[i, j, k] = sum_r U[i,r]*V[j,r]*W[k,r]
                # For each i: T[i,:,:] = U[i,:] @ (V * W)^T  (outer product weighted)
                # Reshape: T_i = T[i] (shape mp x pn)
                # T_i = U[i] @ diag(V_col * W_col) ... 
                # Actually: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # Fix V, W: for each i, T[i] (mp x pn) = sum_r U[i,r] * (V[:,r] outer W[:,r])
                # So T[i] = U[i] @ M_VW where M_VW[r, (j,k)] = V[j,r]*W[k,r]
                # M_VW shape: rank x (mp*pn)
                M_VW = np.zeros((rank, mp * pn))
                for r in range(rank):
                    M_VW[r, :] = (V[:, r][:, None] * W[:, r][None, :]).flatten()
                # T reshaped: (nm, mp*pn)
                T_flat = T.reshape(nm, mp * pn)
                # Solve: T_flat[i] = U[i] @ M_VW  =>  T_flat = U @ M_VW
                # U = T_flat @ M_VW^+ (pseudo-inverse)
                # Use lstsq: T_flat.T = M_VW.T @ U.T
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)
                    U = U_new.T
                except np.linalg.LinAlgError:
                    break

                # Solve for V: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # For each j: T[:,:,j] (nm x pn) = sum_r V[j,r] * (U[:,r] outer W[:,r])
                M_UW = np.zeros((rank, nm * pn))
                for r in range(rank):
                    M_UW[r, :] = (U[:, r][:, None] * W[:, r][None, :]).flatten()
                T_flat_v = T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_UW.T, T_flat_v.T, rcond=None)
                    V = V_new.T
                except np.linalg.LinAlgError:
                    break

                # Solve for W: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # For each k: T[:,:,k] (nm x mp) = sum_r W[k,r] * (U[:,r] outer V[:,r])
                M_UV = np.zeros((rank, nm * mp))
                for r in range(rank):
                    M_UV[r, :] = (U[:, r][:, None] * V[:, r][None, :]).flatten()
                T_flat_w = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_UV.T, T_flat_w.T, rcond=None)
                    W = W_new.T
                except np.linalg.LinAlgError:
                    break

                # Check convergence: compute error
                if it % 10 == 9 or it == max_als - 1:
                    # Reconstruct and check
                    err = 0.0
                    for i in range(nm):
                        for j in range(mp):
                            for k in range(pn):
                                val = 0.0
                                for r in range(rank):
                                    val += U[i, r] * V[j, r] * W[k, r]
                                err += (val - T[i, j, k]) ** 2
                    err = np.sqrt(err)
                    if err < 1e-8:
                        success = True
                        break

            if success:
                break

        if success:
            # Round to nearest half-integer to get clean entries
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2

            # Verify with rounded values
            err = 0.0
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        val = 0.0
                        for r in range(rank):
                            val += U_r[i, r] * V_r[j, r] * W_r[k, r]
                        err += (val - T[i, j, k]) ** 2
            err = np.sqrt(err)

            if err < 1e-6:
                best_result = (U_r, V_r, W_r, rank)
                best_rank = rank
                break  # Found lowest rank, stop

            # If rounding failed, keep unrounded but verify exactness differently
            # Try: maybe entries are actually integers
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            err2 = 0.0
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        val = 0.0
                        for r in range(rank):
                            val += U_i[i, r] * V_i[j, r] * W_i[k, r]
                        err2 += (val - T[i, j, k]) ** 2
            err2 = np.sqrt(err2)
            if err2 < 1e-6:
                best_result = (U_i, V_i, W_i, rank)
                best_rank = rank
                break

    if best_result is None:
        # Fall back to schoolbook
        U, V, W, rank = U_fb, V_fb, W_fb, best_rank
    else:
        U, V, W, rank = best_result

    # Convert to lists of lists (or keep as numpy arrays - task says integer/half-integer entries)
    # Return as lists to be safe
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
