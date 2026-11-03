def _try_better_decomposition(deadline):
    """
    Try to find a decomposition with rank < 84 using sparse ALS search.
    Initialize from the block decomposition, then run alternating least squares
    with sparsity to reduce effective rank.
    """
    import numpy as np
    import time

    n, m, p = 3, 5, 6
    T = _build_target_tensor(n, m, p)
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18
    
    # Start from block decomposition
    U0, V0, W0 = _block_decomposition_356()
    r0 = U0.shape[1]  # 84

    best_result = None
    best_rank = r0

    # We will try ranks 83 down to 80
    # But we will do multiple restarts for each rank
    num_restarts = 5
    max_iter_per_restart = 30
    
    for target_rank in range(83, 79, -1):
        if time.time() > deadline - 10:
            break
            
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random Gaussian + small noise from the top singular vectors
            # Use SVD of the tensor to get a good initialization
            T_mat = T.reshape(nm, mp * pn)
            try:
                U_svd, s, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
            except Exception:
                U_svd = np.random.randn(nm, target_rank)
                Vt_svd = np.random.randn(mp * pn, target_rank)

            # Mix SVD and Block Decomp for initialization
            if restart == 0:
                # Use block decomp truncated
                U = U0[:, :target_rank].copy() if target_rank <= r0 else np.random.randn(nm, target_rank)
                V = V0[:, :target_rank].copy() if target_rank <= r0 else np.random.randn(mp, target_rank)
                W = W0[:, :target_rank].copy() if target_rank <= r0 else np.random.randn(pn, target_rank)
            else:
                # Random initialization with some structure
                np.random.seed(restart)
                U = np.random.randn(nm, target_rank) * 0.1
                V = np.random.randn(mp, target_rank) * 0.1
                W = np.random.randn(pn, target_rank) * 0.1
                # Add SVD components
                if target_rank <= U_svd.shape[1]:
                    U += U_svd[:, :target_rank] * 0.1
                    # Note: V and W initialization from SVD is tricky for 3-way
                    # Just use random for V, W in restarts > 0

            # Run ALS iterations
            prev_err = float('inf')
            for it in range(max_iter_per_restart):
                if time.time() > deadline - 2:
                    break

                # Update U
                Z_flat = np.zeros((mp * pn, target_rank))
                for b in range(mp):
                    for c in range(pn):
                        Z_flat[b * pn + c, :] = V[b, :] * W[c, :]

                U_new = np.zeros((nm, target_rank))
                for a in range(nm):
                    rhs = T[a, :, :].flatten()
                    try:
                        U_new[a, :] = np.linalg.lstsq(Z_flat, rhs, rcond=None)[0]
                    except Exception:
                        U_new[a, :] = U[a, :]

                # Update V
                Z_flat_V = np.zeros((nm * pn, target_rank))
                for a in range(nm):
                    for c in range(pn):
                        Z_flat_V[a * pn + c, :] = U[a, :] * W[c, :]

                V_new = np.zeros((mp, target_rank))
                for b in range(mp):
                    rhs = T[:, b, :].flatten()
                    try:
                        V_new[b, :] = np.linalg.lstsq(Z_flat_V, rhs, rcond=None)[0]
                    except Exception:
                        V_new[b, :] = V[b, :]

                # Update W
                Z_flat_W = np.zeros((nm * mp, target_rank))
                for a in range(nm):
                    for b in range(mp):
                        Z_flat_W[a * mp + b, :] = U[a, :] * V[b, :]

                W_new = np.zeros((pn, target_rank))
                for c in range(pn):
                    rhs = T[:, :, c].flatten()
                    try:
                        W_new[c, :] = np.linalg.lstsq(Z_flat_W, rhs, rcond=None)[0]
                    except Exception:
                        W_new[c, :] = W[c, :]

                U, V, W = U_new, V_new, W_new

                # Compute error
                T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
                err = np.linalg.norm(T_comp - T)
                if abs(prev_err - err) < 1e-6:
                    break
                prev_err = err

            # Verify the result
            if _verify_decomp(n, m, p, U, V, W):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = (U.copy(), V.copy(), W.copy())
                    break  # Found a valid decomposition at this rank, try even lower

    if best_result is not None:
        return best_result
    return None
