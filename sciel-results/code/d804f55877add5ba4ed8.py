def _khatri_rao(V, W):
    # V: (n2, rank), W: (n3, rank)
    # Result: (n2*n3, rank) where row (b*n3+c, r) = V[b,r]*W[c,r]
    n2, r = V.shape
    n3, _ = W.shape
    return (V[:, None, :] * W[None, :, :]).reshape(n2*n3, r)

def _als_decompose(T, rank, n_iters=50, n_restarts=3, seed=0):
    n1, n2, n3 = T.shape
    rng = np.random.default_rng(seed)
    best_err = np.inf
    best = None
    T_flat = T.reshape(n1, n2*n3)
    for restart in range(n_restarts):
        U = rng.standard_normal((n1, rank))
        V = rng.standard_normal((n2, rank))
        W = rng.standard_normal((n3, rank))
        for it in range(n_iters):
            # Update U
            VW = _khatri_rao(V, W)  # (n2*n3, rank)
            U = T_flat @ np.linalg.pinv(VW)  # (n1, rank)
            # Update V
            UW = _khatri_rao(U, W)  # (n1*n3, rank)
            T_flat_v = T.reshape(n2, n1*n3).T  # (n1*n3, n2)
            V = T_flat_v @ np.linalg.pinv(UW)  # (n2, rank)
            # Update W
            UV = _khatri_rao(U, V)  # (n1*n2, rank)
            T_flat_w = T.reshape(n3, n1*n2).T  # (n1*n2, n3)
            W = T_flat_w @ np.linalg.pinv(UV)  # (n3, rank)
        # Compute error
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < best_err:
            best_err = err
            best = (U, V, W)
    return best, best_err
