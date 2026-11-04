def _check_and_reduce(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    tensors = np.zeros((rank, n*m*p))
    for c in range(rank):
        T_c = U_3d[:,:,c] * V_3d[:,:,c] * W_3d[:,:,c]
        tensors[c] = T_c.flatten()
    
    r = np.linalg.matrix_rank(tensors, tol=1e-10)
    if r < rank:
        # Find a minimal independent subset
        kept = []
        for c in range(rank):
            if not kept:
                kept.append(c)
                continue
            sub = tensors[kept]
            try:
                _, residuals, _, _ = np.linalg.lstsq(sub.T, tensors[c], rcond=1e-10)
                if residuals[0] < 1e-8:
                    continue
            except:
                pass
            kept.append(c)
        if len(kept) < rank:
            # Adjust coefficients
            # This is complex; for now just return original
            pass
    return U, V, W
