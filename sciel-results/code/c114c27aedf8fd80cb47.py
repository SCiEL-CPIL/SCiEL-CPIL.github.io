def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    return np.allclose(T, T_target, atol=tol)

def _try_reduce_rank(U, V, W, n, m, p, max_iter=5000, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for attempt in range(rank):
        reduced = False
        for col_to_remove in range(rank):
            mask = [r for r in range(rank) if r != col_to_remove]
            U_r = U[:, mask]
            V_r = V[:, mask]
            W_r = W[:, mask]
            
            if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol):
                U, V, W = U_r, V_r, W_r
                rank = U.shape[1]
                reduced = True
                break
        if not reduced:
            break
            
    return U, V, W
