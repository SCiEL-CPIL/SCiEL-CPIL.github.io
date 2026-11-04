def get_schoolbook():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    R = 42
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0
                
    U_sb, V_sb, W_sb = get_schoolbook()
    
    def solve_for_rank(R, seed=0, time_limit=10.0, init=None):
        np.random.seed(seed)
        if init is None:
            U = np.random.randn(nm, R) * 0.1
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1
        else:
            U, V, W = init
            
        def residuals(x):
            U_flat, V_flat, W_flat = x[:nm*R], x[nm*R:nm*R+mp*R], x[nm*R+mp*R:]
            U_mat = U_flat.reshape(nm, R)
            V_mat = V_flat.reshape(mp, R)
            W_mat = W_flat.reshape(pn, R)
            T_calc = np.einsum('ur,vr,wr->uvw', U_mat, V_mat, W_mat)
            return (T_calc - T_target).flatten()
            
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        start_time = time.time()
        try:
            res = least_squares(residuals, x0, method='trf', max_nfev=2000)
            if time.time() - start_time > time_limit:
                return None
            if res.cost < 1e-8:
                U_mat = res.x[:nm*R].reshape(nm, R)
                V_mat = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                W_mat = res.x[nm*R+mp*R:].reshape(pn, R)
                return {
                    "U": U_mat.tolist(),
                    "V": V_mat.tolist(),
                    "W": W_mat.tolist()
                }
        except Exception:
            pass
        return None

    # Try to reduce rank by 1
    # Take schoolbook, drop last column
    U_init = U_sb[:, :41]
    V_init = V_sb[:, :41]
    W_init = W_sb[:, :41]
    
    result = solve_for_rank(41, seed=0, time_limit=20.0, init=(U_init, V_init, W_init))
    if result:
        return result
        
    # Try random initializations for R=41
    for seed in range(5):
        result = solve_for_rank(41, seed=seed, time_limit=10.0)
        if result:
            return result
            
    # Try R=40
    U_init = U_sb[:, :40]
    V_init = V_sb[:, :40]
    W_init = W_sb[:, :40]
    result = solve_for_rank(40, seed=0, time_limit=20.0, init=(U_init, V_init, W_init))
    if result:
        return result
        
    # Fallback
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
