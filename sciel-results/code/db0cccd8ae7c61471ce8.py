import numpy as np

def construct():
    n, m, p = 3, 3, 4
    rank = 22
    
    # Build target tensor T[o, a, b]
    # o = k*n + i (0..11), a = j*m + i (0..8), b = k*p + j (0..11)
    T = np.zeros((p*n, n*m, m*p))
    for o in range(p*n):
        i2 = o % n
        k2 = o // n
        for a in range(n*m):
            i1 = a % m
            j1 = a // m
            for b in range(m*p):
                j2 = b % p
                k3 = b // p
                if i1 == i2 and j1 == j2 and k3 == k2:
                    T[o, a, b] = 1.0
    
    # ALS with multiple random restarts
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for trial in range(20):
        np.random.seed(trial)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for iteration in range(500):
            # Solve for U
            M = np.einsum('br,or->ob r', V, W).reshape(m*p * p*n, rank)
            T_flat = T.transpose(1, 0, 2).reshape(n*m, p*n * m*p)
            U_new, _, _, _ = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
            U = U_new.T
            
            # Solve for V
            M2 = np.einsum('ar,or->oa r', U, W).reshape(n*m * p*n, rank)
            T_flat2 = T.transpose(2, 0, 1).reshape(m*p, p*n * n*m)
            V_new, _, _, _ = np.linalg.lstsq(M2.T, T_flat2.T, rcond=None)
            V = V_new.T
            
            # Solve for W
            M3 = np.einsum('ar,br->ab r', U, V).reshape(n*m * m*p, rank)
            T_flat3 = T.transpose(0, 1, 2).reshape(p*n, n*m * m*p)
            W_new, _, _, _ = np.linalg.lstsq(M3.T, T_flat3.T, rcond=None)
            W = W_new.T
            
            if iteration % 50 == 0:
                T_recon = np.einsum('ar,br,or->oab', U, V, W)
                err = np.max(np.abs(T_recon - T))
                if err < 1e-8:
                    break
        
        T_recon = np.einsum('ar,br,or->oab', U, V, W)
        err = np.max(np.abs(T_recon - T))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-8:
            break
    
    U, V, W = best_U, best_V, best_W
    
    # Try rounding to half-integers
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    T_r = np.einsum('ar,br,or->oab', U_r, V_r, W_r)
    err_r = np.max(np.abs(T_r - T))
    
    # Try rounding to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    T_i = np.einsum('ar,br,or->oab', U_i, V_i, W_i)
    err_i = np.max(np.abs(T_i - T))
    
    if err_r < err_i and err_r < 1e-6:
        U_out, V_out, W_out = U_r, V_r, W_r
    elif err_i < 1e-6:
        U_out, V_out, W_out = U_i, V_i, W_i
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
