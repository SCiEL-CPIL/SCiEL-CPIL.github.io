# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n, m, p = 2, 3, 7
    nm, mp, pn = n*m, m*p, p*n
    
    # Build tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Unfoldings for ALS
    T1 = T.reshape(nm, mp * pn)
    T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)

    def round_to_half(x):
        return np.round(2 * x) / 2.0

    def compute_residual(U, V, W):
        r = U.shape[1]
        T_rec = np.zeros_like(T)
        for c in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return np.linalg.norm(T_rec - T)

    def local_search(U, V, W, max_iters=100):
        U = round_to_half(U).astype(np.float64)
        V = round_to_half(V).astype(np.float64)
        W = round_to_half(W).astype(np.float64)
        
        r = U.shape[1]
        
        def get_num_errors():
            T_rec = np.zeros_like(T)
            for c in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            return np.sum(np.abs(T_rec - T) > 1e-9)

        num_errors = get_num_errors()
        if num_errors == 0:
            return U, V, W, True

        for _ in range(max_iters):
            mat_name = random.choice(['U', 'V', 'W'])
            if mat_name == 'U':
                mat = U
                idx = random.choice(nm)
                col = random.choice(r)
            elif mat_name == 'V':
                mat = V
                idx = random.choice(mp)
                col = random.choice(r)
            else:
                mat = W
                idx = random.choice(pn)
                col = random.choice(r)
                
            old_val = mat[idx, col]
            
            # Try +0.5
            mat[idx, col] = old_val + 0.5
            err_new = get_num_errors()
            if err_new < num_errors:
                num_errors = err_new
            else:
                # Try -0.5
                mat[idx, col] = old_val - 0.5
                err_new = get_num_errors()
                if err_new < num_errors:
                    num_errors = err_new
                else:
                    mat[idx, col] = old_val
            
            if num_errors == 0:
                return U, V, W, True
                
        return U, V, W, False

    def try_rank(rank, seeds=10):
        for seed in range(seeds):
            random.seed(seed)
            np.random.seed(seed)
            
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # ALS
            for it in range(50):
                # Update U
                V_exp = V.T
                W_exp = W.T
                X = np.einsum('rm,rn->rmn', V_exp, W_exp).reshape(rank, mp*pn)
                U_new, _, _, _ = np.linalg.lstsq(X.T, T1.T, rcond=None)
                U = U_new.T
                
                # Update V
                U_exp = U.T
                W_exp = W.T
                Y = np.einsum('rm,rn->rmn', U_exp, W_exp).reshape(rank, nm*pn)
                V_new, _, _, _ = np.linalg.lstsq(Y.T, T2.T, rcond=None)
                V = V_new.T
                
                # Update W
                U_exp = U.T
                V_exp = V.T
                Z = np.einsum('rm,rn->rmn', U_exp, V_exp).reshape(rank, nm*mp)
                W_new, _, _, _ = np.linalg.lstsq(Z.T, T3.T, rcond=None)
                W = W_new.T
                
            residual = compute_residual(U, V, W)
            
            if residual < 1e-6:
                U_r, V_r, W_r, success = local_search(U, V, W)
                if success:
                    return U_r, V_r, W_r, rank
            elif residual < 1.0:
                 U_r, V_r, W_r, success = local_search(U, V, W)
                 if success:
                    return U_r, V_r, W_r, rank

        return None, None, None, None

    # Try ranks from 38 upwards
    for rank in range(38, 43):
        U, V, W, r_found = try_rank(rank, seeds=10)
        if U is not None:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
