"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p

    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def schoolbook():
        U = np.zeros((nm, schoolbook_rank))
        V = np.zeros((mp, schoolbook_rank))
        W = np.zeros((pn, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_best, V_best, W_best = schoolbook()
    best_rank = schoolbook_rank

    rng = np.random.default_rng(42)

    def check_decomposition(U, V, W, tol=1e-8):
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.linalg.norm(recon - T) / np.linalg.norm(T)
        return err < tol, err

    def try_rank(r, max_time=10):
        """Try to find rank-r decomposition using vectorized ALS."""
        t0 = time.time()
        for restart in range(5):
            if time.time() > deadline or time.time() - t0 > max_time:
                return None
            
            U = rng.standard_normal((nm, r))
            V = rng.standard_normal((mp, r))
            W = rng.standard_normal((pn, r))
            
            prev_err = np.inf
            for it in range(150):
                if time.time() > deadline or time.time() - t0 > max_time:
                    return None
                
                # Update U: U[a] = argmin ||(V*W) @ U[a] - T[a]||
                VW = np.einsum('bc,dc->bdc', V, W).reshape(mp*pn, r)
                try:
                    G = VW.T @ VW
                    G_inv = np.linalg.inv(G + 1e-10 * np.eye(r))
                    VW_G = VW.T @ G_inv
                    T_flat = T.reshape(nm, mp*pn)
                    U = T_flat @ VW_G
                except:
                    pass
                
                # Update V
                UW = np.einsum('ac,dc->adc', U, W).reshape(nm*pn, r)
                try:
                    G = UW.T @ UW
                    G_inv = np.linalg.inv(G + 1e-10 * np.eye(r))
                    UW_G = UW.T @ G_inv
                    T_flat = T.reshape(mp, nm*pn)
                    V = T_flat @ UW_G
                except:
                    pass
                
                # Update W
                UV = np.einsum('ac,bc->abc', U, V).reshape(nm*mp, r)
                try:
                    G = UV.T @ UV
                    G_inv = np.linalg.inv(G + 1e-10 * np.eye(r))
                    UV_G = UV.T @ G_inv
                    T_flat = T.reshape(pn, nm*mp)
                    W = T_flat @ UV_G
                except:
                    pass
                
                recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.linalg.norm(recon - T) / np.linalg.norm(T)
                
                if err < 1e-10:
                    return U, V, W
                
                if abs(prev_err - err) < 1e-12 and it > 30:
                    break
                prev_err = err
            
            valid, err = check_decomposition(U, V, W)
            if valid:
                return U, V, W
        
        return None

    for r in range(schoolbook_rank - 1, max(1, schoolbook_rank - 20), -1):
        if time.time() > deadline:
            break
        result = try_rank(r, max_time=8)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
