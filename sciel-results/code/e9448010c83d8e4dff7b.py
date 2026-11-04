import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _error(params, nm, mp, pn, rank, T_flat):
    U = params[:nm * rank].reshape(nm, rank)
    V = params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
    W = params[nm * rank + mp * rank:].reshape(pn, rank)
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    err = recon - T_flat.reshape(nm, mp, pn)
    return np.sum(err * err)

def _try_rank(n, m, p, rank, T_flat, time_budget):
    nm, mp, pn = n * m, m * p, p * n
    n_params = (nm + mp + pn) * rank
    start = time.time()
    best_err = np.inf
    best_params = None
    
    max_restarts = max(1, int(time_budget / 0.3))
    for restart in range(max_restarts):
        if time.time() - start > time_budget * 0.92:
            break
        for scale in [0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            if time.time() - start > time_budget * 0.92:
                break
            params0 = np.random.randn(n_params) * scale
            try:
                res = minimize(
                    _error, params0,
                    args=(nm, mp, pn, rank, T_flat),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 1500}
                )
                err = _error(res.x, nm, mp, pn, rank, T_flat)
                if err < best_err:
                    best_err = err
                    best_params = res.x.copy()
                
                # Polish step: if we got close, try to drive it to machine precision
                if err < 1e-6:
                    res2 = minimize(
                        _error, res.x,
                        args=(nm, mp, pn, rank, T_flat),
                        method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 3000}
                    )
                    err2 = _error(res2.x, nm, mp, pn, rank, T_flat)
                    if err2 < best_err:
                        best_err = err2
                        best_params = res2.x.copy()
                        
                if best_err < 1e-10:
                    return best_err, best_params
            except Exception:
                continue
    return best_err, best_params

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    deadline = time.time() + 140
    
    # Schoolbook fallback (rank 24)
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(nm)]
    V_fb = [[0] * rank_fb for _ in range(mp)]
    W_fb = [[0] * rank_fb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    for rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline - 5:
            break
        remaining = deadline - 5 - time.time()
        budget = remaining * 0.35 if rank <= 20 else remaining * 0.25
        best_err, best_params = _try_rank(n, m, p, rank, T_flat, budget)
        
        if best_err < 1e-8 and best_params is not None:
            U = best_params[:nm * rank].reshape(nm, rank)
            V = best_params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
            W = best_params[nm * rank + mp * rank:].reshape(pn, rank)
            
            # Snap to half-integers
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            recon_s = np.einsum('ac,bc,dc->abd', U_s, V_s, W_s)
            if np.max(np.abs(recon_s - T)) < 1e-9:
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            
            # Try integer snap
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            recon_i = np.einsum('ac,bc,dc->abd', U_i, V_i, W_i)
            if np.max(np.abs(recon_i - T)) < 1e-9:
                return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
            
            # Return float version if close enough
            recon_f = np.einsum('ac,bc,dc->abd', U, V, W)
            if np.max(np.abs(recon_f - T)) < 1e-6:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
