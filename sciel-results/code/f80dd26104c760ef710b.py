import numpy as np
import time
import scipy.optimize as so

DIMENSIONS = [3, 4, 5]
_START = time.time()
DEADLINE = _START + 140.0

def _remaining():
    return DEADLINE - time.time()

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _recon(U, V, W):
    return np.einsum('al,bl,cl->abc', U, V, W)

def _res(U, V, W, T):
    return np.linalg.norm(_recon(U, V, W) - T)

def _als(U, V, W, T, iters=500, tol=1e-14):
    r = U.shape[1]
    eye = np.eye(r)
    prev = _res(U, V, W, T)
    for it in range(iters):
        if _remaining() < 8: 
            break
        # Update U
        # Minimize ||T - sum_l U_l V_l W_l^T||_F
        # For fixed V, W, U is solution to linear system
        # (V^T V * W^T W) U^T = (T * V * W)^T
        G_U = (V.T @ V) * (W.T @ W)
        G_U += 1e-12 * (np.trace(G_U) / r) * eye
        # rhs: einsum('abc,bl,cl->al', T, V, W)
        rhs_U = np.einsum('abc,bl,cl->al', T, V, W)
        try:
            U = np.linalg.solve(G_U, rhs_U.T).T
        except np.linalg.LinAlgError:
            U = np.linalg.lstsq(G_U, rhs_U.T, rcond=None)[0].T
            
        # Update V
        G_V = (U.T @ U) * (W.T @ W)
        G_V += 1e-12 * (np.trace(G_V) / r) * eye
        rhs_V = np.einsum('abc,al,cl->bl', T, U, W)
        try:
            V = np.linalg.solve(G_V, rhs_V.T).T
        except np.linalg.LinAlgError:
            V = np.linalg.lstsq(G_V, rhs_V.T, rcond=None)[0].T
            
        # Update W
        G_W = (U.T @ U) * (V.T @ V)
        G_W += 1e-12 * (np.trace(G_W) / r) * eye
        rhs_W = np.einsum('abc,al,bl->cl', T, U, V)
        try:
            W = np.linalg.solve(G_W, rhs_W.T).T
        except np.linalg.LinAlgError:
            W = np.linalg.lstsq(G_W, rhs_W.T, rcond=None)[0].T
            
        cur = _res(U, V, W, T)
        if abs(prev - cur) < tol:
            prev = cur
            break
        prev = cur
    return U, V, W, cur

def _polish(U, V, W, T, max_iter=1000):
    # Flatten U, V, W into a single vector x
    # x = [U_flat, V_flat, W_flat]
    U0, V0, W0 = U, V, W
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def objective(x):
        U = x[:U0.size].reshape(U0.shape)
        V = x[U0.size:U0.size+V0.size].reshape(V0.shape)
        W = x[U0.size+V0.size:].reshape(W0.shape)
        recon = _recon(U, V, W)
        return np.linalg.norm(recon - T)
        
    try:
        res = so.minimize(objective, x0, method='BFGS', options={'maxiter': max_iter, 'gtol': 1e-12})
        if res.success or res.fun < 1e-10:
            U = res.x[:U0.size].reshape(U0.shape)
            V = res.x[U0.size:U0.size+V0.size].reshape(V0.shape)
            W = res.x[U0.size+V0.size:].reshape(W0.shape)
            return U, V, W, res.fun
    except Exception:
        pass
    return U0, V0, W0, _res(U0, V0, W0, T)

def _snap_half(U, V, W, T, tol=1e-8):
    # Try rounding to half-integers
    Us = np.round(U * 2) / 2
    Vs = np.round(V * 2) / 2
    Ws = np.round(W * 2) / 2
    if _res(Us, Vs, Ws, T) < tol:
        return Us, Vs, Ws, True
    return None, None, None, False

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    rng = np.random.default_rng(42)
    
    # Candidate ranks: Start from a reasonable lower bound.
    # R(3,4,5) is likely < 60. Let's try 40-60.
    # We search from low to high to find the minimum.
    ranks = list(range(40, 61))
    
    best = None
    for r in ranks:
        if _remaining() < 15:
            break
            
        # Number of restarts: more for lower ranks, less for higher
        nrestart = max(5, min(20, 1000 // r))
        
        found_at_rank = False
        for it in range(nrestart):
            if _remaining() < 10:
                break
                
            # Initialization strategy
            if it % 3 == 0:
                U = rng.standard_normal((n*m, r))
                V = rng.standard_normal((m*p, r))
                W = rng.standard_normal((p*n, r))
            elif it % 3 == 1:
                # Sparse initialization
                U = rng.standard_normal((n*m, r)) * (rng.random((n*m, r)) < 0.2)
                V = rng.standard_normal((m*p, r)) * (rng.random((m*p, r)) < 0.2)
                W = rng.standard_normal((p*n, r)) * (rng.random((p*n, r)) < 0.2)
            else:
                # Scaled random
                U = rng.standard_normal((n*m, r)) * 0.1
                V = rng.standard_normal((m*p, r)) * 0.1
                W = rng.standard_normal((p*n, r)) * 0.1
                
            U, V, W, res = _als(U, V, W, T, iters=300)
            
            if res < 1e-6:
                # Polish
                U, V, W, res = _polish(U, V, W, T, max_iter=500)
                
            if res < 1e-8:
                # Try to snap
                Us, Vs, Ws, ok = _snap_half(U, V, W, T, tol=1e-7)
                if ok:
                    found_at_rank = True
                    if best is None or r < best[0]:
                        best = (r, res, Us, Vs, Ws)
                    break # Found at this rank, move to next (lower) rank? 
                    # Wait, we are iterating ranks in ascending order. 
                    # If we find a solution at rank r, it's the minimum found so far.
                    # We should break the inner loop and continue to next rank?
                    # No, if we find a solution at rank r, we should check if a lower rank works?
                    # We are going 40, 41, ... 
                    # If we find at 50, we should stop? No, we want the MINIMUM.
                    # So we should search 40, 41... and stop at the first success.
                    # But what if 40 fails and 41 succeeds? Then 41 is the answer.
                    # So we should break the OUTER loop as well?
                    # Yes, if we find a valid decomposition at rank r, and we are searching in ascending order,
                    # then r is the minimal rank found. 
                    # However, it's possible that a lower rank exists but we failed to find it due to bad initialization.
                    # To be safe, we can continue a bit, but given time constraints, breaking is usually acceptable 
                    # if we do enough restarts for lower ranks.
                    # Let's break the outer loop too to save time, assuming our search for lower ranks was exhaustive enough.
                    pass
        
        if found_at_rank:
            # We found a solution at rank r. Since we iterate ascending, this is the min.
            # We can break the outer loop to return this result.
            break

    if best is not None:
        r, res, U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return _schoolbook(n, m, p)
