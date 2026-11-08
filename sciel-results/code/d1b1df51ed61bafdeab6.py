import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target multiplication tensor T[u, v, w]
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    T_flat = T.reshape(-1)

    def reconstruct(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Rec[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def error_fn(U, V, W):
        Rec = reconstruct(U, V, W)
        return np.linalg.norm(Rec - T)

    def snap_to_half(X):
        # Snap to nearest 0.5
        return np.round(X * 2) / 2

    def solve_als(U, V, W, max_iter=50, tol=1e-8):
        R = U.shape[1]
        current_err = float('inf')
        
        for it in range(max_iter):
            # Update W
            # A_W (nm*mp, R), b_w (nm*mp,)
            P_UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            A_W = P_UV.reshape(nm * mp, R)
            W_new = np.zeros((pn, R), dtype=np.float64)
            for w in range(pn):
                b = T[:, :, w].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                W_new[w] = sol
            W = W_new
            
            # Update V
            P_UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
            A_V = P_UW.reshape(nm * pn, R)
            V_new = np.zeros((mp, R), dtype=np.float64)
            for v in range(mp):
                b = T[:, v, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                V_new[v] = sol
            V = V_new
            
            # Update U
            P_VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
            A_U = P_VW.reshape(mp * pn, R)
            U_new = np.zeros((nm, R), dtype=np.float64)
            for u in range(nm):
                b = T[u, :, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                U_new[u] = sol
            U = U_new
            
            err = error_fn(U, V, W)
            if err < current_err:
                current_err = err
            else:
                # If error increases, might be oscillating. 
                # For ALS on this problem, it usually monotonically decreases or gets stuck.
                # If stuck, break.
                if it > 5:
                    break
            
            if err < tol:
                break
                
        return U, V, W, current_err

    def polish(U, V, W, max_iter=100):
        # Flatten variables
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residuals(x):
            n_U = nm * R
            n_V = mp * R
            n_W = pn * R
            U_r = x[:n_U].reshape(nm, R)
            V_r = x[n_U:n_U+n_V].reshape(mp, R)
            W_r = x[n_U+n_V:].reshape(pn, R)
            Rec = reconstruct(U_r, V_r, W_r)
            return (Rec - T).flatten()
        
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=max_iter)
            x_opt = result.x
            U_opt = x_opt[:nm*R].reshape(nm, R)
            V_opt = x_opt[nm*R:nm*R+mp*R].reshape(mp, R)
            W_opt = x_opt[nm*R+mp*R:].reshape(pn, R)
            err = np.linalg.norm(residuals(x_opt))
            return U_opt, V_opt, W_opt, err
        except Exception:
            return U, V, W, error_fn(U, V, W)

    deadline = time.time() + 140
    best_solution = None
    best_err = float('inf')
    
    # Try rank 44
    R = 44
    
    # Number of restarts depends on time.
    # Each restart: ALS (fast) + Polish (slower).
    # Let's try up to 20 restarts.
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Initialization
        # Random init
        U = np.random.randn(nm, R) * 1.0
        V = np.random.randn(mp, R) * 1.0
        W = np.random.randn(pn, R) * 1.0
        
        # ALS
        U, V, W, err_als = solve_als(U, V, W, max_iter=30, tol=1e-6)
        
        if err_als < 1e-3:
            # Polish
            U, V, W, err_pol = polish(U, V, W, max_iter=200)
            
            if err_pol < 1e-8:
                # Try snapping to half-integers
                U_snap = snap_to_half(U)
                V_snap = snap_to_half(V)
                W_snap = snap_to_half(W)
                
                err_snap = error_fn(U_snap, V_snap, W_snap)
                
                if err_snap < 1e-8:
                    # Found a valid half-integer solution
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                else:
                    # If snapping fails, return the float solution (assuming evaluator is tolerant)
                    # Or try other rounding strategies?
                    # For now, return float if it's extremely accurate
                    if err_pol < 1e-10:
                         return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Track best float solution just in case
        if err_pol < best_err:
            best_err = err_pol
            best_solution = (U.copy(), V.copy(), W.copy())

    # If we found a very good float solution but not snapped, return it
    if best_solution is not None and best_err < 1e-8:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook decomposition of rank n*m*p = 48
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
