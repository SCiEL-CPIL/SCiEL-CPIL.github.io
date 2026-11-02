import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T[i,j,k]
    T = np.zeros((nm, mp, pn))
    for i in range(nm):
        a, b = divmod(i, m)
        for j in range(mp):
            c, d = divmod(j, p)
            k = d * n + a
            if b == c:
                T[i, j, k] = 1.0
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    def compute_residual(U, V, W):
        """Compute the exact residual sum of squares."""
        product = np.einsum('ir,jr,kr->ijk', U, V, W)
        res = product - T
        return np.sum(res ** 2)
    
    def compute_grads(U, V, W):
        """Compute gradients of residual sum of squares."""
        product = np.einsum('ir,jr,kr->ijk', U, V, W)
        res = product - T
        
        # grad_U[i,r] = 2 * sum_{j,k} res[i,j,k] * V[j,r] * W[k,r]
        C1 = np.einsum('ijk,kr->ijr', res, W)
        grad_U = 2.0 * np.einsum('jr,ijr->ir', V, C1)
        
        # grad_V[j,r] = 2 * sum_{i,k} res[i,j,k] * U[i,r] * W[k,r]
        C2 = np.einsum('ijk,kr->ijr', res, U)
        grad_V = 2.0 * np.einsum('ir,ijr->jr', W, C2)
        
        # grad_W[k,r] = 2 * sum_{i,j} res[i,j,k] * U[i,r] * V[j,r]
        C3 = np.einsum('ijk,jr->ikr', res, V)
        grad_W = 2.0 * np.einsum('ir,ikr->kr', U, C3)
        
        return grad_U, grad_V, grad_W
    
    def verify(U, V, W, tol=1e-8):
        """Verify if decomposition is exact."""
        product = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.all(np.abs(product - T) < tol)
    
    def snap_to_half_int(U, V, W):
        """Snap continuous factors to nearest half-integer grid."""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap
    
    def polish(U, V, W, max_iter=1000):
        """High-precision polish using Newton-like iterations."""
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            product = np.einsum('ir,jr,kr->ijk', U, V, W)
            res = product - T
            if np.max(np.abs(res)) < 1e-14:
                break
            # Simple gradient step with adaptive learning rate
            gU, gV, gW = compute_grads(U, V, W)
            lr = 1e-4
            U = U - lr * gU
            V = V - lr * gV
            W = W - lr * gW
        return U, V, W
    
    def try_optimize(R, n_restarts=12, max_iter=600):
        """Try to find rank R decomposition using L-BFGS optimization."""
        best_res = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random values
            scale = 0.5
            U = np.random.randn(nm, R) * scale
            V = np.random.randn(mp, R) * scale
            W = np.random.randn(pn, R) * scale
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                if time.time() > deadline - 10:
                    return 1e10
                Ux = x[:nm*R].reshape(nm, R)
                Vx = x[nm*R:nm*R+mp*R].reshape(mp, R)
                Wx = x[nm*R+mp*R:].reshape(pn, R)
                return compute_residual(Ux, Vx, Wx)
            
            def gradient(x):
                if time.time() > deadline - 10:
                    return np.zeros_like(x)
                Ux = x[:nm*R].reshape(nm, R)
                Vx = x[nm*R:nm*R+mp*R].reshape(mp, R)
                Wx = x[nm*R+mp*R:].reshape(pn, R)
                gU, gV, gW = compute_grads(Ux, Vx, Wx)
                return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            
            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                options={'maxiter': max_iter, 'maxfun': 3000,
                                        'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30})
                
                Ux = result.x[:nm*R].reshape(nm, R)
                Vx = result.x[nm*R:nm*R+mp*R].reshape(mp, R)
                Wx = result.x[nm*R+mp*R:].reshape(pn, R)
                
                # Try snapping to half-integers
                U_snap, V_snap, W_snap = snap_to_half_int(Ux, Vx, Wx)
                res_snap = compute_residual(U_snap, V_snap, W_snap)
                
                res_cont = compute_residual(Ux, Vx, Wx)
                
                # Use the better of continuous and snapped
                if res_snap < res_cont:
                    Ux, Vx, Wx = U_snap, V_snap, W_snap
                    res = res_snap
                else:
                    res = res_cont
                
                if res < best_res:
                    best_res = res
                    best_params = (Ux.copy(), Vx.copy(), Wx.copy())
                
                if res < 1e-12:
                    break
                    
            except Exception:
                continue
        
        return best_res, best_params
    
    # Try ranks from 83 down
    min_rank = 60
    for R in range(rank_sb - 1, min_rank - 1, -1):
        if time.time() > deadline - 15:
            break
        
        res, params = try_optimize(R, n_restarts=12, max_iter=600)
        
        if res < 1e-10 and params is not None:
            U, V, W = params
            # Polish with high precision
            U_pol, V_pol, W_pol = polish(U, V, W, max_iter=500)
            res_pol = compute_residual(U_pol, V_pol, W_pol)
            
            if res_pol < 1e-10:
                # Try snapping the polished result
                U_snap, V_snap, W_snap = snap_to_half_int(U_pol, V_pol, W_pol)
                res_snap = compute_residual(U_snap, V_snap, W_snap)
                
                if res_snap < 1e-10:
                    U, V, W = U_snap, V_snap, W_snap
                else:
                    U, V, W = U_pol, V_pol, W_pol
                
                if verify(U, V, W, tol=1e-8):
                    best_rank = R
                    best_U, best_V, best_W = U, V, W
                    break
    
    if best_rank < rank_sb:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
