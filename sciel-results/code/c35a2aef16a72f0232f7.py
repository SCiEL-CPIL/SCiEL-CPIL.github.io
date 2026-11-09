import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 6]
_DEADLINE_SECONDS = 135  # Safety margin below 150s kill

def _make_residual(n, m, p, R):
    """Create residual function for tensor decomposition of size n x m x p with rank R."""
    def residual(x):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:2*n*m*R].reshape(m*p, R)
        W = x[2*n*m*R:].reshape(p*n, R)
        
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return (T_calc - 1.0).flatten()
    return residual

def _verify_half_integer(U, V, W, n, m, p, R):
    """Verify exact decomposition with half-integer entries."""
    U_int = np.round(U * 2).astype(np.int64)
    V_int = np.round(V * 2).astype(np.int64)
    W_int = np.round(W * 2).astype(np.int64)
    
    U_t = U_int.reshape(n, m, R)
    V_t = V_int.reshape(m, p, R)
    W_t = W_int.reshape(p, n, R)
    
    T_calc = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    return np.all(T_calc == 8)  # 2^3 = 8

def _schoolbook_init(n, m, p, R):
    """Create structured initialization near schoolbook pattern."""
    n_vars = (n*m + m*p + p*n) * R
    x0 = np.zeros(n_vars)
    
    U_init = np.zeros((n*m, R))
    V_init = np.zeros((m*p, R))
    W_init = np.zeros((p*n, R))
    
    # Fill in schoolbook pattern for min(R, n*m*p) terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= R:
                    break
                U_init[i*m+j, col] = 1.0
                V_init[j*p+k, col] = 1.0
                W_init[k*n+i, col] = 1.0
                col += 1
    
    x0[:n*m*R] = U_init.flatten()
    x0[n*m*R:2*n*m*R] = V_init.flatten()
    x0[2*n*m*R:] = W_init.flatten()
    return x0

def _try_rank(n, m, p, R, residual, t0, deadline, max_restarts=30):
    """Try to find exact half-integer decomposition for given rank R."""
    n_vars = (n*m + m*p + p*n) * R
    
    for restart in range(max_restarts):
        if time.time() - t0 > deadline:
            return None
        
        if restart == 0:
            x0 = _schoolbook_init(n, m, p, R)
        elif restart == 1:
            # Perturbed schoolbook
            x0 = _schoolbook_init(n, m, p, R)
            x0 += np.random.normal(0, 0.3, n_vars)
        elif restart == 2:
            # Different structured init
            x0 = np.zeros(n_vars)
            U_init = np.zeros((n*m, R))
            V_init = np.zeros((m*p, R))
            W_init = np.zeros((p*n, R))
            # Try diagonal-like pattern
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = (i * m * p + j * p + k) % R
                        U_init[i*m+j, idx] = 1.0
                        V_init[j*p+k, idx] = 1.0
                        W_init[k*n+i, idx] = 1.0
            x0[:n*m*R] = U_init.flatten()
            x0[n*m*R:2*n*m*R] = V_init.flatten()
            x0[2*n*m*R:] = W_init.flatten()
        else:
            # Random inits with varying scales
            scale = 0.3 + 0.5 * np.random.rand()
            x0 = np.random.uniform(-scale, scale, n_vars)
        
        try:
            # Phase 1: Coarse optimization
            res1 = least_squares(residual, x0, method='trf', x_scale='jac',
                                max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            
            if res1.cost < 1e-8:
                # Phase 2: Polish to high precision
                res2 = least_squares(residual, res1.x, method='trf', x_scale='jac',
                                    max_nfev=3000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                res = res2
            else:
                res = res1
            
            if res.cost < 1e-10:
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R:2*n*m*R].reshape(m*p, R)
                W = res.x[2*n*m*R:].reshape(p*n, R)
                
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if _verify_half_integer(U_r, V_r, W_r, n, m, p, R):
                    return (U_r, V_r, W_r)
        except Exception:
            continue
    
    return None

def construct():
    t0 = time.time()
    n, m, p = 3, 3, 6
    deadline = t0 + _DEADLINE_SECONDS
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Try ranks from lower bound upward
    # Known: 3x3x3 has rank 23 (Strassen-like), so 3x3x6 might be around 46 or better
    
    best_result = None
    
    # Try a range of ranks. Start from reasonable estimates.
    # For 3x3x6, we can think of it as two 3x3x3 problems, so rank <= 2*23 = 46
    # But there might be better decompositions.
    
    for R in range(18, 55):
        if time.time() - t0 > deadline - 10:
            break
        
        residual = _make_residual(n, m, p, R)
        result = _try_rank(n, m, p, R, residual, t0, deadline - 5, max_restarts=20)
        
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
