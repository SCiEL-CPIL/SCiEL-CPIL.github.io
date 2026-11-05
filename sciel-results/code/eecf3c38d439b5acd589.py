import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh_tridiagonal

def construct():
    """
    Optimizes diagonal and off-diagonal entries of a 12x12 symmetric tridiagonal
    matrix to minimize the variance of its eigenvalues, serving as a proxy for
    minimizing the C4 uncertainty bound.
    """
    N = 12
    
    # Initial guess based on standard Laguerre Jacobi matrix
    d_init = np.array([2*i + 1 for i in range(N)], dtype=float)
    o_init = np.array([float(i) for i in range(1, N)], dtype=float)
    
    def cost(params):
        d = params[:N]
        o = params[N:]
        
        try:
            w, _ = eigh_tridiagonal(d, o, select='all')
        except Exception:
            return 1e10
            
        if np.any(w <= 0):
            penalty = np.sum(w[w <= 0]**2)
            return 1e10 + penalty
            
        # Objective: minimize variance of eigenvalues
        var = np.var(w)
        return var
    
    # Combine parameters: d (12) + o (11) = 23 parameters
    x0 = np.concatenate([d_init, o_init])
    
    # Bounds: diagonal entries should be positive, off-diagonals positive
    bounds = [(1.0, 100.0)] * N + [(0.1, 50.0)] * (N - 1)
    
    res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
    
    if res.success:
        d_opt = res.x[:N]
        o_opt = res.x[N:]
        w, _ = eigh_tridiagonal(d_opt, o_opt, select='all')
        if np.any(w <= 0):
            from scipy.special import roots_laguerre
            roots, _ = roots_laguerre(12)
            return [float(r) for r in roots]
        return [float(r) for r in w]
    else:
        # Fallback: try with just off-diagonal optimization
        def cost_offdiag(params):
            o = params
            d = d_init
            try:
                w, _ = eigh_tridiagonal(d, o, select='all')
            except Exception:
                return 1e10
            if np.any(w <= 0):
                return 1e10 + np.sum(w[w <= 0]**2)
            return np.var(w)
        
        x0_off = o_init.copy()
        bounds_off = [(0.1, 50.0)] * (N - 1)
        res2 = minimize(cost_offdiag, x0_off, method='L-BFGS-B', bounds=bounds_off,
                       options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
        
        if res2.success:
            o_opt = res2.x
            w, _ = eigh_tridiagonal(d_init, o_opt, select='all')
            if np.any(w <= 0):
                from scipy.special import roots_laguerre
                roots, _ = roots_laguerre(12)
                return [float(r) for r in roots]
            return [float(r) for r in w]
        else:
            from scipy.special import roots_laguerre
            roots, _ = roots_laguerre(12)
            return [float(r) for r in roots]
