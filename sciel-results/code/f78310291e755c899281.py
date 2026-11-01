import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195.0  # 15s safety margin before 210s kill

def _fact_ratio(n):
    """Compute (2n)! / (4^n n!) * sqrt(pi) using log-sum for numerical stability."""
    if n < 0:
        return 0.0
    if n == 0:
        return np.sqrt(np.pi)
    # Use log-gamma for efficiency and stability
    from math import lgamma
    log_val = lgamma(2*n + 1) - n * np.log(4.0) - lgamma(n + 1) + 0.5 * np.log(np.pi)
    return np.exp(log_val)

# Precompute I[n] = integral x^{2n} e^{-x^2} dx for n=0..13
_I_cache = None
def _get_I():
    global _I_cache
    if _I_cache is None:
        _I_cache = np.array([_fact_ratio(n) for n in range(14)])
    return _I_cache

def _compute_bound(params):
    """
    Compute the uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) for
    g(x) = P(x) * exp(-x^2/2) where P is a polynomial combination of H0,H4,H8,H12.
    
    Returns the bound value (the upper bound C4 we want to minimize).
    """
    a0, a4, a8 = params
    # H12(0) = 10395 (physicist's Hermite)
    # g(0) = a0*1 + a4*3 + a8*105 + a12*10395 = 0
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 10395.0
    
    # Build polynomial P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 166320x^4 - 302400x^2 + 10395
    
    p = np.zeros(7)
    # p0 (x^0 term)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 10395.0*a12
    # p1 (x^2 term)
    p[1] = -6.0*a4 - 420.0*a8 - 302400.0*a12
    # p2 (x^4 term)
    p[2] = a4 + 210.0*a8 + 166320.0*a12
    # p3 (x^6 term)
    p[3] = -28.0*a8 - 30240.0*a12
    # p4 (x^8 term)
    p[4] = a8 + 2520.0*a12
    # p5 (x^10 term)
    p[5] = -84.0*a12
    # p6 (x^12 term)
    p[6] = a12
    
    I = _get_I()
    
    # Compute moments m_k = integral x^{2k} |P|^2 e^{-x^2} dx for k=0,1,2
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for k in range(3):
        for i in range(7):
            pi = p[i]
            if pi == 0.0:
                continue
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    m[k] += pi * p[j] * I[idx]
    
    m0, m2 = m[0], m[1]
    
    # Fourier side: G(xi) = sqrt(2*pi) * P(-i*xi) * e^{-xi^2/2}
    # P(-i*xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k}
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0) ** k)
    
    # n_k = 2*pi * integral xi^{2k} |Q|^2 e^{-xi^2} dxi = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for k in range(3):
        for i in range(7):
            qi = q[i]
            if qi == 0.0:
                continue
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0 * np.pi * qi * q[j] * I[idx]
    
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)

def _objective(params):
    return _compute_bound(params)

def _objective_grad(params):
    """Compute bound and analytic gradient via finite differences (robust fallback)."""
    bound = _compute_bound(params)
    if not np.isfinite(bound) or bound > 1e10:
        return bound, np.array([1e10, 1e10, 1e10])
    
    # Central differences for gradient
    eps = 1e-7
    grad = np.zeros(3)
    for i in range(3):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        f_plus = _compute_bound(params_plus)
        f_minus = _compute_bound(params_minus)
        if np.isfinite(f_plus) and np.isfinite(f_minus):
            grad[i] = (f_plus - f_minus) / (2.0 * eps)
        else:
            grad[i] = 0.0
    
    return bound, grad

def _try_minimize(start, method='L-BFGS-B', maxiter=3000):
    """Try to minimize from a starting point, return (value, params) or (inf, None)."""
    if time.time() > _DEADLINE:
        return np.inf, None
    
    try:
        if method == 'L-BFGS-B':
            res = minimize(_objective_grad, start, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-14})
        elif method == 'Nelder-Mead':
            res = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-17})
        else:
            res = minimize(_objective, start, method=method,
                          options={'maxiter': maxiter})
        
        if res.fun < np.inf and np.isfinite(res.fun):
            return res.fun, res.x.copy()
    except Exception:
        pass
    return np.inf, None

def construct():
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Initial bound at seed
    seed_val = _compute_bound(seed)
    
    best_val = seed_val
    best_params = seed.copy()
    
    # Multiple starting points for global search
    starts = [
        seed,
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([0.5, 0.05, 0.0005]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-1.2, 0.06, 0.0006]),
        np.array([-2.5, 0.12, 0.0012]),
    ]
    
    # Add random starts
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.8, 3))
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        val, params = _try_minimize(start, method='L-BFGS-B', maxiter=2000)
        if val < best_val:
            best_val = val
            best_params = params
    
    # Phase 2: Nelder-Mead polish from best
    if time.time() < _DEADLINE - 15:
        val, params = _try_minimize(best_params, method='Nelder-Mead', maxiter=50000)
        if val < best_val:
            best_val = val
            best_params = params
    
    # Phase 3: Local 1D coordinate refinement (high precision polish)
    if time.time() < _DEADLINE - 10:
        for _ in range(3):  # 3 rounds of coordinate descent
            for coord in range(3):
                if time.time() > _DEADLINE - 5:
                    break
                
                def _1d_obj(x):
                    p = best_params.copy()
                    p[coord] = x
                    return _compute_bound(p)
                
                # Bracket around current value
                x0 = best_params[coord]
                # Try a small range
                x_min = x0 - 0.5 * (1.0 + abs(x0))
                x_max = x0 + 0.5 * (1.0 + abs(x0))
                
                try:
                    res = minimize_scalar(_1d_obj, bounds=(x_min, x_max), method='bounded',
                                        options={'maxiter': 1000, 'xatol': 1e-14})
                    if res.success and res.fun < best_val:
                        best_params[coord] = res.x
                        best_val = res.fun
                except Exception:
                    pass
    
    # Phase 4: Final L-BFGS-B polish with very tight tolerance
    if time.time() < _DEADLINE - 5:
        val, params = _try_minimize(best_params, method='L-BFGS-B', maxiter=5000)
        if val < best_val:
            best_val = val
            best_params = params
    
    return np.array(best_params)
