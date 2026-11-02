"""Improved program: log-objective + analytic gradients + BFGS + Nelder-Mead + polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite
import time

_DEADLINE = 190  # safety margin before 210s kill

def _hermite_h(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 7:
        # H_7(x) = 128x^7 - 4480x^5 + 40320x^3 - 67200x
        return 128*x**7 - 4480*x**5 + 40320*x**3 - 67200*x
    elif n == 8:
        return 256*x**8 - 4480*x**6 + 26880*x**4 - 67200*x**2 + 40320
    elif n == 11:
        # H_11(x) = 2048x^11 - 147456x^9 + 3133440x^7 - 30198720x^5 + 122305920x^3 - 145152000x
        return (2048*x**11 - 147456*x**9 + 3133440*x**7 
                - 30198720*x**5 + 122305920*x**3 - 145152000*x)
    elif n == 12:
        return (4096*x**12 - 147456*x**10 + 2580480*x**8 
                - 20321280*x**6 + 81729600*x**4 
                - 172972800*x**2 + 103219200)
    else:
        raise ValueError(f"Hermite H_{n} not implemented")

_quad_data = None

def _get_quad_data():
    global _quad_data
    if _quad_data is None:
        n_quad = 200
        x_quad, w_quad = roots_hermite(n_quad)
        
        H0 = _hermite_h(0, x_quad)
        H4 = _hermite_h(4, x_quad)
        H8 = _hermite_h(8, x_quad)
        H12 = _hermite_h(12, x_quad)
        
        # Derivatives: H_n'(x) = 2n H_{n-1}(x)
        dH0 = np.zeros_like(x_quad)
        dH4 = 8 * _hermite_h(3, x_quad)
        dH8 = 16 * _hermite_h(7, x_quad)
        dH12 = 24 * _hermite_h(11, x_quad)
        
        x2 = x_quad**2
        x2w = x2 * w_quad
        
        _quad_data = (x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12, x2, x2w)
    return _quad_data

H0_0 = 1.0
H4_0 = 12.0
H8_0 = 40320.0
H12_0 = 103219200.0

def _compute_bound_and_grad(coeffs):
    """
    Compute log(var_x) + log(var_k) and its gradient.
    Returns (log_val, grad) or (inf, None).
    """
    a0, a4, a8 = coeffs
    
    # d = -(a0 + 12*a4 + 40320*a8) / 103219200
    d = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12, x2, x2w = _get_quad_data()
    
    # P = a0*H0 + a4*H4 + a8*H8 + d*H12
    # dP = a0*dH0 + a4*dH4 + a8*dH8 + d*dH12
    # Q = dP - x*P  (this is the "derivative term" for the Gaussian-weighted function)
    
    # For each coefficient ci in {a0, a4, a8}:
    # P_i = Hi + (ddi) * H12  where ddi = -Hi(0)/H12(0)
    # Q_i = dHi - x*Hi + ddi*(dH12 - x*H12)
    
    d_d0 = -H0_0 / H12_0
    d_d4 = -H4_0 / H12_0
    d_d8 = -H8_0 / H12_0
    
    P0 = H0 + d_d0 * H12
    P4 = H4 + d_d4 * H12
    P8 = H8 + d_d8 * H12
    
    Q_base = dH12 - x_quad * H12
    Q0 = dH0 + d_d0 * Q_base
    Q4 = dH4 + d_d4 * Q_base
    Q8 = dH8 + d_d8 * Q_base
    
    # P = a0*P0 + a4*P4 + a8*P8
    P = a0*P0 + a4*P4 + a8*P8
    P2 = P * P
    
    # Q = a0*Q0 + a4*Q4 + a8*Q8
    Q = a0*Q0 + a4*Q4 + a8*Q8
    Q2 = Q * Q
    
    norm2 = np.sum(w_quad * P2)
    if norm2 <= 1e-15:
        return float('inf'), None
    
    var_x = np.sum(x2w * P2) / norm2
    if var_x <= 1e-15:
        return float('inf'), None
    
    norm_deriv2 = np.sum(w_quad * Q2)
    if norm_deriv2 <= 1e-15:
        return float('inf'), None
    
    var_k = norm_deriv2 / norm2
    
    log_val = np.log(var_x) + np.log(var_k)
    
    # Gradients:
    # d(log var_x)/dc = (1/var_x) * d(var_x)/dc
    # var_x = Sx / S0 where Sx = sum(w*x2*P2), S0 = sum(w*P2)
    # d(var_x)/dc = (dSx*S0 - Sx*dS0) / S0^2
    # dS0/dc = 2*sum(w*P*Pi), dSx/dc = 2*sum(w*x2*P*Pi)
    
    # d(log var_k)/dc = (1/var_k) * d(var_k)/dc
    # var_k = Sd / S0 where Sd = sum(w*Q2)
    # d(var_k)/dc = (dSd*S0 - Sd*dS0) / S0^2
    # dSd/dc = 2*sum(w*Q*Qi)
    
    dS0_0 = 2.0 * np.sum(w_quad * P * P0)
    dS0_4 = 2.0 * np.sum(w_quad * P * P4)
    dS0_8 = 2.0 * np.sum(w_quad * P * P8)
    
    dSx_0 = 2.0 * np.sum(x2w * P * P0)
    dSx_4 = 2.0 * np.sum(x2w * P * P4)
    dSx_8 = 2.0 * np.sum(x2w * P * P8)
    
    dSd_0 = 2.0 * np.sum(w_quad * Q * Q0)
    dSd_4 = 2.0 * np.sum(w_quad * Q * Q4)
    dSd_8 = 2.0 * np.sum(w_quad * Q * Q8)
    
    Sx = np.sum(x2w * P2)
    Sd = norm_deriv2
    
    # d(log var_x)/dc = (dSx*S0 - Sx*dS0) / (S0 * Sx)  [since var_x = Sx/S0, 1/var_x * d(var_x) = S0/Sx * (dSx*S0 - Sx*dS0)/S0^2 = (dSx*S0 - Sx*dS0)/(S0*Sx)]
    # Actually: d(log(Sx/S0))/dc = dSx/Sx - dS0/S0
    gx0 = dSx_0 / Sx - dS0_0 / norm2
    gx4 = dSx_4 / Sx - dS0_4 / norm2
    gx8 = dSx_8 / Sx - dS0_8 / norm2
    
    # d(log var_k)/dc = d(log(Sd/S0))/dc = dSd/Sd - dS0/S0
    gk0 = dSd_0 / Sd - dS0_0 / norm2
    gk4 = dSd_4 / Sd - dS0_4 / norm2
    gk8 = dSd_8 / Sd - dS0_8 / norm2
    
    grad = np.array([gx0 + gk0, gx4 + gk4, gx8 + gk8])
    
    return log_val, grad

def _compute_bound(coeffs):
    """Original bound (product) for compatibility."""
    if time.time() > _DEADLINE:
        return float('inf')
    val, _ = _compute_bound_and_grad(coeffs)
    if val == float('inf'):
        return float('inf')
    return np.exp(val)

def construct():
    start_time = time.time()
    
    seed = np.array([-113/100, 1/25, 1.0/3240])
    
    best_result = seed.copy()
    best_log_val = _compute_bound_and_grad(seed)[0]
    
    if best_log_val == float('inf'):
        best_log_val = 999.0
    
    def log_objective(params):
        if time.time() > _DEADLINE:
            return float('inf')
        val, _ = _compute_bound_and_grad(params)
        return val
    
    def log_objective_grad(params):
        if time.time() > _DEADLINE:
            return float('inf'), np.zeros(3)
        val, grad = _compute_bound_and_grad(params)
        if val == float('inf'):
            return 999.0, np.zeros(3)
        return val, grad
    
    # Phase 1: BFGS from seed with analytic gradients
    try:
        result = minimize(log_objective_grad, seed, method='L-BFGS-B',
                         jac=True, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.fun < best_log_val:
            best_log_val = result.fun
            best_result = result.x.copy()
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead from seed (robust, no gradient)
    try:
        result = minimize(log_objective, seed, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < best_log_val:
            best_log_val = result.fun
            best_result = result.x.copy()
    except Exception:
        pass
    
    # Phase 3: BFGS from best so far
    if time.time() < _DEADLINE - 40:
        try:
            result = minimize(log_objective_grad, best_result, method='L-BFGS-B',
                             jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < best_log_val:
                best_log_val = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
    
    # Phase 4: Random restarts
    np.random.seed(42)
    if time.time() < _DEADLINE - 30:
        for i in range(15):
            if time.time() > _DEADLINE - 30:
                break
            # Generate diverse starting points
            scale = 0.5 + i * 0.15
            random_start = np.random.uniform(-scale, scale, 3)
            # Also try perturbations of best
            if i % 3 == 0:
                random_start = best_result + np.random.uniform(-0.1, 0.1, 3)
            elif i % 3 == 1:
                random_start = best_result + np.random.uniform(-0.5, 0.5, 3)
            
            try:
                result = minimize(log_objective_grad, random_start, method='L-BFGS-B',
                                 jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                if result.fun < best_log_val:
                    best_log_val = result.fun
                    best_result = result.x.copy()
            except Exception:
                pass
            
            # Nelder-Mead from random start
            if time.time() < _DEADLINE - 30:
                try:
                    result = minimize(log_objective, random_start, method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                    if result.fun < best_log_val:
                        best_log_val = result.fun
                        best_result = result.x.copy()
                except Exception:
                    pass
    
    # Phase 5: Final high-precision polish
    if time.time() < _DEADLINE - 10:
        try:
            result = minimize(log_objective_grad, best_result, method='L-BFGS-B',
                             jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < best_log_val:
                best_log_val = result.fun
                best_result = result.x.copy()
        except Exception:
            pass
        
        # One more Nelder-Mead polish with very tight tolerance
        if time.time() < _DEADLINE - 5:
            try:
                result = minimize(log_objective, best_result, method='Nelder-Mead',
                                 options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                if result.fun < best_log_val:
                    best_log_val = result.fun
                    best_result = result.x.copy()
            except Exception:
                pass
    
    return np.array(best_result)

# EVOLVE-BLOCK-END
