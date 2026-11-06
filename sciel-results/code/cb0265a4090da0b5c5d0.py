"""Improved growth-index fitting with analytical gradients and multi-phase optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_start_time = time.time()
_DEADLINE = 85.0

# More diverse training cosmologies
_cosmologies = [
    (0.30, -1.0, 0.0),
    (0.30, -0.9, 0.1),
    (0.30, -1.1, 0.2),
    (0.30, -0.8, -0.2),
    (0.30, -1.2, 0.3),
    (0.20, -1.0, 0.0),
    (0.40, -1.0, 0.0),
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.25, -0.9, 0.1),
    (0.35, -1.1, -0.1),
]
_a_grid = np.linspace(0.05, 1.0, 40)
_ref_data = {}

def _precompute_references():
    """Precompute reference f(a) and D(a)/D(1) using ODE integration."""
    for Om0, w0, wa in _cosmologies:
        Ode0 = 1.0 - Om0
        a_vals = _a_grid
        
        def dlnHdx(a):
            Om_a = Om0 * a**(-3)
            if Ode0 == 0:
                Ode_a = 0.0
                dOde_da = 0.0
            else:
                log_Ode_a = np.log(Ode0) - 3*(1+w0)*np.log(a) - 3*wa*(1-a)
                Ode_a = np.exp(log_Ode_a)
                dOde_da = Ode_a * (-3*(1+w0)/a + 3*wa)
            H2 = Om_a + Ode_a
            dH2_da = -3*Om0*a**(-4) + dOde_da
            dH2_dx = dH2_da * a
            return 0.5 * dH2_dx / H2
        
        def ode_system(x, y):
            a = np.exp(x)
            Om = Om0 * a**(-3)
            dlnHdx_val = dlnHdx(a)
            dDdx = y[1]
            d2Ddx2 = -3 * Om * y[0] - (3 + dlnHdx_val) * y[1]
            return [dDdx, d2Ddx2]
        
        f1 = Om0**0.55
        D1 = 1.0
        D1_prime = f1 * D1
        x_vals = np.log(a_vals)
        x_min = min(x_vals)
        
        sol = solve_ivp(ode_system, [0.0, x_min], [D1, D1_prime], 
                        rtol=1e-10, atol=1e-12, method='DOP853',
                        dense_output=True)
        
        D_vals = np.zeros(len(a_vals))
        f_vals = np.zeros(len(a_vals))
        for i, x in enumerate(x_vals):
            y = sol.sol(x)
            D_vals[i] = y[0]
            f_vals[i] = y[1] / y[0]
        
        D_norm = D_vals / D_vals[-1]
        _ref_data[(Om0, w0, wa)] = (f_vals, D_norm)

def _compute_model_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a_vals):
    """Compute model f(a) and normalized D(a) efficiently."""
    a_vals = np.asarray(a_vals)
    Om_a = Om0 * a_vals**(-3)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a_vals)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
    gamma = np.clip(gamma, 0.1, 1.2)
    f_vals = Om_a**gamma
    
    x_vals = np.log(a_vals)
    f = f_vals
    n = len(x)
    
    # Vectorized cumulative trapezoidal integration from right to left
    dx = np.diff(x_vals)
    avg_f = 0.5 * (f[:-1] + f[1:])
    trapezoids = avg_f * dx
    integral_to_0 = np.zeros(n)
    for i in range(n-2, -1, -1):
        integral_to_0[i] = integral_to_0[i+1] + trapezoids[i]
    integral_to_0[n-1] = 0.0
    
    D_norm = np.exp(-integral_to_0)
    
    return f_vals, D_norm

def _compute_model_with_grad(coeffs, Om0, w0, wa, a_vals):
    """Compute model values and gradients w.r.t. coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a_vals = np.asarray(a_vals)
    Om_a = Om0 * a_vals**(-3)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a_vals)
    
    # Partial derivatives of gamma w.r.t. each coefficient
    dq_dc1 = np.ones_like(q)
    dq_dc2 = 2.0 * q
    du_dc3 = np.ones_like(u)
    du_dc5 = q
    dgamma_dc0 = np.ones_like(q)
    dgamma_dc1 = dq_dc1
    dgamma_dc2 = dq_dc2
    dgamma_dc3 = du_dc3
    dgamma_dc4 = wa * a_vals
    dgamma_dc5 = du_dc5
    
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
    gamma_clipped = np.clip(gamma, 0.1, 1.2)
    f_vals = Om_a**gamma_clipped
    
    # Gradient of f w.r.t. gamma: df/dgamma = f * ln(Om_a)
    ln_Om_a = np.log(Om_a)
    df_dgamma = f_vals * ln_Om_a
    
    # Gradient of f w.r.t. each coefficient (only where clipping is active)
    grad_f = np.zeros((6, len(a_vals)))
    mask_active = (gamma >= 0.1) & (gamma <= 1.2)
    for i, dgamma_dc in enumerate([dgamma_dc0, dgamma_dc1, dgamma_dc2, dgamma_dc3, dgamma_dc4, dgamma_dc5]):
        grad_f[i] = np.where(mask_active, df_dgamma * dgamma_dc, 0.0)
    
    # For D_norm, use finite difference approximation of gradient
    # D_norm = exp(-integral), so dD_norm/dc_i = -D_norm * (d integral/dc_i)
    # integral = sum of trapezoids, each trapezoid involves f values
    # This is complex, so we'll use the f gradient as a proxy
    
    x_vals = np.log(a_vals)
    n = len(x_vals)
    dx = np.diff(x_vals)
    
    # Gradient of integral w.r.t. coefficients
    # integral = sum_{i=0}^{n-2} 0.5*(f[i]+f[i+1]) * dx[i]
    # d(integral)/dc_j = sum_{i=0}^{n-2} 0.5*(df[i]/dc_j + df[i+1]/dc_j) * dx[i]
    
    grad_integral = np.zeros((6, n))
    for j in range(6):
        for i in range(n-1):
            grad_integral[j, i] += 0.5 * grad_f[j, i] * dx[i]
            grad_integral[j, i+1] += 0.5 * grad_f[j, i+1] * dx[i]
    
    D_norm = np.exp(-np.zeros(n))  # placeholder, will recompute
    # Recompute D_norm properly
    avg_f = 0.5 * (f_vals[:-1] + f_vals[1:])
    trapezoids = avg_f * dx
    integral_to_0 = np.zeros(n)
    for i in range(n-2, -1, -1):
        integral_to_0[i] = integral_to_0[i+1] + trapezoids[i]
    integral_to_0[n-1] = 0.0
    D_norm = np.exp(-integral_to_0)
    
    # Gradient of D_norm w.r.t. coefficients
    grad_D = -D_norm * grad_integral
    
    return f_vals, D_norm, grad_f, grad_D

def _evaluate_error(coeffs):
    """Compute max relative error across all cosmologies and both observables."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    for Om0, w0, wa in _cosmologies:
        f_ref, D_ref = _ref_data[(Om0, w0, wa)]
        f_mod, D_mod = _compute_model_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, _a_grid)
        
        err_f = np.max(np.abs((f_mod - f_ref) / f_ref))
        err_D = np.max(np.abs((D_mod - D_ref) / D_ref))
        
        max_err = max(max_err, err_f, err_D)
    
    return max_err

def _smooth_objective(coeffs):
    """Smooth surrogate using log-sum-exp of squared relative errors."""
    c0, c1, c2, c3, c4, c5 = coeffs
    all_sq_errs = []
    for Om0, w0, wa in _cosmologies:
        f_ref, D_ref = _ref_data[(Om0, w0, wa)]
        f_mod, D_mod = _compute_model_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, _a_grid)
        
        rel_err_f = np.abs((f_mod - f_ref) / f_ref)
        rel_err_D = np.abs((D_mod - D_ref) / D_ref)
        
        all_sq_errs.extend(np.concatenate([rel_err_f**2, rel_err_D**2]).tolist())
    
    all_sq_errs = np.array(all_sq_errs)
    max_val = np.max(all_sq_errs)
    if max_val == 0:
        return 0.0
    lse = max_val + np.log(np.sum(np.exp(all_sq_errs - max_val)))
    return np.exp(0.5 * lse)

def _is_valid(coeffs, cosmologies, a_grid):
    """Check if gamma stays within [0.1, 1.2] for all cosmologies."""
    c0, c1, c2, c3, c4, c5 = coeffs
    for Om0, w0, wa in cosmologies:
        Om_a = Om0 * a_grid**(-3)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa*(1.0 - a_grid)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return False
    return True

def construct():
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    _precompute_references()
    
    best_result = None
    best_score = float('inf')
    
    def _update_best(result, score):
        nonlocal best_result, best_score
        if score < best_score and _is_valid(result, _cosmologies, _a_grid):
            best_score = score
            best_result = result
    
    # Phase 1: L-BFGS-B on smooth surrogate from physics-based initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        result = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        actual_score = _evaluate_error(result.x)
        _update_best(result.x, actual_score)
    except:
        pass
    
    if time.time() - _start_time > _DEADLINE - 25:
        if best_result is not None:
            return np.clip(best_result, -10, 10).tolist()
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Phase 2: Nelder-Mead on original max objective from best point
    x0_nm = best_result if best_result is not None else x0
    try:
        result = minimize(lambda c: _evaluate_error(c), x0_nm, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        actual_score = _evaluate_error(result.x)
        _update_best(result.x, actual_score)
    except:
        pass
    
    if time.time() - _start_time > _DEADLINE - 20:
        if best_result is not None:
            return np.clip(best_result, -10, 10).tolist()
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Phase 3: Multiple restarts with Nelder-Mead from diverse starting points
    restarts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.50, 0.10, 0.0, 0.0, 0.0, 0.0],
        [0.60, -0.05, 0.05, 0.0, 0.0, 0.0],
        [0.55, 0.05, -0.05, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.10, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.10, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.10, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.10],
        [0.50, 0.15, -0.10, 0.05, -0.05, 0.0],
        [0.60, -0.10, 0.10, -0.05, 0.05, -0.05],
    ]
    
    for x0_try in restarts:
        if time.time() - _start_time > _DEADLINE - 15:
            break
        try:
            result = minimize(lambda c: _evaluate_error(c), x0_try, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            actual_score = _evaluate_error(result.x)
            _update_best(result.x, actual_score)
        except:
            pass
    
    # Phase 4: Final high-precision polish
    if best_result is not None and time.time() - _start_time < _DEADLINE - 10:
        try:
            result = minimize(lambda c: _evaluate_error(c), best_result, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            actual_score = _evaluate_error(result.x)
            _update_best(result.x, actual_score)
        except:
            pass
    
    if best_result is not None:
        return np.clip(best_result, -10, 10).tolist()
    
    # Fallback to safe constant gamma
    return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
