"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp
import time

def _compute_om_a(a, Om0, w0, wa):
    """Compute Omega_m(a) for CPL dark energy."""
    Omd0 = 1.0 - Om0
    if a <= 0:
        return 1.0
    a3 = a ** 3
    de_term = Omd0 * np.exp(3.0 * (1.0 + w0) * a) if (1.0 + w0) != 0 else Omd0
    total = Om0 / a3 + de_term
    return Om0 / a3 / total

def _compute_gamma(c, a, Om0, w0, wa):
    """Compute gamma from the six-coefficient formula with sqrt(q) term."""
    Om_a = _compute_om_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    sqrt_q = np.sqrt(max(q, 0.0))
    gamma = c[0] + c[1]*q + c[2]*sqrt_q + c[3]*u + c[4]*wa*a + c[5]*u*q
    return gamma

def _compute_f(c, a, Om0, w0, wa):
    """Compute f = Om(a)^gamma."""
    gamma = _compute_gamma(c, a, Om0, w0, wa)
    Om_a = _compute_om_a(a, Om0, w0, wa)
    return Om_a ** gamma

def _compute_D_normalized(c, a, Om0, w0, wa, n_points=500):
    """Compute D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)."""
    if a >= 1.0:
        return 1.0
    x_vals = np.linspace(np.log(a), 0.0, n_points)
    a_vals = np.exp(x_vals)
    f_vals = np.array([_compute_f(c, ai, Om0, w0, wa) for ai in a_vals])
    integral = np.trapz(f_vals, x_vals)
    return np.exp(-integral)

def _reference_f_ode(a, Om0, w0, wa):
    """High-accuracy reference f using Riccati ODE integration from a=1."""
    if a >= 1.0:
        return _compute_om_a(a, Om0, w0, wa) ** 0.55
    
    def riccati(x, y):
        ai = np.exp(x)
        Om_ai = _compute_om_a(ai, Om0, w0, wa)
        Omd_ai = 1.0 - Om_ai
        w_ai = w0 + wa * (1.0 - ai)
        Hprime_over_H = 1.5 * Om_ai - 1.5 * (1.0 + w_ai) * Omd_ai
        f_val = y[0]
        D_val = y[1]
        f_prime = 1.5 * Om_ai - f_val**2 - (3.0 + Hprime_over_H) * f_val
        D_prime = f_val * D_val
        return [f_prime, D_prime]
    
    x0 = 0.0
    xf = np.log(a)
    y0 = [0.55, 1.0]
    
    try:
        sol = solve_ivp(riccati, [x0, xf], y0, rtol=1e-10, atol=1e-12, 
                       max_step=0.01, dense_output=False)
        if sol.success and len(sol.y[0]) > 0:
            return sol.y[0, -1]
    except:
        pass
    
    n_steps = 2000
    dx = xf / n_steps
    f_val = 0.55
    D_val = 1.0
    for i in range(n_steps):
        x_i = x0 + i * dx
        ai = np.exp(x_i)
        Om_ai = _compute_om_a(ai, Om0, w0, wa)
        Omd_ai = 1.0 - Om_ai
        w_ai = w0 + wa * (1.0 - ai)
        Hprime_over_H = 1.5 * Om_ai - 1.5 * (1.0 + w_ai) * Omd_ai
        f_prime = 1.5 * Om_ai - f_val**2 - (3.0 + Hprime_over_H) * f_val
        f_val += f_prime * dx
        D_val *= np.exp(f_val * dx)
    return f_val

def _reference_D_ode(a, Om0, w0, wa):
    """High-accuracy reference D(a)/D(1) using ODE integration."""
    if a >= 1.0:
        return 1.0
    
    def riccati(x, y):
        ai = np.exp(x)
        Om_ai = _compute_om_a(ai, Om0, w0, wa)
        Omd_ai = 1.0 - Om_ai
        w_ai = w0 + wa * (1.0 - ai)
        Hprime_over_H = 1.5 * Om_ai - 1.5 * (1.0 + w_ai) * Omd_ai
        f_val = y[0]
        D_val = y[1]
        f_prime = 1.5 * Om_ai - f_val**2 - (3.0 + Hprime_over_H) * f_val
        D_prime = f_val * D_val
        return [f_prime, D_prime]
    
    x0 = 0.0
    xf = np.log(a)
    y0 = [0.55, 1.0]
    
    try:
        sol = solve_ivp(riccati, [x0, xf], y0, rtol=1e-10, atol=1e-12, 
                       max_step=0.01, dense_output=False)
        if sol.success and len(sol.y[1]) > 0:
            return sol.y[1, -1]
    except:
        pass
    
    n_steps = 2000
    dx = xf / n_steps
    f_val = 0.55
    D_val = 1.0
    for i in range(n_steps):
        x_i = x0 + i * dx
        ai = np.exp(x_i)
        Om_ai = _compute_om_a(ai, Om0, w0, wa)
        Omd_ai = 1.0 - Om_ai
        w_ai = w0 + wa * (1.0 - ai)
        Hprime_over_H = 1.5 * Om_ai - 1.5 * (1.0 + w_ai) * Omd_ai
        f_prime = 1.5 * Om_ai - f_val**2 - (3.0 + Hprime_over_H) * f_val
        f_val += f_prime * dx
        D_val *= np.exp(f_val * dx)
    return D_val

def _evaluate_score(c, models, a_values, use_smooth=False, smooth_beta=50.0):
    """Compute the score: 100 * max relative error over both f and D."""
    max_err = 0.0
    all_errs = []
    
    for (Om0, w0, wa) in models:
        for a in a_values:
            if a <= 0 or a > 1.0:
                continue
            f_approx = _compute_f(c, a, Om0, w0, wa)
            f_ref = _reference_f_ode(a, Om0, w0, wa)
            
            D_approx = _compute_D_normalized(c, a, Om0, w0, wa)
            D_ref = _reference_D_ode(a, Om0, w0, wa)
            
            if abs(f_ref) > 1e-10:
                err_f = abs(f_approx - f_ref) / abs(f_ref)
            else:
                err_f = abs(f_approx - f_ref)
            
            if abs(D_ref) > 1e-10:
                err_D = abs(D_approx - D_ref) / abs(D_ref)
            else:
                err_D = abs(D_approx - D_ref)
            
            all_errs.append(err_f)
            all_errs.append(err_D)
    
    if not all_errs:
        return 1e6
    
    if use_smooth:
        beta = smooth_beta
        scaled = np.array(all_errs) * beta
        max_scaled = np.max(scaled)
        if max_scaled > 700:
            smooth_max = max_scaled / beta
        else:
            smooth_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
        return 100.0 * smooth_max
    else:
        return 100.0 * np.max(all_errs)

def _gamma_constraint_check(c, models, a_values):
    """Check that gamma is in [0.1, 1.2] at all locations."""
    for (Om0, w0, wa) in models:
        for a in a_values:
            if a <= 0 or a > 1.0:
                continue
            gamma = _compute_gamma(c, a, Om0, w0, wa)
            if gamma < 0.1 or gamma > 1.2:
                return False
    return True

def _penalized_score(c, models, a_values, use_smooth=False, smooth_beta=50.0):
    """Score with penalty for gamma constraint violation."""
    base_score = _evaluate_score(c, models, a_values, use_smooth, smooth_beta)
    if not _gamma_constraint_check(c, models, a_values):
        return base_score + 1000.0
    return base_score

def _optimize_coeffs():
    """Optimize the six coefficients using DE + NM polish."""
    start_time = time.time()
    deadline = start_time + 75.0
    
    models = [
        (0.3, 0.0, 0.0),
        (0.3, -0.5, 0.5),
        (0.3, -1.2, 0.8),
        (0.25, -0.9, 0.3),
        (0.35, -0.7, -0.2),
        (0.28, -1.0, 0.6),
    ]
    
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    bounds = [(-10, 10)] * 6
    
    best_c = None
    best_score = 1e6
    
    try:
        de_result = differential_evolution(
            _penalized_score,
            bounds,
            args=(models, a_values, False, 50.0),
            maxiter=40,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol'
        )
        elapsed = time.time() - start_time
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_c = de_result.x
        
        if time.time() < deadline - 15.0:
            nm_result = minimize(
                _penalized_score,
                best_c,
                args=(models, a_values, True, 50.0),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_c = nm_result.x
        
        if time.time() < deadline - 5.0:
            polish_result = minimize(
                _penalized_score,
                best_c,
                args=(models, a_values, False, 50.0),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_c = polish_result.x
        
    except Exception as e:
        print(f"Optimization error: {e}")
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_c = np.clip(best_c, -10.0, 10.0)
    
    return list(best_c)

# Pre-compute optimized coefficients at import time
_OPTIMIZED_COEFFS = _optimize_coeffs()

def construct():
    return _OPTIMIZED_COEFFS
# EVOLVE-BLOCK-END
