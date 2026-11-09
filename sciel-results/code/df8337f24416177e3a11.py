"""Improved growth-index formula with optimized coefficients."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

_START_TIME = time.time()
_TIME_LIMIT = 80.0  # Safety margin before 90s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _Om(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    a = np.asarray(a, dtype=float)
    # DE density parameter
    w_eff = w0 + wa * (1 - a)
    # DE relative to matter at a=1
    # Om_DE(a) = Om_DE(0) * a^(-3(1+w_eff)) / (1 + wa/3 * (1-a^3))
    # Simplified: use integral form
    # For CPL: Om_DE(a) = Om_DE(0) * a^(-3(1+w0+wa)) * (1 + wa/3*(1-a^3))^(-1)
    # Actually the standard CPL form:
    # rho_DE(a) = rho_DE(0) * a^(-3(1+w0+wa)) * (1 + wa/3 * (1-a^3))^(-1)
    # Let's use a more direct approach
    
    Om0_de = 1.0 - Om0
    if abs(wa) < 1e-10:
        # w = w0 constant
        factor = a ** (-3.0 * (1.0 + w0))
    else:
        # CPL with wa
        # rho_DE(a)/rho_DE(0) = a^(-3(1+w0+wa)) * (1 + wa/3*(1-a^3))^(-1)
        # But this is approximate. Let's use the standard form:
        # E^2(a) = Om_m * a^(-3) + Om_DE * a^(-3(1+w0+wa)) * (1 + wa/3*(1-a^3))^(-1)
        # Actually, the exact CPL form is:
        # E^2(a) = Om_m*a^(-3) + Om_DE * exp(-3*wa*(1-a)) * a^(-3*(1+w0))
        # No, let me use the standard definition properly.
        
        # For CPL w(a) = w0 + wa*(1-a):
        # rho_DE(a) = rho_DE(0) * a^(-3(1+w0+wa)) * (1 + wa/3*(1-a^3))^(-1)
        # This is the standard form used in cosmology.
        
        if a <= 0:
            factor = np.inf
        else:
            a3 = a ** 3
            denom = 1.0 + (wa / 3.0) * (1.0 - a3)
            if abs(denom) < 1e-15:
                factor = np.sign(1.0 + wa/3.0) * 1e15
            else:
                factor = a ** (-3.0 * (1.0 + w0 + wa)) / denom
    
    if np.isscalar(a) or a.ndim == 0:
        a = np.array([a])
    
    Om_m = Om0 * a ** (-3.0)
    Om_de = Om0_de * factor if not np.isscalar(factor) else Om0_de * factor
    
    # Handle array vs scalar
    if np.isscalar(Om_de):
        Om_de = np.full_like(Om_m, Om_de)
    else:
        Om_de = np.asarray(Om_de)
    
    total = Om_m + Om_de
    return Om_m / total

def _compute_f_and_D(a_vals, coeffs, Om0, w0, wa):
    """Compute f and D(a)/D(1) for given coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a_vals = np.asarray(a_vals, dtype=float)
    
    Om_a = _Om(a_vals, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a_vals)
    
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
    
    # Clamp gamma to valid range
    gamma = np.clip(gamma, 0.1, 1.2)
    
    f = Om_a ** gamma
    
    # Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    D_ratio = np.ones_like(a_vals)
    for i, a in enumerate(a_vals):
        if a >= 1.0:
            D_ratio[i] = 1.0
            continue
        if a <= 0:
            D_ratio[i] = 0.0
            continue
        
        def integrand(x):
            a_x = np.exp(x)
            Om_x = _Om(a_x, Om0, w0, wa)
            q_x = 1.0 - Om_x
            u_x = 1.0 + w0 + wa * (1.0 - a_x)
            gamma_x = c0 + c1 * q_x + c2 * q_x**2 + c3 * u_x + c4 * wa * a_x + c5 * u_x * q_x
            gamma_x = np.clip(gamma_x, 0.1, 1.2)
            return Om_x ** gamma_x
        
        try:
            integral, _ = quad(integrand, np.log(a), 0.0, limit=50)
            D_ratio[i] = np.exp(-integral)
        except:
            D_ratio[i] = 0.0
    
    return f, D_ratio

def _true_solution(a_vals, Om0, w0, wa):
    """Compute true f and D using the ODE solution."""
    a_vals = np.asarray(a_vals, dtype=float)
    
    # For the true solution, we need to solve the growth equation
    # f = d ln D / d ln a = Om(a)^gamma_true
    # where gamma_true comes from the exact solution
    
    # Actually, the "true" f is defined by the ODE:
    # f' + (2+3H'/H)f + f^2 = 0.5 * Om(a)
    # with boundary condition f(1) = 1
    
    # Let's numerically integrate the ODE
    from scipy.integrate import solve_ivp
    
    def ode_rhs(ln_a, y):
        a = np.exp(ln_a)
        Om_a = _Om(a, Om0, w0, wa)
        # H'/H = (1/2)(3*Om_m - 2*Om_DE*(1+w_DE))
        # For CPL: w_DE = w0 + wa*(1-a)
        Om_m = Om_a
        Om_de = 1.0 - Om_a
        w_de = w0 + wa * (1.0 - a)
        Hprime_over_H = 0.5 * (3.0 * Om_m - 2.0 * Om_de * (1.0 + w_de))
        
        f = y[0]
        df_dlna = -f**2 - (2.0 + 3.0 * Hprime_over_H) * f + 0.5 * Om_a
        return [df_dlna]
    
    f_true = np.zeros_like(a_vals)
    D_ratio = np.zeros_like(a_vals)
    
    for i, a in enumerate(a_vals):
        if a >= 1.0:
            f_true[i] = 1.0
            D_ratio[i] = 1.0
            continue
        if a <= 0:
            f_true[i] = 0.0
            D_ratio[i] = 0.0
            continue
        
        try:
            # Integrate from ln(1)=0 to ln(a)
            sol = solve_ivp(ode_rhs, [0.0, np.log(a)], [1.0], 
                           method='RK45', rtol=1e-10, atol=1e-12,
                           dense_output=True)
            if sol.success:
                f_true[i] = sol.y[0, -1]
                # D(a)/D(1) = exp(integral from ln(a) to 0 of f(exp(x)) dx)
                # We can get this from the dense output
                x_eval = np.linspace(np.log(a), 0.0, 100)
                y_eval = sol.sol(x_eval)
                integral = np.trapz(y_eval[0], x_eval)
                D_ratio[i] = np.exp(integral)
            else:
                f_true[i] = 0.0
                D_ratio[i] = 0.0
        except:
            f_true[i] = 0.0
            D_ratio[i] = 0.0
    
    return f_true, D_ratio

def _objective(coeffs, cosmologies, a_grid):
    """Compute worst relative error across all cosmologies."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Check bounds
    if any(abs(c) > 10.0 for c in coeffs):
        return 1e6
    
    max_err = 0.0
    
    for Om0, w0, wa in cosmologies:
        f_approx, D_approx = _compute_f_and_D(a_grid, coeffs, Om0, w0, wa)
        f_true, D_true = _true_solution(a_grid, Om0, w0, wa)
        
        # Filter valid values
        valid = (f_true > 1e-6) & (D_true > 1e-6) & (D_approx > 1e-6)
        
        if np.sum(valid) < 10:
            continue
        
        err_f = np.abs(f_approx[valid] - f_true[valid]) / np.abs(f_true[valid])
        err_D = np.abs(D_approx[valid] - D_true[valid]) / np.abs(D_true[valid])
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
    
    return 100.0 * max_err

def construct():
    """Return optimized 6 coefficients."""
    if _time_left() < 10.0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Representative cosmologies for optimization
    cosmologies = [
        (0.3, -1.0, 0.0),    # LambdaCDM
        (0.3, -1.0, 0.5),    # CPL with wa=0.5
        (0.3, -1.0, -0.5),   # CPL with wa=-0.5
        (0.25, -1.0, 0.0),   # Lower Om0
        (0.35, -1.0, 0.0),   # Higher Om0
        (0.3, -0.9, 0.0),    # w0 = -0.9
        (0.3, -1.1, 0.0),    # w0 = -1.1
        (0.3, -0.9, 0.3),    # w0=-0.9, wa=0.3
        (0.3, -1.0, 0.8),    # Strong wa
        (0.3, -1.2, -0.3),   # Phantom-ish
    ]
    
    # Scale factor grid
    a_grid = np.linspace(0.05, 1.0, 20)
    
    # Initial guess: constant gamma = 0.55
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    bounds = [(-10, 10)] * 6
    
    best_x = x0
    best_score = _objective(x0, cosmologies, a_grid)
    
    # Try Nelder-Mead first
    if _time_left() > 30.0:
        try:
            res = minimize(_objective, x0, args=(cosmologies, a_grid),
                          method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success and res.fun < best_score:
                best_x = res.x
                best_score = res.fun
        except:
            pass
    
    # Try COBYLA if time allows
    if _time_left() > 15.0:
        try:
            res = minimize(_objective, best_x, args=(cosmologies, a_grid),
                          method='COBYLA',
                          options={'maxiter': 100, 'rhobeg': 0.1})
            if res.fun < best_score:
                best_x = res.x
                best_score = res.fun
        except:
            pass
    
    # Ensure bounds
    best_x = np.clip(best_x, -10.0, 10.0)
    
    return list(best_x)
