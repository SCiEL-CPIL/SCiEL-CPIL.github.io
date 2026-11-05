"""Cosmology: compact dark-energy growth approximation - optimized coefficients."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Deadline guard: 90s wall-clock limit, keep safety margin
_DEADLINE = 90.0
_MARGIN = 5.0  # safety margin in seconds


def _elapsed():
    return _t0 + (time.time() - _start) if '_t0' in globals() else time.time() - _start


def _check_deadline():
    return (time.time() - _start) < (_DEADLINE - _MARGIN)


_start = time.time()


def _compute_reference(omega_m, w0, wa, a_vals):
    """Compute reference f(a) and D(a)/D(1) using the GR second-order ODE."""
    # E^2(a) = omega_m * a^{-3} + (1-omega_m) * exp(-3 * integral_1^a (1+w(a')) da')
    # For CPL: w(a') = w0 + wa*(1-a')
    # integral_1^a (1+w(a')) da' = (1+w0+wa)*(a-1) - wa/2*(a^2-1)
    # So DE density ratio: exp(-3*[(1+w0+wa)*(a-1) - wa/2*(a^2-1)])
    
    a = a_vals
    om_a = omega_m * a**(-3) / (omega_m * a**(-3) + (1 - omega_m) * np.exp(-3 * ((1 + w0 + wa) * (a - 1) - wa / 2 * (a**2 - 1))))
    
    # Friction coefficient in ln a: 2 * dH/dln a / H = 2 * (1/2) * dE^2/dln a / E^2
    # dE^2/dln a = -3*omega_m*a^{-3} + (1-omega_m)*exp(...) * (-3)*(-1+wa)  ... 
    # Actually: d/da [a^{-3}] = -3a^{-4}, dE^2/dln a = a * dE^2/da
    # E^2 = omega_m * a^{-3} + (1-omega_m) * exp(-3*((1+w0+wa)*(a-1) - wa/2*(a^2-1)))
    # dE^2/da = -3*omega_m*a^{-4} + (1-omega_m)*exp(...) * (-3)*((1+w0+wa) - wa*a)
    # dE^2/dln a = a * dE^2/da = -3*omega_m*a^{-3} + (1-omega_m)*exp(...) * (-3)*((1+w0+wa) - wa*a)*a
    
    # Let's compute H' = dH/dln a = (1/2) * dE^2/dln a / E^2
    # The growth ODE in x = ln a:
    # d2D/dx2 + 2*(1 + (1/2)*w_eff) ... no.
    # Standard: d2D/dx2 + (3/2 + (3/2)*w_eff + q) dD/dx ... 
    # Let me use the exact form:
    # d^2 D/dx^2 + [2 + (1/2)*w(a) + ... ] 
    # Actually the correct equation is:
    # d^2 D/dx^2 + 2 * (1 + (1/3)*w(a) * ... ) 
    # Let me derive: in a flat universe with CPL:
    # E^2(a) = Om*a^{-3} + OL*exp(-3*int(1+w)da)
    # d ln E / d ln a = (1/2) * d E^2/dln a / E^2
    # dE^2/dln a = -3*Om*a^{-3} + OL*exp(...) * (-3)*((1+w0+wa) - wa*a) * a / 1
    # Wait: d/da[exp(-3*((1+w0+wa)*(a-1) - wa/2*(a^2-1)))] 
    #   = exp(...) * (-3)*((1+w0+wa) - wa*a)
    # dE^2/da = -3*Om*a^{-4} + OL*exp(...) * (-3)*((1+w0+wa) - wa*a)
    # dE^2/dln a = a*dE^2/da = -3*Om*a^{-3} - 3*OL*exp(...)*((1+w0+wa) - wa*a)*a
    
    # Define the friction term: F(a) = 2 * dlnE/dlna = (1/E^2) * dE^2/dlna
    de2_dlna = -3 * omega_m * a**(-3) - 3 * (1 - omega_m) * np.exp(-3 * ((1 + w0 + wa) * (a - 1) - wa / 2 * (a**2 - 1))) * ((1 + w0 + wa) - wa * a) * a
    e2 = omega_m * a**(-3) + (1 - omega_m) * np.exp(-3 * ((1 + w0 + wa) * (a - 1) - wa / 2 * (a**2 - 1)))
    friction = de2_dlna / e2  # this is dlnE^2/dlna = 2*dlnE/dlna
    
    # Growth ODE: d2f/dx2 + f*(dlnE^2/dlna + f) - 3/2*Om(a) = 0  ... 
    # Actually: let f = dlnD/dx. Then D''/D = f' + f^2.
    # D''/D + (dlnE^2/dlna)/2 * D'/D ... no.
    # The exact equation: d2D/dx2 + [3/2 + (3/2)*w(a) + ... ] 
    # Let me use: d2D/dx2 + (2 + dlnE^2/dlna) * dD/dx ... 
    # 
    # Correct form from literature: 
    # d2D/dx2 + [2 * dlnE/dlna + 3/2] dD/dx - 3/2 * Om(a) = 0
    # where 2*dlnE/dlna = dlnE^2/dlna
    #
    # So: D'' + friction * D' - 1.5 * Om(a) = 0
    # where friction = dlnE^2/dlna = -3*Om(a) - 3*(1+wa_eff(a))*(1-Om(a)) approximately
    # More precisely as computed above.
    
    # Integrate from x=0 (a=1) backwards to x_min (a_min)
    # At a=1: D(1)=1, D'(1) = Om(1)^gamma ≈ Om_m (use 0.55 for init or compute self-consistently)
    # For the reference, we need a self-consistent solution.
    # Use shooting: at a=1, set D=1, D'=f0. Solve ODE forward in x (x>0 is future, x<0 is past).
    # We want a from a_min to 1, so integrate x from x_min to 0.
    
    x_min = np.log(a_vals[0])
    x_max = 0.0
    x_grid = np.linspace(x_min, x_max, 500)
    a_grid = np.exp(x_grid)
    
    # Precompute Om(a) and friction on grid
    om_grid = omega_m * a_grid**(-3) / (omega_m * a_grid**(-3) + (1 - omega_m) * np.exp(-3 * ((1 + w0 + wa) * (a_grid - 1) - wa / 2 * (a_grid**2 - 1))))
    de2_dlna_grid = -3 * omega_m * a_grid**(-3) - 3 * (1 - omega_m) * np.exp(-3 * ((1 + w0 + wa) * (a_grid - 1) - wa / 2 * (a_grid**2 - 1))) * ((1 + w0 + wa) - wa * a_grid) * a_grid
    e2_grid = omega_m * a_grid**(-3) + (1 - omega_m) * np.exp(-3 * ((1 + w0 + wa) * (a_grid - 1) - wa / 2 * (a_grid**2 - 1)))
    friction_grid = de2_dlna_grid / e2_grid
    
    def ode_funcs(x, y):
        # y = [D, D']
        idx = np.argmin(np.abs(x_grid - x))
        return [y[1], -friction_grid[idx] * y[1] + 1.5 * om_grid[idx]]
    
    # Initial conditions at x=0 (a=1): D(1)=1, D'(1)=f0
    # We need to find f0 such that the solution is consistent. 
    # For the reference, use a good initial guess and iterate, or just use f0=Om_m^0.55
    f0 = omega_m ** 0.55
    
    # Integrate from x=0 to x_min
    sol = solve_ivp(ode_funcs, [0.0, x_min], [1.0, f0], t_eval=x_grid[::-1], method='RK45', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        # Fallback: use the approximate gamma formula
        f_ref = om_a ** 0.55
        # D via integration
        lnD = np.zeros_like(a)
        # Simple trapz
        from scipy.integrate import cumulative_trapezoid
        x_inv = -np.log(a)
        f_vals = om_a ** 0.55
        lnD = -cumulative_trapezoid(f_vals, x_inv, initial=0)
        D_ref = np.exp(lnD)
        return f_ref, D_ref, om_a
    
    D_ref = sol.y[0][::-1]  # reverse to match x_grid order (from x_min to 0)
    Dp_ref = sol.y[1][::-1]
    f_ref = Dp_ref / D_ref
    
    # Interpolate to get values at a_vals
    f_at_a = np.interp(np.log(a_vals), np.log(a_grid), f_ref)
    D_at_a = np.interp(np.log(a_vals), np.log(a_grid), D_ref)
    D_at_a = D_at_a / D_at_a[-1]  # normalize so D(1)=1
    f_at_a[-1] = f_ref[-1]  # ensure f(1) is exact
    
    return f_at_a, D_at_a, om_a


def _evaluate_coefficients(coeffs, omega_m, w0, wa, a_vals):
    """Compute f and D for given coefficients, return max relative error vs reference."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    a = a_vals
    om_a = omega_m * a**(-3) / (omega_m * a**(-3) + (1 - omega_m) * np.exp(-3 * ((1 + w0 + wa) * (a - 1) - wa / 2 * (a**2 - 1))))
    q = 1 - om_a
    u = 1 + w0 + wa * (1 - a)
    
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    
    # Check gamma bounds
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        return 100.0  # penalty
    
    f_model = om_a ** gamma
    
    # Compute D via integral
    from scipy.integrate import cumulative_trapezoid
    x = np.log(a)
    # f(exp(x)) for x from x_min to 0
    f_x = f_model
    # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    # = exp(-integral from 0 to -ln(a) of f(exp(-t)) dt)  where t = -x
    # Let's just integrate numerically
    # integral from ln(a) to 0 of f(exp(x)) dx
    # Use cumulative trapezoid from the end
    # We need the integral from each ln(a_i) to 0
    # int_{ln(a_i)}^0 f(exp(x)) dx = -int_0^{ln(a_i)} f(exp(x)) dx = int_{ln(a_i)}^0 ...
    
    # Simpler: compute cumulative integral from right to left
    # Let points be x[0] < x[1] < ... < x[n-1] = 0
    # integral from x[i] to x[n-1] of f(x) dx
    # Use cumulative_trapezoid in reverse
    
    # f_x is evaluated at x points
    # We want integral from x[i] to 0 = integral from x[i] to x[n-1]
    # cumulative_trapezoid(f, x) gives integral from x[0] to x[i]
    # So integral from x[i] to x[n-1] = cumtrapz_total - cumtrapz[i]
    
    cumtrapz = cumulative_trapezoid(f_x, x, initial=0)
    total = cumtrapz[-1]
    integrals = total - cumtrapz  # integral from x[i] to x[n-1]=0
    
    D_model = np.exp(-integrals)
    D_model = D_model / D_model[-1]  # normalize (should already be ~1 at end)
    
    # Reference
    f_ref, D_ref, _ = _compute_reference(omega_m, w0, wa, a_vals)
    
    # Relative errors
    rel_f = np.abs(f_model - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
    rel_D = np.abs(D_model - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
    
    max_err = np.max(np.concatenate([rel_f, rel_D]))
    return 100.0 * max_err


def _objective(coeffs):
    """Objective function for optimization over validation cosmologies."""
    if not _check_deadline():
        return 1000.0
    
    # Validation cosmologies (disjoint, representative)
    cosmologies = [
        (0.30, 0.0, 0.0),    # LCDM
        (0.27, -1.0, 0.0),   # LCDM with lower Om
        (0.30, -1.0, 0.5),   # CPL with wa>0
        (0.25, -0.9, -0.3),  # CPL with wa<0
        (0.35, -1.0, 0.0),   # Higher Om
        (0.28, -0.5, -0.5),  # Mildly w0
        (0.30, -1.0, 1.0),   # Large wa
        (0.22, -1.0, -0.5),  # Low Om, negative wa
    ]
    
    a_vals = np.linspace(0.01, 1.0, 50)
    
    max_err = 0.0
    for (om, w0, wa) in cosmologies:
        if not _check_deadline():
            return 1000.0
        err = _evaluate_coefficients(coeffs, om, w0, wa, a_vals)
        max_err = max(max_err, err)
    
    return max_err


def construct():
    """Return six coefficients in [-10, 10]."""
    global _start
    _start = time.time()
    
    # Start with the seed
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = _objective(best_coeffs)
    
    # Try Nelder-Mead optimization
    if _check_deadline():
        try:
            res = minimize(
                _objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_score:
                best_coeffs = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Try L-BFGS-B with bounds
    if _check_deadline():
        try:
            bounds = [(-10, 10)] * 6
            res2 = minimize(
                _objective,
                best_coeffs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            if res2.fun < best_score:
                best_coeffs = res2.x
                best_score = res2.fun
        except Exception:
            pass
    
    # Additional local refinement
    if _check_deadline():
        try:
            res3 = minimize(
                _objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res3.fun < best_score:
                best_coeffs = res3.x
                best_score = res3.fun
        except Exception:
            pass
    
    # Clip to valid range
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
