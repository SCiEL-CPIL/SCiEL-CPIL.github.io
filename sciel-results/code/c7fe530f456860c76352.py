import numpy as np
import time
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

_start_time = time.time()
_DEADLINE = 75.0  # 75s out of 90s, leaving 15s margin

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def _H2_of_a(a, Om_m, w0, wa):
    """Compute H²(a)/H0² for CPL dark energy."""
    Om_de = 1.0 - Om_m
    # Dark energy density: rho_de ∝ a^(-3(1+w0+wa*(1-a)))
    w = w0 + wa * (1.0 - a)
    rho_de = Om_de * np.power(a, -3.0 * (1.0 + w))
    rho_m = Om_m / a**3
    rho_k = 0.0  # assume flat universe
    return rho_m + rho_de + rho_k

def _Om_m_of_a(a, Om_m, w0, wa):
    """Compute Om_m(a) = rho_m / rho_total."""
    H2 = _H2_of_a(a, Om_m, w0, wa)
    rho_m = Om_m / a**3
    return rho_m / H2

def _growth_ode(x, y, Om_m, w0, wa):
    """
    ODE for growth factor in terms of x = ln(a).
    y = [D, D']
    D'' + (3 + H'/H) D' - (3/2) Om_m(a) D = 0
    """
    a = np.exp(x)
    D, Dp = y
    H2 = _H2_of_a(a, Om_m, w0, wa)
    # d ln H / dx = (1/2) d ln H2 / dx
    # d ln H2 / dx = (1/H2) * dH2/dx
    # H2 = Om_m*a^-3 + Om_de*a^(-3(1+w0+wa*(1-a)))
    # dH2/dx = -3*Om_m*a^-3 + Om_de * (-3*(1+w0+wa*(1-a)) + 3*wa) * a^(-3(1+w0+wa*(1-a)))
    #        = -3*Om_m*a^-3 + Om_de * (-3*(1+w0) + 3*wa*a) * a^(-3(1+w0+wa*(1-a)))
    Om_de = 1.0 - Om_m
    w = w0 + wa * (1.0 - a)
    rho_m = Om_m / a**3
    rho_de = Om_de * np.power(a, -3.0 * (1.0 + w))
    drho_m_dx = -3.0 * rho_m
    drho_de_dx = rho_de * (-3.0 * (1.0 + w) + 3.0 * wa * a)
    dH2_dx = drho_m_dx + drho_de_dx
    dlnH_dx = 0.5 * dH2_dx / H2
    Om_ma = _Om_m_of_a(a, Om_m, w0, wa)
    Dpp = -(3.0 + dlnH_dx) * Dp + 1.5 * Om_ma * D
    return [Dp, Dpp]

def _solve_growth(a_grid, Om_m, w0, wa):
    """
    Solve the growth ODE on a_grid (in a-space) for given cosmology.
    Returns f = D'/D and D/D(1) on a_grid.
    """
    # Work in x = ln(a)
    x_grid = np.log(a_grid)
    x0 = np.log(1.0)  # a=1
    
    # Integrate from a=1 backwards to min(a_grid)
    # We need D and D' at a=1 as initial conditions.
    # Use the known asymptotic: for small z, D ≈ a, D' ≈ 1 at a=1 (EdS-like normalization)
    # Actually, let's integrate forward from a small value to a=1, then normalize.
    
    a_min = a_grid.min()
    a_max = 1.0
    
    # Integrate from a_min to a_max
    x_min = np.log(a_min)
    x_max = np.log(a_max)
    
    # Initial conditions at x_min: D ~ a^(3/2) for matter-dominated era (approx)
    D0 = np.exp(1.5 * x_min)  # D ∝ a^(3/2)
    Dp0 = 1.5 * np.exp(1.5 * x_min)  # D' = dD/dx = (3/2) a^(3/2) * (1/a) * a = (3/2) a^(3/2)
    # Wait, D' = dD/dx where x = ln a, so D' = a * dD/da
    # If D = a^(3/2), then dD/da = (3/2)a^(1/2), so D' = a * (3/2)a^(1/2) = (3/2)a^(3/2)
    # Yes, Dp0 = 1.5 * D0
    
    sol = solve_ivp(
        _growth_ode, [x_min, x_max], [D0, Dp0],
        args=(Om_m, w0, wa),
        t_eval=x_grid,
        rtol=1e-10, atol=1e-12,
        method='DOP853'
    )
    
    if not sol.success:
        return None, None
    
    D_vals = sol.y[0]
    Dp_vals = sol.y[1]
    
    # f = D'/D
    f_vals = Dp_vals / D_vals
    
    # D(a)/D(1): normalize so D(1) = 1
    # Find index where a is closest to 1
    D_at_1 = D_vals[-1]  # since a_max = 1
    D_norm = D_vals / D_at_1
    
    return f_vals, D_norm

def _compute_error(coeffs, a_grid, cosmologies):
    """Compute max relative error of f and D for given coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for Om_m, w0, wa in cosmologies:
        # Compute reference values
        f_ref, D_ref = _solve_growth(a_grid, Om_m, w0, wa)
        if f_ref is None:
            continue
        
        # Compute our approximation
        a = a_grid
        Om_ma = _Om_m_of_a(a, Om_m, w0, wa)
        q = 1.0 - Om_ma
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 100.0  # penalty
        
        f_approx = np.power(Om_ma, gamma)
        
        # Relative error in f
        err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
        
        # For D: our D(a)/D(1) = exp(-∫_a^1 f(exp(x)) dx)
        # We need to integrate f_approx from a to 1
        # f(exp(x)) where x = ln(a'), so we integrate over x from ln(a) to 0
        D_approx = np.zeros_like(a)
        for i, ai in enumerate(a):
            if ai >= 1.0:
                D_approx[i] = 1.0
            else:
                # Integrate f(exp(x)) from x=ln(ai) to x=0
                x_vals = np.linspace(np.log(ai), 0.0, 50)
                a_vals = np.exp(x_vals)
                Om_ma_vals = _Om_m_of_a(a_vals, Om_m, w0, wa)
                q_vals = 1.0 - Om_ma_vals
                u_vals = 1.0 + w0 + wa * (1.0 - a_vals)
                gamma_vals = c0 + c1*q_vals + c2*q_vals**2 + c3*u_vals + c4*wa*a_vals + c5*u_vals*q_vals
                f_vals = np.power(Om_ma_vals, gamma_vals)
                integral = np.trapz(f_vals, x_vals)
                D_approx[i] = np.exp(-integral)
        
        err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
    
    return 100.0 * max_err

def construct():
    global _start_time
    _start_time = time.time()
    
    # Default fallback
    default_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        # Representative cosmologies for fitting
        # Om_m in [0.2, 0.4], w0 in [-1.2, -0.8], wa in [-0.5, 0.5]
        cosmologies = [
            (0.3, -1.0, 0.0),    # Lambda CDM
            (0.3, -1.0, 0.3),    # CPL with positive wa
            (0.3, -1.0, -0.3),   # CPL with negative wa
            (0.25, -1.1, 0.0),   # Low Om_m, more negative w0
            (0.35, -0.9, 0.0),   # High Om_m, less negative w0
            (0.3, -1.2, 0.5),    # Very negative w0, positive wa
            (0.3, -0.8, -0.5),   # Less negative w0, negative wa
            (0.2, -1.0, 0.0),    # Low matter
            (0.4, -1.0, 0.0),    # High matter
        ]
        
        # Scale factor grid for fitting
        a_grid = np.logspace(np.log10(0.1), 0.0, 20)  # a from 0.1 to 1
        
        # Compute reference data
        ref_data = []
        for Om_m, w0, wa in cosmologies:
            f_ref, D_ref = _solve_growth(a_grid, Om_m, w0, wa)
            if f_ref is not None:
                ref_data.append((a_grid, f_ref, D_ref, Om_m, w0, wa))
        
        if not ref_data:
            return default_coeffs
        
        # Objective function for least_squares
        def objective(coeffs):
            if _check_deadline():
                return [100.0] * len(ref_data)
            c0, c1, c2, c3, c4, c5 = coeffs
            
            # Check bounds
            if not all(0.1 <= g <= 1.2 for g in [c0, c1, c2, c3, c4, c5]):
                return [100.0] * len(ref_data)
            
            residuals = []
            for a_grid, f_ref, D_ref, Om_m, w0, wa in ref_data:
                a = a_grid
                Om_ma = _Om_m_of_a(a, Om_m, w0, wa)
                q = 1.0 - Om_ma
                u = 1.0 + w0 + wa * (1.0 - a)
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                # Check gamma bounds
                if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                    residuals.extend([100.0] * len(a))
                    continue
                
                f_approx = np.power(Om_ma, gamma)
                
                # f residuals
                residuals.extend(np.abs(f_approx - f_ref) / np.abs(f_ref))
                
                # D residuals - approximate integration
                for i, ai in enumerate(a):
                    if ai >= 1.0:
                        residuals.append(abs(1.0 - D_ref[i]) / abs(D_ref[i]))
                    else:
                        x_vals = np.linspace(np.log(ai), 0.0, 30)
                        a_vals = np.exp(x_vals)
                        Om_ma_vals = _Om_m_of_a(a_vals, Om_m, w0, wa)
                        q_vals = 1.0 - Om_ma_vals
                        u_vals = 1.0 + w0 + wa * (1.0 - a_vals)
                        gamma_vals = c0 + c1*q_vals + c2*q_vals**2 + c3*u_vals + c4*wa*a_vals + c5*u_vals*q_vals
                        f_vals = np.power(Om_ma_vals, gamma_vals)
                        integral = np.trapz(f_vals, x_vals)
                        D_approx = np.exp(-integral)
                        residuals.append(abs(D_approx - D_ref[i]) / abs(D_ref[i]))
            
            return residuals
        
        # Bounds for coefficients
        bounds = ([-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10])
        
        # Initial guess
        x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Run optimization
        result = least_squares(
            objective, x0, bounds=bounds,
            method='trf', max_nfev=200, xtol=1e-8, ftol=1e-8
        )
        
        best_coeffs = result.x
        
        # Verify gamma bounds on a dense grid
        for Om_m, w0, wa in cosmologies:
            a_test = np.linspace(0.1, 1.0, 100)
            Om_ma = _Om_m_of_a(a_test, Om_m, w0, wa)
            q = 1.0 - Om_ma
            u = 1.0 + w0 + wa * (1.0 - a_test)
            gamma = best_coeffs[0] + best_coeffs[1]*q + best_coeffs[2]*q**2 + best_coeffs[3]*u + best_coeffs[4]*wa*a_test + best_coeffs[5]*u*q
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return default_coeffs
        
        # All coefficients must be in [-10, 10]
        if not all(-10 <= c <= 10 for c in best_coeffs):
            return default_coeffs
        
        return [float(c) for c in best_coeffs]
    
    except Exception:
        return default_coeffs

# Ensure construct is callable
def construct():
    global _start_time
    _start_time = time.time()
    
    default_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        # Representative cosmologies
        cosmologies = [
            (0.3, -1.0, 0.0),
            (0.3, -1.0, 0.3),
            (0.3, -1.0, -0.3),
            (0.25, -1.1, 0.0),
            (0.35, -0.9, 0.0),
            (0.3, -1.2, 0.5),
            (0.3, -0.8, -0.5),
            (0.2, -1.0, 0.0),
            (0.4, -1.0, 0.0),
        ]
        
        a_grid = np.logspace(np.log10(0.1), 0.0, 15)
        
        ref_data = []
        for Om_m, w0, wa in cosmologies:
            f_ref, D_ref = _solve_growth(a_grid, Om_m, w0, wa)
            if f_ref is not None:
                ref_data.append((a_grid, f_ref, D_ref, Om_m, w0, wa))
        
        if not ref_data:
            return default_coeffs
        
        def objective(coeffs):
            if _check_deadline():
                return [100.0] * (len(ref_data) * len(a_grid) * 2)
            c0, c1, c2, c3, c4, c5 = coeffs
            
            residuals = []
            for a_g, f_ref, D_ref, Om_m, w0, wa in ref_data:
                a = a_g
                Om_ma = _Om_m_of_a(a, Om_m, w0, wa)
                q = 1.0 - Om_ma
                u = 1.0 + w0 + wa * (1.0 - a)
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                    residuals.extend([100.0] * len(a) * 2)
                    continue
                
                f_approx = np.power(Om_ma, gamma)
                residuals.extend(np.abs(f_approx - f_ref) / np.abs(f_ref))
                
                for i, ai in enumerate(a):
                    if ai >= 0.999:
                        residuals.append(abs(1.0 - D_ref[i]) / max(abs(D_ref[i]), 1e-10))
                    else:
                        n_pts = 25
                        x_vals = np.linspace(np.log(ai), 0.0, n_pts)
                        a_vals = np.exp(x_vals)
                        Om_ma_vals = _Om_m_of_a(a_vals, Om_m, w0, wa)
                        q_vals = 1.0 - Om_ma_vals
                        u_vals = 1.0 + w0 + wa * (1.0 - a_vals)
                        gamma_vals = c0 + c1*q_vals + c2*q_vals**2 + c3*u_vals + c4*wa*a_vals + c5*u_vals*q_vals
                        f_vals = np.power(Om_ma_vals, gamma_vals)
                        integral = np.trapz(f_vals, x_vals)
                        D_approx = np.exp(-integral)
                        residuals.append(abs(D_approx - D_ref[i]) / max(abs(D_ref[i]), 1e-10))
            
            return residuals
        
        bounds = ([-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10])
        x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        result = least_squares(
            objective, x0, bounds=bounds,
            method='trf', max_nfev=150, xtol=1e-8, ftol=1e-8
        )
        
        best_coeffs = result.x
        
        # Verify gamma bounds
        for Om_m, w0, wa in cosmologies:
            a_test = np.linspace(0.1, 1.0, 50)
            Om_ma = _Om_m_of_a(a_test, Om_m, w0, wa)
            q = 1.0 - Om_ma
            u = 1.0 + w0 + wa * (1.0 - a_test)
            gamma = best_coeffs[0] + best_coeffs[1]*q + best_coeffs[2]*q**2 + best_coeffs[3]*u + best_coeffs[4]*wa*a_test + best_coeffs[5]*u*q
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return default_coeffs
        
        if not all(-10 <= c <= 10 for c in best_coeffs):
            return default_coeffs
        
        return [float(c) for c in best_coeffs]
    
    except Exception:
        return default_coeffs
