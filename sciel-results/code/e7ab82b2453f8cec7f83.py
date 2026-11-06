"""Improved six-coefficient growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START
def construct():
    """Return six finite coefficients in [-10,10] for the growth-index ansatz.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a), Om(a) is matter density fraction.
    f = Om(a)^gamma, D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    """
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    def Om_a(a, Om0, w0, wa):
        """Matter density fraction as function of scale factor."""
        a = np.asarray(a, dtype=float)
        scalar_input = a.ndim == 0
        a = np.atleast_1d(a)
        result = np.ones_like(a)
        valid = a > 0
        if np.any(valid):
            av = a[valid]
            de_factor = (1-Om0) * np.exp(-3*w0*np.log(av) - 3*wa*(av-1))
            result[valid] = Om0 / (Om0 + de_factor)
        if scalar_input:
            return float(result[0])
        return result
    
    def gr_reference(a_arr, Om0, w0, wa):
        """Compute reference f and D using the GR ODE."""
        a_arr = np.atleast_1d(a_arr)
        
        def ode_rhs_D(ln_a, state):
            a_local = np.exp(ln_a)
            Om_local = Om_a(a_local, Om0, w0, wa)
            E2 = Om0 * a_local**(-3) + (1-Om0) * np.exp(-3*w0*np.log(a_local) - 3*wa*(a_local-1))
            E2_prime = -3*Om0*a_local**(-4) + (1-Om0)*np.exp(-3*w0*np.log(a_local) - 3*wa*(a_local-1)) * (-3*w0/a_local - 3*wa)
            dlnE2_dlna = a_local * E2_prime / E2
            coeff = 3.0 + 0.5 * dlnE2_dlna
            return [state[1], -coeff*state[1] + 1.5*Om_local*state[0]]
        
        a_min = 0.05
        n_pts = 200
        ln_a_vals = np.linspace(0.0, np.log(a_min), n_pts)
        
        try:
            sol = solve_ivp(ode_rhs_D, [0.0, np.log(a_min)], [1.0, 0.55], 
                           t_eval=ln_a_vals, rtol=1e-10, atol=1e-12, method='DOP853')
            
            if not sol.success:
                Om_ref = Om_a(a_arr, Om0, w0, wa)
                f_ref = Om_ref**0.55
                D_ref = np.ones_like(a_arr)
                return f_ref, D_ref
            
            # Interpolate
            D_interp = np.interp(np.log(a_arr), sol.t, sol.y[0])
            Dp_interp = np.interp(np.log(a_arr), sol.t, sol.y[1])
            f_ref = Dp_interp
            D_ref = D_interp
            return f_ref, D_ref
            
        except Exception:
            Om_ref = Om_a(a_arr, Om0, w0, wa)
            f_ref = Om_ref**0.55
            D_ref = np.ones_like(a_arr)
            return f_ref, D_ref
    
    def compute_f_D(a_arr, c, Om0, w0, wa):
        """Compute f and normalized D for a range of a values (vectorized)."""
        a_arr = np.asarray(a_arr, dtype=float)
        Om = Om_a(a_arr, Om0, w0, wa)
        q = 1.0 - Om
        u = 1.0 + w0 + wa*(1.0 - a_arr)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_arr + c[5]*u*q
        gamma = np.clip(gamma, 0.1, 1.2)
        f = Om**gamma
        
        # Compute D(a)/D(1) = exp(-∫_{ln(a)}^0 f(e^x) dx)
        # Use cumulative integration on a grid
        D_norm = np.ones_like(a_arr)
        
        # Create a fine grid for integration
        a_grid_min = 0.01
        a_grid_max = 1.0
        n_grid = 500
        a_grid = np.linspace(a_grid_min, a_grid_max, n_grid)
        ln_grid = np.log(a_grid)
        
        # Compute f on the grid
        Om_grid = Om_a(a_grid, Om0, w0, wa)
        q_grid = 1.0 - Om_grid
        u_grid = 1.0 + w0 + wa*(1.0 - a_grid)
        gamma_grid = c[0] + c[1]*q_grid + c[2]*q_grid**2 + c[3]*u_grid + c[4]*wa*a_grid + c[5]*u_grid*q_grid
        gamma_grid = np.clip(gamma_grid, 0.1, 1.2)
        f_grid = Om_grid**gamma_grid
        
        # Cumulative integral from ln(a) to 0: ∫_{ln(a)}^0 f(e^x) dx
        # = -∫_0^{ln(a)} f(e^x) dx for a < 1
        # We compute cumulative integral from left to right on ln_grid
        # cumulative[i] = ∫_{ln_grid[0]}^{ln_grid[i]} f(e^x) dx
        # Then ∫_{ln(a)}^0 f dx = ∫_0^{ln(a)} f dx for a>1, or -∫_{ln(a)}^0 for a<1
        # Actually: ∫_{ln(a)}^0 = -∫_0^{ln(a)}
        # For a < 1: ln(a) < 0, ∫_0^{ln(a)} is negative, so ∫_{ln(a)}^0 is positive
        # For a > 1: ln(a) > 0, ∫_0^{ln(a)} is positive, so ∫_{ln(a)}^0 is negative
        
        # Compute cumulative integral from ln_grid[0] to each point
        dx = np.diff(ln_grid)
        avg_f = 0.5 * (f_grid[:-1] + f_grid[1:])
        cum_int = np.zeros(n_grid)
        cum_int[1:] = np.cumsum(avg_f * dx)
        
        # Find index of a=1 (ln=0)
        idx_1 = np.searchsorted(ln_grid, 0.0)
        if idx_1 < n_grid:
            # Interpolate to get exact value at ln=0
            if idx_1 > 0:
                frac = (0.0 - ln_grid[idx_1-1]) / (ln_grid[idx_1] - ln_grid[idx_1-1])
                cum_at_1 = cum_int[idx_1-1] + frac * (cum_int[idx_1] - cum_int[idx_1-1])
            else:
                cum_at_1 = 0.0
        else:
            cum_at_1 = cum_int[-1]
        
        # For each a in a_arr, compute ∫_{ln(a)}^0 f dx
        for i, ai in enumerate(a_arr):
            if ai >= 1.0:
                D_norm[i] = 1.0
            else:
                ln_ai = np.log(ai)
                idx = np.searchsorted(ln_grid, ln_ai)
                if idx == 0:
                    cum_ai = 0.0
                elif idx >= n_grid:
                    cum_ai = cum_int[-1]
                else:
                    frac = (ln_ai - ln_grid[idx-1]) / (ln_grid[idx] - ln_grid[idx-1])
                    cum_ai = cum_int[idx-1] + frac * (cum_int[idx] - cum_int[idx-1])
                # ∫_{ln(a)}^0 f dx = cum_at_1 - cum_ai (since both are from ln_grid[0])
                integral = cum_at_1 - cum_ai
                D_norm[i] = np.exp(-integral)
        
        return f, D_norm
    
    # Define test cosmologies for fitting - broader range
    test_models = [
        (0.20, -1.0, 0.0),
        (0.25, -1.0, 0.0),
        (0.30, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.40, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.30, -0.8, 0.3),
        (0.30, -0.8, -0.3),
        (0.25, -1.2, 0.3),
        (0.35, -0.8, 0.2),
        (0.30, -1.0, 1.0),
        (0.20, -1.5, 0.5),
        (0.40, -0.9, -0.3),
        (0.30, -1.0, 0.8),
        (0.30, -1.0, -0.8),
        (0.28, -1.1, 0.2),
        (0.32, -0.9, -0.4),
    ]
    
    a_test = np.linspace(0.05, 0.95, 20)
    
    # Pre-compute references
    references = {}
    for Om0, w0, wa in test_models:
        if time.time() > deadline:
            break
        f_ref, D_ref = gr_reference(a_test, Om0, w0, wa)
        references[(Om0, w0, wa)] = (f_ref, D_ref)
    
    if not references:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def objective(c):
        if time.time() > deadline:
            return 1e10 * np.ones(len(a_test) * len(test_models) * 2)
        
        residuals = []
        for Om0, w0, wa in test_models:
            if (Om0, w0, wa) not in references:
                continue
            f_ref, D_ref = references[(Om0, w0, wa)]
            f_approx, D_approx = compute_f_D(a_test, c, Om0, w0, wa)
            
            # Relative errors
            f_rel = (f_approx - f_ref) / np.abs(f_ref)
            D_rel = (D_approx - D_ref) / np.abs(D_ref)
            residuals.extend(f_rel)
            residuals.extend(D_rel)
        
        return np.array(residuals)
    
    # Initial guess: constant gamma
    c0_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_c = c0_init.copy()
    best_cost = np.inf
    
    try:
        from scipy.optimize import least_squares
        result = least_squares(objective, c0_init, bounds=([-10]*6, [10]*6), 
                               method='trf', max_nfev=300, xtol=1e-10, ftol=1e-10)
        if np.all(np.isfinite(result.x)):
            best_c = np.clip(result.x, -10, 10)
            best_cost = result.cost
    except Exception:
        pass
    
    # Verify gamma constraint across a wider range
    def check_gamma(c):
        for Om0, w0, wa in test_models:
            for a in np.linspace(0.01, 1.0, 50):
                Om = Om_a(a, Om0, w0, wa)
                q = 1.0 - Om
                u = 1.0 + w0 + wa*(1.0 - a)
                g = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
                if g < 0.1 or g > 1.2:
                    return False
        return True
    
    if not check_gamma(best_c):
        # Try to adjust by reducing coefficients
        for scale in [0.5, 0.25, 0.1]:
            adjusted = best_c * scale
            if check_gamma(adjusted):
                best_c = adjusted
                break
        else:
            best_c = c0_init.copy()
    
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
