"""Improved growth-index approximation with optimized coefficients."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

# EVOLVE-BLOCK-START
def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    Omega_DE0 = 1.0 - Om0
    ratio_DE = (a ** (-3*(1+w0+wa))) * np.exp(3*wa*(a-1))
    Om_a = Om0 * a**(-3) / (Om0 * a**(-3) + Omega_DE0 * ratio_DE)
    return Om_a

def _compute_gamma(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    # Clamp to valid range
    gamma = np.clip(gamma, 0.1, 1.2)
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    Om_a = _Om_of_a(a, Om0, w0, wa)
    gamma = _compute_gamma(a, coeffs, Om0, w0, wa)
    return Om_a ** gamma

def _D_normalized(a, coeffs, Om0, w0, wa):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    if a >= 1.0:
        return 1.0
    ln_a = np.log(a)
    # Integrate f(exp(x)) from ln_a to 0
    def integrand(x):
        a_x = np.exp(x)
        return _f_of_a(a_x, coeffs, Om0, w0, wa)
    integral_val, _ = quad(integrand, ln_a, 0.0)
    return np.exp(-integral_val)

def _solve_growth_ODE_fast(a_evals, Om0, w0, wa, n_steps=2000):
    """
    Solve the growth ODE numerically for a set of a_evals.
    Returns arrays of D_ref and f_ref corresponding to a_evals.
    """
    # Define a grid for integration
    a_grid = np.linspace(0.05, 1.0, n_steps)
    
    # Pre-compute Om(a) and q(a) on the grid
    Om_grid = np.array([_Om_of_a(ai, Om0, w0, wa) for ai in a_grid])
    H2_grid = Om_grid * a_grid**(-3)
    
    # Compute q(a) = 0.5 * d ln H2 / d ln a
    # Using central differences
    dlnH2_dlna = np.zeros_like(a_grid)
    # log(H2) = log(Om) - 3 log(a)
    # d/dlna = a * d/da
    # Let's just use numerical derivative of H2
    dH2_da = np.gradient(H2_grid, a_grid)
    dlnH2_dlna = a_grid * dH2_da / H2_grid
    q_grid = 0.5 * dlnH2_dlna
    
    # ODE: D'' + (3/a - q(a))D' - (3/2)Om(a)D = 0
    # Let y1 = D, y2 = D'
    # y1' = y2
    # y2' = -(3/a - q(a))y2 + (3/2)Om(a)y1
    
    # Initial conditions at a=0.05 (early matter domination)
    # D ~ a, so D(0.05) = 0.05, D'(0.05) = 1.0
    y = np.zeros((2, n_steps))
    y[0, 0] = a_grid[0]
    y[1, 0] = 1.0
    
    dt = a_grid[1] - a_grid[0]
    
    for i in range(n_steps - 1):
        ai = a_grid[i]
        qi = q_grid[i]
        Omi = Om_grid[i]
        
        # RHS function
        def rhs(y_val):
            D_val = y_val[0]
            Dp_val = y_val[1]
            Dpp = -(3.0/ai - qi) * Dp_val + (1.5 * Omi) * D_val
            return np.array([Dp_val, Dpp])
        
        # RK4 Step
        k1 = rhs(y[:, i])
        k2 = rhs(y[:, i] + 0.5 * dt * k1)
        k3 = rhs(y[:, i] + 0.5 * dt * k2)
        k4 = rhs(y[:, i] + dt * k3)
        
        y[:, i+1] = y[:, i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    # Normalize D such that D(1) = 1
    D_grid = y[0] / y[0, -1]
    Dp_grid = y[1] / y[0, -1]
    f_grid = Dp_grid / D_grid
    
    # Interpolate to a_evals
    # Use linear interpolation for simplicity and speed
    D_evals = np.interp(a_evals, a_grid, D_grid)
    f_evals = np.interp(a_evals, a_grid, f_grid)
    
    return D_evals, f_evals

def construct():
    """Find optimal six coefficients by minimizing max error over test cosmologies."""
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline for safety
    
    # Test cosmologies (w0, wa, Om0)
    # Include a diverse set to ensure robustness
    test_models = [
        (-1.0, 0.0, 0.3),      # LCDM
        (-1.0, 0.1, 0.3),      # CPL with wa=0.1
        (-0.9, 0.0, 0.3),      # CPL with w0=-0.9
        (-1.1, 0.0, 0.3),      # CPL with w0=-1.1
        (-1.0, 0.3, 0.3),      # CPL with wa=0.3
        (-0.8, -0.2, 0.3),     # Mixed
        (-1.2, 0.5, 0.3),      # Strong CPL
        (-1.0, 0.0, 0.27),     # LCDM with different Om
        (-0.95, 0.1, 0.28),    # Near-LCDM CPL
        (-1.05, -0.1, 0.32),   # Another near-LCDM
        (-1.0, -0.5, 0.3),     # Phantom-ish
        (-0.7, 0.0, 0.3),      # Matter-like DE
    ]
    
    # a values to check
    # Use a denser grid for better minimax representation
    a_values = np.linspace(0.1, 1.0, 50)
    
    # Pre-compute reference data
    ref_data = {}
    for (w0, wa, Om0) in test_models:
        try:
            D_ref, f_ref = _solve_growth_ODE_fast(a_values, Om0, w0, wa)
            ref_data[(w0, wa, Om0)] = (D_ref, f_ref)
        except Exception:
            # If a model fails, skip it or use a fallback
            pass
            
    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    def objective(coeffs):
        """Max relative error over all test models and a values."""
        max_err = 0.0
        for (w0, wa, Om0) in test_models:
            if (w0, wa, Om0) not in ref_data:
                continue
            D_ref, f_ref = ref_data[(w0, wa, Om0)]
            
            # Compute approximations vectorized if possible, but loops are fine for 50 points
            # To speed up, we can vectorize the gamma calculation
            a_arr = a_values
            Om_arr = np.array([_Om_of_a(ai, Om0, w0, wa) for ai in a_arr])
            q_arr = 1.0 - Om_arr
            u_arr = 1.0 + w0 + wa * (1.0 - a_arr)
            
            c0, c1, c2, c3, c4, c5 = coeffs
            gamma_arr = c0 + c1*q_arr + c2*q_arr**2 + c3*u_arr + c4*wa*a_arr + c5*u_arr*q_arr
            gamma_arr = np.clip(gamma_arr, 0.1, 1.2)
            
            f_approx_arr = Om_arr ** gamma_arr
            
            # Calculate D_approx
            # D(a) = exp(-int_a^1 f(a')/a' da') ... wait, the task says exp(-integral_ln(a)^0 f(exp(x)) dx)
            # Let x = ln(a'), dx = da'/a'.
            # Integral from ln(a) to 0 of f(exp(x)) dx
            # This is equivalent to integral from a to 1 of f(a')/a' da'
            
            # We can compute this integral numerically for each a in the array
            # Since a_values is sorted, we can use cumulative trapezoidal integration from the end
            # f_over_a = f_approx_arr / a_arr
            # integral_i = sum_{j>i} 0.5 * (f_j/a_j + f_i/a_i) * (a_j - a_i) ... 
            # Actually, let's just use quad for each point to be safe, or a simple trapezoidal rule on the grid.
            # Trapezoidal rule is fast and accurate enough for optimization.
            
            f_over_a = f_approx_arr / a_arr
            # We want integral from a_i to 1.
            # Let's compute cumulative integral from 1 down to a_min.
            # da = a[1:] - a[:-1]
            # avg_f = 0.5 * (f_over_a[1:] + f_over_a[:-1])
            # step_int = avg_f * da
            # Cumulative sum from the end.
            
            da = np.diff(a_arr)
            avg_f = 0.5 * (f_over_a[1:] + f_over_a[:-1])
            step_int = avg_f * da
            
            # cumulative integral from a_i to a_{i+1} is step_int[i]
            # Integral from a_i to 1 is sum(step_int[i:] )
            # We can use cumsum from the end
            cum_int = np.zeros_like(a_arr)
            cum_int[-2] = step_int[-1]
            for i in range(len(step_int)-2, -1, -1):
                cum_int[i] = cum_int[i+1] + step_int[i]
                
            D_approx_arr = np.exp(-cum_int)
            
            # Relative errors
            err_f = np.abs(f_approx_arr - f_ref) / np.abs(f_ref)
            err_D = np.abs(D_approx_arr - D_ref) / np.abs(D_ref)
            
            # Handle potential zeros in reference (unlikely for f, D is normalized to 1 at a=1)
            err_f = np.nan_to_num(err_f, nan=0.0, posinf=1.0)
            err_D = np.nan_to_num(err_D, nan=0.0, posinf=1.0)
            
            max_err = max(max_err, np.max(err_f), np.max(err_D))
        
        return max_err

    # Start from constant gamma
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0.copy()
    best_err = objective(x0)
    
    # Use Nelder-Mead with the fast objective
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if result.fun < best_err:
            best_coeffs = result.x
            best_err = result.fun
        
        # Check deadline
        if time.time() > deadline:
            return best_coeffs.tolist()
        
        # Polish with a tighter tolerance
        if time.time() < deadline - 10.0:
            result2 = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True}
            )
            if result2.fun < best_err:
                best_coeffs = result2.x
                best_err = result2.fun

    except Exception:
        pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    
    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
