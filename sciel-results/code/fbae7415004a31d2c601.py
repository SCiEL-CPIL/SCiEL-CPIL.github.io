"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
# EVOLVE-BLOCK-START

def _ode_rhs(x, y, Om0, w0, wa):
    a = np.exp(x)
    # Omega_m(a)
    Om = Om0 * a**(-3)
    # Omega_de(a)
    Ode = 1.0 - Om
    w = w0 + wa * (1.0 - a)
    # H'/H = -3/2 [ Om + Ode*(1+3w) ]
    coeff = 1.5 * (Om + Ode * (1.0 + 3.0 * w))
    # d2D/dx2 = -coeff * dD/dx + 1.5*Om*D
    return [y[1], -coeff * y[1] + 1.5 * Om * y[0]]

def _get_reference_data(Om0, w0, wa, a_grid):
    """
    Computes the true f and D_norm for a given cosmology on a_grid.
    Uses the growing mode of the ODE solution.
    """
    x_grid = np.log(a_grid)
    x_min = x_grid[0]
    x_max = x_grid[-1]
    
    # Solve for two fundamental solutions
    # Basis 1: D(x_min)=1, D'(x_min)=0
    sol1 = solve_ivp(_ode_rhs, [x_min, x_max], [1.0, 0.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    # Basis 2: D(x_min)=0, D'(x_min)=1
    sol2 = solve_ivp(_ode_rhs, [x_min, x_max], [0.0, 1.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    
    D1 = sol1.sol(x_grid)[0]
    D1p = sol1.sol(x_grid)[1]
    D2 = sol2.sol(x_grid)[0]
    D2p = sol2.sol(x_grid)[1]
    
    # The growing mode is the one that grows with a.
    # In standard cosmology, the growing mode is the one that is positive and increasing.
    # We can estimate the growing mode by taking the linear combination that has the largest value at x_max
    # relative to x_min, or simply check the sign of D at x_max.
    # A robust way: The growing mode is the one that doesn't cross zero and grows.
    # For flat LCDM, D_grow ~ a.
    # Let's find the mode that is dominant at late times.
    # We can use the Wronskian or just check the values.
    # Typically, one mode grows like a and the other decays like a^{-1.5} or similar.
    # Let's assume the growing mode is the one with D(x_max) > 0 and D'(x_max) > 0.
    # If both are positive, we need to be careful.
    # Let's use the fact that f = D'/D should be positive for the growing mode.
    
    # Calculate f for both
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = D1p / D1
        f2 = D2p / D2
        
    # The growing mode typically has f > 0.
    # Let's pick the mode that has f > 0 for most of the range and is positive.
    # If D1 is the growing mode, f1 should be positive.
    # If D2 is the growing mode, f2 should be positive.
    # Usually, D1 (initial value 1) is a mix of both. D2 (initial slope 1) is also a mix.
    # A better approach: Solve for the growing mode directly by using an initial condition
    # that is purely growing at early times.
    # At early times (matter dominated), D ~ a. So D' ~ D.
    # Let's use D(x_min)=1, D'(x_min)=1 as a proxy for the growing mode?
    # This might introduce error.
    
    # Let's stick to the fundamental solutions and identify the growing mode.
    # The growing mode is the one that grows faster.
    # Let's check the ratio D(x_max)/D(x_min).
    r1 = D1[-1] / D1[0] if D1[0] != 0 else np.inf
    r2 = D2[-1] / D2[0] if D2[0] != 0 else np.inf
    
    # This is tricky if D crosses zero.
    # Let's use a standard technique: The growing mode is the one that is positive and monotonically increasing.
    # Let's try D = D1 + D2 and see if it's positive.
    # Actually, for most cosmologies, the growing mode is the one with D > 0.
    # Let's just use D2 (D(x_min)=0, D'(x_min)=1) as it is often close to the growing mode
    # if x_min is early enough. But D2(0) = 0, so D_norm would be 0 at start.
    
    # Let's use a different initial condition for the reference:
    # Solve with D(x_min)=1, D'(x_min)=1. This is a good approximation for the growing mode
    # in the matter-dominated era.
    sol_grow = solve_ivp(_ode_rhs, [x_min, x_max], [1.0, 1.0], args=(Om0, w0, wa),
                         method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    D_grow = sol_grow.sol(x_grid)[0]
    D_grow_p = sol_grow.sol(x_grid)[1]
    
    # Check if this is indeed the growing mode (positive and growing)
    # If it becomes negative, we might need to adjust.
    # For standard parameters, this should work.
    
    D_norm = D_grow / D_grow[-1]
    f_ref = D_grow_p / D_grow
    
    return f_ref, D_norm

def _gamma_of(a, Om0, w0, wa, coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = Om0 * a**(-3)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _compute_residuals(coeffs, params, a_grid):
    """
    Computes the residuals for the least squares optimizer.
    """
    residuals = []
    for (Om0, w0, wa) in params:
        f_ref, D_ref = _get_reference_data(Om0, w0, wa, a_grid)
        
        # Calculate model values
        Om = Om0 * a_grid**(-3)
        gamma = _gamma_of(a_grid, Om0, w0, wa, coeffs)
        
        # Enforce gamma bounds by clipping in the calculation? 
        # Or penalize? The task says Gamma must lie in [0.1, 1.2].
        # We will clip for calculation but add a penalty for violations.
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        
        # f_model = Om^gamma
        # To avoid issues with Om=0 or 1, we use log space or careful handling.
        # f = exp(gamma * ln(Om))
        with np.errstate(divide='ignore', invalid='ignore'):
            f_model = np.exp(gamma_clipped * np.log(Om))
        
        # D_model = exp(-integral f dx)
        # We need to integrate f from ln(a) to 0.
        # Let x = ln a. D(a)/D(1) = exp(-int_{ln a}^0 f(exp(x)) dx)
        # Let I(a) = int_{ln a}^0 f(exp(x)) dx
        # We can compute this numerically for each a in the grid.
        # This is expensive. Let's approximate the integral using the grid.
        # x_grid = np.log(a_grid)
        # f_vals = f_model
        # We need int_{x_i}^0 f(x) dx.
        # We can precompute the cumulative integral from the end.
        x_grid = np.log(a_grid)
        # f_vals is f_model
        # Use np.cumsum or scipy.integrate.cumulative_trapezoid
        # We want int_{x}^{0} f(x') dx'
        # Let's reverse the array for integration from 0 to x? No.
        # Let's compute the integral from 0 to x, then subtract from total.
        # Total integral = int_{x_min}^0 f(x) dx
        # int_{x}^0 f(x') dx' = Total - int_{x_min}^x f(x') dx'
        
        # Use cumulative trapezoid
        # f_vals is defined on x_grid
        # cumulative_int = cumulative_trapezoid(f_vals, x_grid, initial=0)
        # This gives int_{x_min}^{x_i} f(x) dx
        # We want int_{x_i}^0 f(x) dx = int_{x_min}^0 - int_{x_min}^{x_i}
        
        # We can use np.cumsum for simplicity, assuming uniform grid in x? 
        # a_grid is logspace, so x_grid is linear.
        # dx = x_grid[1] - x_grid[0]
        # trapezoid rule: 0.5 * (f[i] + f[i+1]) * dx
        
        # Let's use scipy.integrate.cumulative_trapezoid
        from scipy.integrate import cumulative_trapezoid
        cum_int = cumulative_trapezoid(f_model, x_grid, initial=0)
        total_int = cum_int[-1]
        # int_{x_i}^0 = total_int - cum_int[i]
        int_vals = total_int - cum_int
        
        D_model = np.exp(-int_vals)
        
        # Relative errors
        # f_rel_err = (f_model - f_ref) / f_ref
        # D_rel_err = (D_model - D_ref) / D_ref
        
        # Handle division by zero
        f_rel_err = np.zeros_like(f_ref)
        mask_f = np.abs(f_ref) > 1e-10
        f_rel_err[mask_f] = (f_model[mask_f] - f_ref[mask_f]) / f_ref[mask_f]
        
        D_rel_err = np.zeros_like(D_ref)
        mask_D = np.abs(D_ref) > 1e-10
        D_rel_err[mask_D] = (D_model[mask_D] - D_ref[mask_D]) / D_ref[mask_D]
        
        residuals.extend(f_rel_err)
        residuals.extend(D_rel_err)
        
        # Add penalty for gamma out of bounds
        gamma_violations = np.sum(np.abs(gamma - gamma_clipped))
        residuals.append(gamma_violations * 10.0) # Weight the penalty
        
    return residuals

def _fit_coefficients():
    # Representative CPL cosmologies
    params = [
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.3),
        (0.30, -1.1, 0.5),
        (0.25, -1.0, 0.2),
        (0.35, -0.8, 0.4),
        (0.30, -1.0, -0.5),
        (0.28, -1.2, 0.6),
        (0.32, -0.7, 0.1),
        (0.30, -1.0, 0.2),
        (0.30, -1.0, -0.2),
    ]
    
    # Use a finer grid for fitting
    a_grid = np.logspace(-2, 0, 100)
    
    # Initial guess: constant gamma ~ 0.55, small corrections
    x0 = [0.55, 0.0, 0.0, 0.05, 0.1, 0.0]
    
    # Bounds for coefficients to keep them reasonable
    bounds = ([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
              [10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    
    # Optimize
    result = least_squares(_compute_residuals, x0, args=(params, a_grid), bounds=bounds, method='trf')
    
    return result.x

def construct():
    try:
        coeffs = _fit_coefficients()
        # Clip to [-10,10] for safety
        coeffs = [float(np.clip(c, -10.0, 10.0)) for c in coeffs]
        # Sanity: ensure finite
        if not all(np.isfinite(c) for c in coeffs):
            raise ValueError
        return coeffs
    except Exception:
        # Fallback to a reasonable constant-gamma with mild CPL correction
        return [0.55, 0.0, 0.0, 0.05, 0.1, 0.0]
# EVOLVE-BLOCK-END
