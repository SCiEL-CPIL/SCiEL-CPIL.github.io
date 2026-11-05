"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def _compute_Om(a, Om0, w0, wa):
    """Compute Om(a) for CPL dark energy."""
    if a <= 0:
        return 0.0
    a_inv3 = a ** (-3)
    a_inv3_cpl = a ** (-3 * (1 + w0 + wa))
    denom = Om0 * a_inv3 + (1 - Om0) * a_inv3_cpl
    if denom == 0:
        return 0.0
    return Om0 * a_inv3 / denom

def _compute_Hprime_over_H(a, Om0, w0, wa):
    """Compute H'/H = (1/2) d ln(H^2/H0^2) / d ln a."""
    if a <= 0:
        return 0.0
    a_inv3 = a ** (-3)
    a_inv3_cpl = a ** (-3 * (1 + w0 + wa))
    A = Om0 * a_inv3
    B = (1 - Om0) * a_inv3_cpl
    denom = A + B
    if denom == 0:
        return 0.0
    dlnH2_dlna = -3 * (A + (1 + w0 + wa) * B) / denom
    return 0.5 * dlnH2_dlna

def _solve_reference(Om0, w0, wa, a_grid, rtol=1e-8, atol=1e-10):
    """
    Solve the growth ODE for a given cosmology and return f_ref and D_ref on a_grid.
    Uses backward integration from early times to find the growing mode initial condition at a=1.
    """
    # Filter valid a values
    valid_mask = (a_grid > 0) & (a_grid < 1)
    if not np.any(valid_mask):
        return None
    
    a_valid = a_grid[valid_mask]
    x_valid = np.log(a_valid)
    
    # Start from a small value where matter domination holds
    a_start = min(a_valid) * 0.5
    if a_start <= 0:
        a_start = 0.01
    x_start = np.log(a_start)
    
    # At a_start, D ~ a_start, D' ~ 1 (in terms of x = ln a)
    D_start = a_start
    Dprime_start = 1.0
    
    def growth_ode(x, y):
        a_x = np.exp(x)
        if a_x <= 0:
            return [0, 0]
        Om_x = _compute_Om(a_x, Om0, w0, wa)
        Hprime_over_H = _compute_Hprime_over_H(a_x, Om0, w0, wa)
        
        D, Dprime = y
        if abs(x) < 1e-10:
            Ddoubleprime = -(Hprime_over_H) * Dprime + (3/2) * Om_x * D
        else:
            Ddoubleprime = -(3/x + Hprime_over_H) * Dprime + (3/2) * Om_x * D
        return [Dprime, Ddoubleprime]
    
    try:
        # Integrate from x_start to 0
        sol = solve_ivp(growth_ode, [x_start, 0], [D_start, Dprime_start], 
                       method='RK45', rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            return None
        
        # Get D and D' at x=0 to normalize
        D_at_0 = sol.y[0, -1]
        Dprime_at_0 = sol.y[1, -1]
        
        if abs(D_at_0) < 1e-10:
            return None
        
        # Normalize so D(0) = 1
        Dprime0 = Dprime_at_0 / D_at_0
        
        # Now integrate from 0 to min(x_valid)
        x_end = min(x_valid)
        if x_end >= 0:
            # All a values are >= 1, which shouldn't happen given our filter
            return None
        
        sol2 = solve_ivp(growth_ode, [0, x_end], [1.0, Dprime0], 
                        method='RK45', rtol=rtol, atol=atol, dense_output=True)
        if not sol2.success:
            return None
        
        # Evaluate at x_valid points
        D_ref = np.zeros_like(a_grid)
        Dprime_ref = np.zeros_like(a_grid)
        f_ref = np.zeros_like(a_grid)
        
        for i, x in enumerate(x_valid):
            D_val = sol2.sol(x)[0]
            Dprime_val = sol2.sol(x)[1]
            
            idx = np.where(valid_mask)[0][i]
            D_ref[idx] = D_val
            Dprime_ref[idx] = Dprime_val
            if abs(D_val) > 1e-10:
                f_ref[idx] = -Dprime_val / D_val
            else:
                f_ref[idx] = 0.0
        
        return f_ref, D_ref
    except:
        return None

def _compute_error(coeffs, cosmologies, a_grid, deadline, reference_cache=None):
    """Compute max relative error for given coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    if reference_cache is None:
        reference_cache = {}
    
    max_err = 0.0
    for cosmo in cosmologies:
        Om0, w0, wa = cosmo
        cosmo_key = (Om0, w0, wa)
        
        if cosmo_key not in reference_cache:
            ref = _solve_reference(Om0, w0, wa, a_grid)
            if ref is None:
                continue
            reference_cache[cosmo_key] = ref
        
        f_ref, D_ref = reference_cache[cosmo_key]
        
        for i, a in enumerate(a_grid):
            if a <= 0 or a >= 1:
                continue
            
            Om_a = _compute_Om(a, Om0, w0, wa)
            if Om_a <= 0:
                return float('inf')
            
            q = 1 - Om_a
            u = 1 + w0 + wa * (1 - a)
            
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            
            if gamma < 0.1 or gamma > 1.2:
                return float('inf')
            
            f_approx = Om_a ** gamma
            
            # Compute D_ratio_approx = exp(-integral of f_approx from ln(a) to 0)
            # Use numerical integration
            x0 = np.log(a)
            
            def integrand(x):
                a_x = np.exp(x)
                Om_x = _compute_Om(a_x, Om0, w0, wa)
                if Om_x <= 0:
                    return 0.0
                q_x = 1 - Om_x
                u_x = 1 + w0 + wa * (1 - a_x)
                gamma_x = c0 + c1 * q_x + c2 * q_x**2 + c3 * u_x + c4 * wa * a_x + c5 * u_x * q_x
                if gamma_x < 0.1 or gamma_x > 1.2:
                    return 0.0
                return Om_x ** gamma_x
            
            try:
                result = solve_ivp(lambda x, y: [-integrand(x)], [x0, 0], [0.0], 
                                  method='RK45', rtol=1e-8, atol=1e-10)
                if not result.success:
                    return float('inf')
                integral_val = -result.y[0, -1]
                D_ratio = np.exp(-integral_val)
            except:
                return float('inf')
            
            # Compute relative errors
            D_ratio_ref = D_ref[i]
            f_ratio_ref = f_ref[i]
            
            if abs(f_ratio_ref) > 1e-10:
                err_f = abs(f_approx - f_ratio_ref) / abs(f_ratio_ref)
            else:
                err_f = abs(f_approx - f_ratio_ref)
            
            if abs(D_ratio_ref) > 1e-10:
                err_D = abs(D_ratio - D_ratio_ref) / abs(D_ratio_ref)
            else:
                err_D = abs(D_ratio - D_ratio_ref)
            
            max_err = max(max_err, err_f, err_D)
            
            if time.time() > deadline:
                return max_err
    
    return max_err

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds, leaving 10s safety margin
    
    # Validation cosmologies (typical values)
    cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -1.0, 0.3),    # CPL with wa=0.3
        (0.3, -1.0, -0.5),   # CPL with wa=-0.5
        (0.25, -1.0, 0.0),   # Lower Om0
        (0.35, -1.0, 0.0),   # Higher Om0
        (0.3, -0.9, 0.1),    # Different w0
        (0.3, -1.1, -0.2),   # Different w0, wa
    ]
    
    # a grid
    a_grid = np.linspace(0.1, 0.99, 20)
    
    # Physically-motivated initial guess: c0=0.55, c3=-0.5 to cancel w0 term
    best_coeffs = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    
    # Cache for reference solutions
    reference_cache = {}
    
    best_err = _compute_error(best_coeffs, cosmologies, a_grid, deadline, reference_cache)
    
    # Optimization
    def objective(coeffs):
        if time.time() > deadline:
            return best_err
        err = _compute_error(coeffs, cosmologies, a_grid, deadline, reference_cache)
        return err
    
    try:
        result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        if result.fun < best_err:
            best_coeffs = result.x
            best_err = result.fun
    except:
        pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
