import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# Constants for the optimization process
START_TIME = time.time()
TIME_LIMIT = 80.0  # Leave 10s margin for 90s limit

def _E2(a, Om0, w0, wa):
    """Calculate E^2(a) for CPL cosmology."""
    # Om_m(a) = Om0 * a^-3
    # Om_L(a) = Om_L0 * a^(-3(1+w0+wa)) * exp(3*wa*(a-1))
    # E2 = Om_m(a) + Om_L(a)
    # Note: This assumes flat universe Om_L0 = 1 - Om0
    OmL0 = 1.0 - Om0
    term_m = Om0 * a**(-3)
    # Exponent part for Lambda
    exp_part = np.exp(3.0 * wa * (a - 1.0))
    power_part = a**(-3.0 * (1.0 + w0 + wa))
    term_l = OmL0 * power_part * exp_part
    return term_m + term_l

def _E2_prime(a, Om0, w0, wa):
    """Calculate d(E^2)/da for CPL cosmology."""
    OmL0 = 1.0 - Om0
    # d/da (Om0 * a^-3) = -3 * Om0 * a^-4
    d_term_m = -3.0 * Om0 * a**(-4)
    
    # d/da (OmL0 * a^P * exp(Q(a-1))) where P = -3(1+w0+wa), Q = 3wa
    P = -3.0 * (1.0 + w0 + wa)
    Q = 3.0 * wa
    power_part = a**P
    exp_part = np.exp(Q * (a - 1.0))
    
    # Derivative of a^P is P * a^(P-1)
    # Derivative of exp(Q(a-1)) is Q * exp(Q(a-1))
    # Product rule:
    d_power = P * a**(P - 1.0)
    d_exp = Q * exp_part
    
    d_term_l = OmL0 * (d_power * exp_part + power_part * d_exp)
    
    return d_term_m + d_term_l

def _Om_m(a, Om0, w0, wa):
    """Calculate Omega_m(a)."""
    E2_val = _E2(a, Om0, w0, wa)
    return Om0 * a**(-3) / E2_val

def _q(a, Om0, w0, wa):
    """Calculate q = 1 - Om_m(a)."""
    return 1.0 - _Om_m(a, Om0, w0, wa)

def _u(a, w0, wa):
    """Calculate u = 1 + w0 + wa*(1-a)."""
    return 1.0 + w0 + wa * (1.0 - a)

def _gamma_model(a, Om0, w0, wa, coeffs):
    """Calculate gamma from the compact ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    q_val = _q(a, Om0, w0, wa)
    u_val = _u(a, w0, wa)
    
    g = c0 + c1 * q_val + c2 * q_val**2 + c3 * u_val + c4 * wa * a + c5 * u_val * q_val
    return g

def _f_model(a, Om0, w0, wa, coeffs):
    """Calculate f = Om_m(a)^gamma(a)."""
    Om_m_val = _Om_m(a, Om0, w0, wa)
    g_val = _gamma_model(a, Om0, w0, wa, coeffs)
    # Ensure Om_m > 0 for power
    if Om_m_val <= 0:
        return 0.0
    return Om_m_val ** g_val

def _reference_ode_rhs(a, y, Om0, w0, wa):
    """RHS for the growth factor ODE."""
    D, D_prime = y
    
    E2_val = _E2(a, Om0, w0, wa)
    E2_prime_val = _E2_prime(a, Om0, w0, wa)
    Om_m_val = _Om_m(a, Om0, w0, wa)
    
    # ODE: D'' + (3/(2a) + 0.5 * E2'/E2) D' - 1.5 * Om_m / a * D = 0
    # D'' = 1.5 * Om_m / a * D - (3/(2a) + 0.5 * E2'/E2) D'
    
    coeff_D_prime = 1.5 / a + 0.5 * E2_prime_val / E2_val
    coeff_D = 1.5 * Om_m_val / a
    
    D_double_prime = coeff_D * D - coeff_D_prime * D_prime
    
    return [D_prime, D_double_prime]

def _get_reference_solution(Om0, w0, wa, a_grid):
    """Solve the ODE for reference D and f."""
    # Solve from a=1 down to min(a_grid) or vice versa.
    # Standard is to solve from a=1 backwards or use initial conditions at a=1.
    # D(1)=1, D'(1) ~ 0.55 * Om_m(1) = 0.55 * Om0
    # Actually, D'(1) is determined by the ODE. We can use a small step or known approx.
    # Better: Solve forward from a_min to 1? No, growth is usually integrated forward.
    # But we need D(1) normalized.
    # Let's solve from a=1 to a_min (backwards) or a_min to 1 (forwards).
    # Forwards is more stable usually if a_min > 0.
    # Let's pick a_min = 0.1.
    
    a_min = min(a_grid)
    a_max = 1.0
    
    # Initial conditions at a=1
    # D(1) = 1
    # D'(1) can be approximated by 0.55 * Om0, or we can solve a small step.
    # To be precise, we can use the fact that D'(a) = f(a) * D(a) / a.
    # At a=1, D'(1) = f(1) * 1.
    # f(1) = Om0^gamma(1).
    # But we don't know gamma(1) from reference.
    # We can use the ODE at a=1 to find D'(1) if we knew D''? No.
    # Standard trick: Solve the ODE. The initial slope is determined by the physics.
    # Actually, we can just set D(1)=1 and D'(1)=0.55*Om0 as a good start, 
    # but for high precision reference, we should ensure consistency.
    # However, the ODE is homogeneous. Any solution is valid up to scale.
    # We normalize D(1)=1. The shape is fixed by the ODE.
    # We need one more condition. Usually, we assume D'(1) is close to 0.55 Om0.
    # But actually, the ODE determines the ratio D'/D at any point given the parameters.
    # Wait, the ODE is 2nd order. We need 2 initial conditions.
    # D(1)=1. What is D'(1)?
    # In the early universe, D ~ a. So D'(1) is not 1.
    # We can use the fact that for small a, D ~ a.
    # Let's solve from a=0.01 to 1.0.
    # At a=0.01, D ~ a. So D(0.01) = 0.01, D'(0.01) = 1.
    # This is a robust initial condition for the linear era.
    
    a_start = 0.01
    y0 = [a_start, 1.0]
    
    sol = solve_ivp(_reference_ode_rhs, [a_start, a_max], y0, args=(Om0, w0, wa), 
                    t_eval=a_grid, method='RK45', rtol=1e-8, atol=1e-10)
    
    if not sol.success:
        return None, None
    
    D_ref = sol.y[0]
    D_prime_ref = sol.y[1]
    
    # Normalize D(1) to 1
    # Find index closest to 1.0
    idx_1 = np.argmin(np.abs(a_grid - 1.0))
    norm_factor = D_ref[idx_1]
    D_ref = D_ref / norm_factor
    
    # Calculate f_ref = a * D' / D
    f_ref = a_grid * D_prime_ref / D_ref
    
    return f_ref, D_ref

def _calculate_error(coeffs, a_grid, cosmologies):
    """Calculate max relative error for given coefficients."""
    max_err_f = 0.0
    max_err_d = 0.0
    
    for Om0, w0, wa in cosmologies:
        # Get reference
        f_ref, D_ref = _get_reference_solution(Om0, w0, wa, a_grid)
        if f_ref is None:
            continue
            
        # Calculate model
        f_model_vals = np.array([_f_model(a, Om0, w0, wa, coeffs) for a in a_grid])
        
        # Calculate D_model by integrating f_model
        # ln D_model(a) = - integral from a to 1 of f(e^x) dx
        # Let's use numerical integration on the grid
        # We have f_model at a_grid points.
        # We need integral from a_i to 1.
        # Let's compute cumulative integral from 1 down to a_min.
        # dx = d(ln a) = da/a.
        # integral from a to 1 of f(a) da/a.
        
        # Sort a_grid if not sorted, but it is.
        # We want D_model(a_i) / D_model(1).
        # D_model(1) = 1.
        # ln D_model(a_i) = - integral_{a_i}^{1} f(a) da/a.
        
        # Let's compute the integral from the end (a=1) backwards.
        # integral_i = integral_{a_i}^{1} f(a) da/a.
        # We can use trapz on the points [a_i, ..., 1].
        
        D_model_vals = np.ones_like(a_grid)
        
        # For each point, we need the integral from that point to 1.
        # To be efficient, we can compute the cumulative sum.
        # Let's define g(a) = f(a)/a.
        # integral_{a_i}^{1} g(a) da.
        
        g_vals = f_model_vals / a_grid
        
        # We need integral from a_i to 1.
        # Let's compute total integral from a_min to 1, then subtract integral from a_min to a_i.
        # Or just loop, since a_grid is small (e.g. 50 points) and cosmologies are few (e.g. 10).
        # 50 * 10 * 10 = 5000 ops. Very fast.
        
        for i, a_i in enumerate(a_grid):
            # Points from a_i to 1
            mask = a_grid >= a_i
            a_pts = a_grid[mask]
            g_pts = g_vals[mask]
            
            # Integrate g_pts over a_pts
            # Use trapz
            integral_val = np.trapz(g_pts, a_pts)
            
            D_model_vals[i] = np.exp(-integral_val)
            
            # Check gamma constraint
            g_val = _gamma_model(a_i, Om0, w0, wa, coeffs)
            if g_val < 0.1 or g_val > 1.2:
                # Heavy penalty
                return 1e6, 1e6, 1e6

        # Relative errors
        # f error
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_f = np.abs((f_model_vals - f_ref) / f_ref)
            # D error
            rel_err_d = np.abs((D_model_vals - D_ref) / D_ref)
            
        # Filter out non-finite
        rel_err_f = rel_err_f[np.isfinite(rel_err_f)]
        rel_err_d = rel_err_d[np.isfinite(rel_err_d)]
        
        if len(rel_err_f) > 0:
            max_err_f = max(max_err_f, np.max(rel_err_f))
        if len(rel_err_d) > 0:
            max_err_d = max(max_err_d, np.max(rel_err_d))
            
    return max_err_f, max_err_d, max(max_err_f, max_err_d)

def construct():
    # Define representative cosmologies for fitting
    # w0 in [-1.2, -0.8], wa in [-0.5, 0.5], Om0 in [0.25, 0.35]
    cosmologies = [
        (0.30, -1.0, 0.0),  # LCDM
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.30, -0.8, 0.0),
        (0.30, -1.2, 0.0),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -0.9, 0.2),
        (0.30, -1.1, -0.2),
        (0.30, -0.5, 0.0), # High w0
    ]
    
    # Grid of a values for evaluation
    a_grid = np.linspace(0.1, 1.0, 50)
    
    # Initial guess: Linder approximation coefficients?
    # gamma ~ 0.55 + 0.05(1+w).
    # q ~ 0.5 Om_m - 0.5 Om_L.
    # This is complex. Let's start with c0=0.55, others 0.
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0
    best_score = 1e9
    
    def objective(x):
        nonlocal best_coeffs, best_score
        # Check time
        if time.time() - START_TIME > TIME_LIMIT:
            return 1e9
            
        err_f, err_d, total_err = _calculate_error(x, a_grid, cosmologies)
        
        if total_err < best_score:
            best_score = total_err
            best_coeffs = x.copy()
            
        return total_err
        
    # Use Nelder-Mead for robustness without gradients
    # Limit function evaluations
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
    
    # Check if we should do another pass with a different start if time permits
    if time.time() - START_TIME < TIME_LIMIT - 10:
        # Try a few random starts
        for _ in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            x_rand = np.random.uniform(-1, 1, 6)
            x_rand[0] = 0.55 # Keep c0 near 0.55
            res2 = minimize(objective, x_rand, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-5})
            # best_coeffs is updated inside objective
            
    # Final check on best_coeffs
    # Ensure coefficients are within [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    
    # Verify gamma constraint on best_coeffs
    # If violated, fall back to seed
    valid = True
    for Om0, w0, wa in cosmologies:
        for a in a_grid:
            g = _gamma_model(a, Om0, w0, wa, best_coeffs)
            if g < 0.1 or g > 1.2:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    return best_coeffs.tolist()

if __name__ == "__main__":
    # Small self-test
    print("Running optimization...")
    coeffs = construct()
    print(f"Result: {coeffs}")
