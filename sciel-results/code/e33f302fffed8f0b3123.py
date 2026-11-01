"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

_start_time = time.time()
_DEADLINE = 180.0  # safety margin before 210s kill

def _hermite_physicist(n, x):
    """Physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    H_nm2 = np.ones_like(x)
    H_nm1 = 2.0 * x
    for k in range(2, n + 1):
        H_n = 2.0 * x * H_nm1 - 2.0 * (k - 1) * H_nm2
        H_nm2 = H_nm1
        H_nm1 = H_n
    return H_nm1

def _compute_objective(coeffs, x_grid):
    """Compute the upper bound on C4 = A(f) * A(F(f)).
    
    For even Hermite functions, F(f) = f (up to convention), so C4 ~ A(f)^2.
    We compute A(f) as the L1 norm of f(x) = sum a_i H_i(x) * exp(-x^2/2).
    """
    a0, a4, a8 = coeffs
    # a12 determined by g(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 4790016.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Compute f(x) on the grid
    H0 = _hermite_physicist(0, x_grid)
    H4 = _hermite_physicist(4, x_grid)
    H8 = _hermite_physicist(8, x_grid)
    H12 = _hermite_physicist(12, x_grid)
    
    f = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x_grid**2 / 2.0)
    
    # A(f) = integral of |f(x)| dx (L1 norm)
    # Use np.trapezoid if available (NumPy 2.0+), else np.trapz
    try:
        A_f = np.trapezoid(np.abs(f), x_grid)
    except AttributeError:
        A_f = np.trapz(np.abs(f), x_grid)
    
    # Since F(f) = f for even Hermite (standard convention), C4 = A_f * A_f
    C4 = A_f * A_f
    return C4

def _get_tangent_basis(u):
    """Generate an orthonormal basis for the tangent space of the sphere at u."""
    # u is a 3D unit vector
    # Find a vector v not parallel to u
    if abs(u[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    
    # Project v onto tangent space
    t1 = v - np.dot(v, u) * u
    t1_norm = np.linalg.norm(t1)
    if t1_norm < 1e-10:
        # Fallback
        v = np.array([0.0, 0.0, 1.0])
        t1 = v - np.dot(v, u) * u
        t1_norm = np.linalg.norm(t1)
    t1 /= t1_norm
    
    # t2 is cross product
    t2 = np.cross(u, t1)
    t2 /= np.linalg.norm(t2)
    
    return t1, t2

def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8)."""
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Not enough time, return seed
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Set up grid for numerical integration
    x_grid = np.linspace(-20, 20, 10000)
    
    # Seed initial guess
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_coeffs = x0.copy()
    best_val = _compute_objective(best_coeffs, x_grid)
    
    # Reparameterize: a = R * u, where u is unit vector
    # We will search for u on the sphere and R > 0
    
    def objective_for_direction(u):
        """Find optimal R for a given unit direction u and return min objective."""
        # a_i = R * u_i
        # Objective J(R) = _compute_objective(R*u, x_grid)
        # J(R) is likely convex in R (quadratic-ish due to A_f^2)
        # Use minimize_scalar to find optimal R
        
        def obj_R(log_R):
            R = np.exp(log_R)
            coeffs = R * u
            return _compute_objective(coeffs, x_grid)
        
        # Initial guess for R: use the norm of x0 projected onto u, or just 1.0
        # x0 norm is small. Let's guess based on x0.
        R_init = np.dot(x0, u)
        if R_init < 1e-6:
            R_init = 1.0
        log_R_init = np.log(R_init)
        
        try:
            res = minimize_scalar(obj_R, bounds=(np.log(1e-4), np.log(1e4)), method='bounded', options={'xatol': 1e-6})
            if res.success:
                R_opt = np.exp(res.x)
                return res.fun, R_opt
            else:
                # Fallback
                R_opt = R_init
                return obj_R(log_R_init), R_opt
        except Exception:
            R_opt = R_init
            return obj_R(log_R_init), R_opt

    # Stochastic local search on the sphere
    rng = np.random.RandomState(42)
    
    # Initialize u with the direction of x0
    u_norm = np.linalg.norm(x0)
    if u_norm < 1e-10:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = x0 / u_norm
        
    # Check initial
    val, R = objective_for_direction(u)
    if val < best_val:
        best_val = val
        best_coeffs = R * u

    step_size = 0.1
    max_steps = 500
    
    for i in range(max_steps):
        elapsed = time.time() - _start_time
        if elapsed > _DEADLINE - 10.0:
            break
        
        # Generate random tangent vector
        t1, t2 = _get_tangent_basis(u)
        angle = rng.uniform(0, 2 * np.pi)
        magnitude = rng.uniform(0, step_size)
        
        # Move on sphere
        # u_new = cos(mag) * u + sin(mag) * (cos(angle)*t1 + sin(angle)*t2)
        # For small mag, this is an approximation. For larger, use rotation.
        # Simple vector addition and renormalization is robust enough for local search
        delta_u = magnitude * (np.cos(angle) * t1 + np.sin(angle) * t2)
        u_new = u + delta_u
        u_new_norm = np.linalg.norm(u_new)
        if u_new_norm < 1e-10:
            continue
        u_new /= u_new_norm
        
        # Evaluate new direction
        new_val, new_R = objective_for_direction(u_new)
        
        if new_val < best_val:
            best_val = new_val
            best_coeffs = new_R * u_new
            u = u_new
            # Shrink step size slightly on success
            step_size *= 0.95
        else:
            # Increase step size if stuck (simple adaptive)
            step_size *= 1.05
            if step_size > 1.0:
                step_size = 1.0
            if step_size < 0.01:
                step_size = 0.01
                
    # Final polish: refine R for the best u
    if best_val > 0:
        # Re-extract u from best_coeffs
        norm_bc = np.linalg.norm(best_coeffs)
        if norm_bc > 1e-10:
            u_best = best_coeffs / norm_bc
            val_final, R_final = objective_for_direction(u_best)
            if val_final < best_val:
                best_val = val_final
                best_coeffs = R_final * u_best

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
