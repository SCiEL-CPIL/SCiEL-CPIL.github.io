"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_phys(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    H0 = np.ones_like(x)
    H1 = 2.0 * x
    for k in range(2, n + 1):
        H2 = 2.0 * x * H1 - 2.0 * (k - 1) * H0
        H0, H1 = H1, H2
    return H1


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 for a test function of the form
    f(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    where a12 is determined by the constraint f(0) = 0.
    
    The objective is the uncertainty product Delta_x * Delta_k for this
    Fourier-invariant function. Since the function is a Fourier eigenfunction
    (all degrees are multiples of 4, so (-i)^n = 1), Delta_x = Delta_k,
    and the product is Delta_x^2.
    
    We minimize this to get the tightest uncertainty bound.
    """
    # Compute a12 from constraint f(0) = 0
    a12 = -(a0 + 12*a4 + 1680*a8) / 2032128.0
    
    # Use numerical integration on a grid
    x = np.linspace(-12, 12, 2000)
    dx = x[1] - x[0]
    
    H0 = np.ones_like(x)
    H4 = _hermite_phys(4, x)
    H8 = _hermite_phys(8, x)
    H12 = _hermite_phys(12, x)
    
    f = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * np.exp(-x**2/2)
    
    # L2 norm squared
    norm_sq = np.sum(f**2) * dx
    if norm_sq < 1e-30:
        return 1e10
    
    # Second moment <x^2>
    x2_norm = np.sum(x**2 * f**2) * dx
    var_x = x2_norm / norm_sq  # Since <x> = 0 by symmetry
    
    # For Fourier eigenfunctions with (-i)^n = 1, Delta_k = Delta_x
    # So the uncertainty product is var_x (which is Delta_x^2)
    bound = var_x
    
    return bound


def _angular_objective(angles):
    """
    Evaluate the bound given two angles (theta, phi) on the unit sphere.
    The coefficients are (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)).
    This exploits the scale invariance of the ratio-based objective.
    """
    theta, phi = angles
    a0 = np.sin(theta) * np.cos(phi)
    a4 = np.sin(theta) * np.sin(phi)
    a8 = np.cos(theta)
    return _compute_bound(a0, a4, a8)


def construct():
    start_time = time.time()
    deadline = 210.0  # seconds (evaluator kills at 210s)
    safety_margin = 15.0  # return 15s before deadline
    
    # Initial guess from the seed
    x0_seed = np.array([-1.13, 0.04, 0.00030864])
    
    # Convert seed to angular parameters
    norm_seed = np.linalg.norm(x0_seed)
    if norm_seed > 1e-15:
        cos_theta = x0_seed[2] / norm_seed
        sin_theta = np.sqrt(x0_seed[0]**2 + x0_seed[1]**2) / norm_seed
        if sin_theta > 1e-15:
            cos_phi = x0_seed[0] / (norm_seed * sin_theta)
            sin_phi = x0_seed[1] / (norm_seed * sin_theta)
        else:
            cos_phi, sin_phi = 1.0, 0.0
        theta_seed = np.arccos(np.clip(cos_theta, -1, 1))
        phi_seed = np.arctan2(sin_phi, cos_phi)
    else:
        theta_seed, phi_seed = np.pi/2, 0.0
    
    # Phase 1: High-resolution grid search on the 2D angular domain
    # theta in [0, pi], phi in [0, 2*pi]
    n_theta = 80
    n_phi = 120
    
    theta_vals = np.linspace(0, np.pi, n_theta)
    phi_vals = np.linspace(0, 2*np.pi, n_phi)
    
    best_val = _compute_bound(*x0_seed)
    best_angles = np.array([theta_seed, phi_seed])
    
    grid_start = time.time()
    
    for i, th in enumerate(theta_vals):
        if time.time() - start_time > deadline - safety_margin:
            break
        
        # Vectorized evaluation over phi for fixed theta
        a0 = np.sin(th) * np.cos(phi_vals)
        a4 = np.sin(th) * np.sin(phi_vals)
        a8 = np.cos(th) * np.ones_like(phi_vals)
        
        for j in range(n_phi):
            if time.time() - start_time > deadline - safety_margin:
                break
            val = _compute_bound(a0[j], a4[j], a8[j])
            if val < best_val:
                best_val = val
                best_angles = np.array([th, phi_vals[j]])
    
    # Phase 2: Local refinement around the best grid point
    if time.time() - start_time < deadline - safety_margin:
        try:
            result = minimize(_angular_objective, best_angles, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            val = result.fun
            if val < best_val:
                best_val = val
                best_angles = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Convert back to coefficient space
    theta_opt, phi_opt = best_angles
    a0_opt = np.sin(theta_opt) * np.cos(phi_opt)
    a4_opt = np.sin(theta_opt) * np.sin(phi_opt)
    a8_opt = np.cos(theta_opt)
    
    # Verify the angular result and compare with seed
    angular_val = _compute_bound(a0_opt, a4_opt, a8_opt)
    
    # Also try a local refinement in coefficient space from the angular result
    if time.time() - start_time < deadline - safety_margin:
        try:
            x0_coeff = np.array([a0_opt, a4_opt, a8_opt])
            result2 = minimize(_compute_bound, x0_coeff, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            val2 = result2.fun
            if val2 < best_val:
                best_val = val2
                return result2.x.copy()
        except Exception:
            pass
    
    # Return the best coefficients found
    if angular_val < _compute_bound(*x0_seed):
        return np.array([a0_opt, a4_opt, a8_opt])
    else:
        return x0_seed

# EVOLVE-BLOCK-END
