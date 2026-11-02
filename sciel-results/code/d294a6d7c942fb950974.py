"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def _hermite_polynomials(x):
    """
    Computes H0, H4, H8, H12 for input x.
    Uses explicit formulas for speed and accuracy.
    H0(x) = 1
    H4(x) = x^4 - 6x^2 + 3
    H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H12(x) = x^12 - 54x^10 + 1080x^8 - 7560x^6 + 15120x^4 - 7560x^2 + 945
    """
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x8 * x4
    
    H0 = np.ones_like(x)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 54*x10 + 1080*x8 - 7560*x6 + 15120*x4 - 7560*x2 + 945
    
    return H0, H4, H8, H12

def _compute_objective(coeffs):
    """
    Computes the uncertainty product bound for the given coefficients.
    coeffs: [a0, a4, a8]
    Returns A(g)^2 = ||g||_4^2 / ||g||_2^3 for FT-invariant g.
    """
    a0, a4, a8 = coeffs
    
    # Enforce g(0) = 0 to find a12
    a12 = -(a0 + 3*a4 + 105*a8) / 945.0
    
    # Use higher-precision Gauss-Hermite quadrature (N=80 for degree 48 poly)
    N = 80
    nodes, weights = np.polynomial.hermite.hermgauss(N)
    
    # Evaluate P(x) at nodes for L2: integral of P(x)^2 e^{-x^2} dx
    H0, H4, H8, H12 = _hermite_polynomials(nodes)
    P_vals = a0*H0 + a4*H4 + a8*H8 + a12*H12
    L2_sq = np.sum(weights * P_vals**2)
    
    # Evaluate P(x) at scaled nodes for L4: integral of P(x)^4 e^{-2x^2} dx
    # Substitution x = t/sqrt(2), dx = dt/sqrt(2)
    x_scaled = nodes / np.sqrt(2.0)
    H0_s, H4_s, H8_s, H12_s = _hermite_polynomials(x_scaled)
    P_vals_s = a0*H0_s + a4*H4_s + a8*H8_s + a12*H12_s
    L4_4 = (1.0/np.sqrt(2.0)) * np.sum(weights * P_vals_s**4)
    
    # Avoid division by zero or negative values
    if L2_sq < 1e-15 or L4_4 < 0:
        return 1e15
    
    # A(g) = ||g||_4 / ||g||_2^{3/2}
    # A(g)^2 = ||g||_4^2 / ||g||_2^3 = (L4_4)^{1/2} / (L2_sq)^{3/2}
    # Minimize A(g)^2
    return np.sqrt(L4_4) / (L2_sq**1.5)

def construct():
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 210s kill
    
    # Seed solution
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_objective(seed)
    best_x = seed.copy()
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = [seed.copy()]
    
    # Local perturbations around seed
    for sigma in [0.01, 0.05, 0.1, 0.5]:
        for _ in range(5):
            x0 = seed + rng.normal(0, sigma, 3)
            starts.append(x0)
    
    # Global random starts
    for _ in range(10):
        x0 = rng.uniform(-2, 2, 3)
        starts.append(x0)
    
    # Structured starts based on symmetry considerations
    for a0 in [-0.5, -0.2, 0.0, 0.2, 0.5]:
        for a4 in [-0.2, -0.05, 0.0, 0.05, 0.2]:
            for a8 in [-0.01, -0.001, 0.0, 0.001, 0.01]:
                x0 = np.array([a0, a4, a8])
                starts.append(x0)
    
    # Remove duplicates and limit to reasonable number
    unique_starts = []
    for s in starts:
        is_dup = False
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-6):
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(s)
    starts = unique_starts[:40]  # Cap at 40 starts
    
    # Multi-start with Powell's method (derivative-free, good for low-dim)
    for x0 in starts:
        if time.time() > deadline - 15.0:
            break
        
        try:
            res = opt.minimize(_compute_objective, x0, method='Powell',
                               options={'xtol': 1e-12, 'ftol': 1e-14, 'maxiter': 200, 'maxfev': 5000})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # Final high-precision polish with Nelder-Mead
    if time.time() < deadline - 5.0:
        try:
            res = opt.minimize(_compute_objective, best_x, method='Nelder-Mead',
                               options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 2000, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Second polish if time remains
    if time.time() < deadline - 3.0:
        try:
            res = opt.minimize(_compute_objective, best_x, method='Powell',
                               options={'xtol': 1e-15, 'ftol': 1e-17, 'maxiter': 500, 'maxfev': 10000})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
