import numpy as np
import time
from scipy.optimize import minimize

def _hermite_poly(n, x):
    """Compute the physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        h_prev = np.ones_like(x)
        h_curr = 2 * x
        for k in range(2, n + 1):
            h_next = 2 * x * h_curr - 2 * (k - 1) * h_prev
            h_prev = h_curr
            h_curr = h_next
        return h_curr

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product upper bound for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is determined by the constraint g(0) = 0.
    
    The uncertainty product A(f)*A(F) for a function f is related to the C4 constant.
    We compute the relevant integrals to estimate the bound.
    """
    # Hermite polynomials at x=0 for the constraint
    # H0(0) = 1, H4(0) = 12, H8(0) = 2520, H12(0) = 579600 (physicists' convention)
    # Actually let's use the standard physicists' Hermite: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
    # H0 = 1, H1 = 2x, H2 = 4x^2 - 2, H4 = 16x^4 - 48x^2 + 12
    # H8(0) = 2520? Let me compute: H_n(0) for even n: H_{2k}(0) = (-1)^k (2k)! / k!
    # H0(0) = 1
    # H2(0) = -2
    # H4(0) = 12
    # H6(0) = -120
    # H8(0) = 1680
    # H12(0) = 277200? Let's compute: H_{2k}(0) = (-1)^k (2k)!/k!
    # k=6: H12(0) = (-1)^6 * 12! / 6! = 479001600 / 720 = 665280
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Constraint: a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # => a12 = -(a0 + 12*a4 + 1680*a8) / 665280
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 665280.0
    
    # Now compute the uncertainty product bound.
    # For g(x) = sum(a_n * H_n(x)) * exp(-x^2/2), we need to compute
    # the L2 norm of g and its Fourier transform, and the position/momentum spreads.
    
    # Use numerical integration on a fine grid
    x = np.linspace(-30, 30, 20001)
    dx = x[1] - x[0]
    
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x**2 / 2.0)
    
    # Check g(0) is approximately 0
    idx0 = np.argmin(np.abs(x))
    # g[0] should be ~0 by construction
    
    # L2 norm of g
    l2_g_sq = np.sum(g**2) * dx
    l2_g = np.sqrt(l2_g_sq)
    
    # L4 norm of g
    l4_g_sq = np.sum(np.abs(g)**4) * dx
    l4_g = np.sqrt(l4_g_sq) # Note: ||g||_4 = (integral |g|^4)^{1/4}, so l4_g_sq here is actually ||g||_4^2
    # Let's be precise:
    # ||g||_4 = (integral |g|^4 dx)^{1/4}
    # ||g||_4^2 = (integral |g|^4 dx)^{1/2}
    # The quantity we want is C4 = ||g||_4 * ||\hat{g}||_4 / ||g||_2
    # Since ||\hat{g}||_4 = ||g||_4, C4 = ||g||_4^2 / ||g||_2
    
    # integral |g|^4 dx is l4_g_sq_integral
    l4_integral = np.sum(np.abs(g)**4) * dx
    norm_4 = l4_integral ** 0.25
    norm_4_sq = l4_integral ** 0.5
    
    c4 = norm_4_sq / l2_g
    
    return c4

def construct():
    # Return the known state-of-the-art coefficients as initial guess
    a0_init = -113.0 / 100.0
    a4_init = 1.0 / 25.0
    a8_init = 1.0 / 3240.0
    
    # Try to optimize
    start_time = time.time()
    deadline = start_time + 180.0  # Must finish within 180s
    
    best_coeffs = np.array([a0_init, a4_init, a8_init])
    best_val = _compute_bound(a0_init, a4_init, a8_init)
    
    # Try multiple starting points with different random seeds
    rng = np.random.default_rng(42)
    
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        if i == 0:
            x0 = np.array([a0_init, a4_init, a8_init])
        else:
            # Random perturbation around the known solution
            scale = 0.5 * (i + 1) / n_starts
            x0 = np.array([a0_init, a4_init, a8_init]) + scale * rng.standard_normal(3)
        
        try:
            result = minimize(
                _compute_bound,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _compute_bound,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return best_coeffs
