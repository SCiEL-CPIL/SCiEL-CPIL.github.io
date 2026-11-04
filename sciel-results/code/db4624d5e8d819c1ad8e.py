"""Minimize the uncertainty-product upper bound C4 for a Hermite-Gaussian test function.

g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
with c12 chosen so that g(0) = 0.
We minimize the upper bound on A(f)*A(F) which for a normalized function relates to
the product of position and momentum variances (or a related quantity).

The strategy: use the fact that for Hermite-Gaussian functions, the variances can be
computed exactly via the coefficients. We set up a generalized eigenvalue problem
or use scipy's minimize to find optimal (c0, c4, c8).
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_coefficients(c0, c4, c8):
    """Return the full coefficient vector [c0, c4, c8, c12] with c12 set so g(0)=0.
    
    H_n(0): H0(0)=1, H4(0)=12, H8(0)=1344, H12(0)=498960 (physicists' Hermite)
    Wait, let me use the standard (probabilists' or physicists') convention.
    
    For the Hermite-Gaussian in quantum mechanics, we use physicists' Hermite polynomials:
    H0(x) = 1
    H1(x) = 2x
    H2(x) = 4x^2 - 2
    H3(x) = 8x^3 - 12x
    H4(x) = 16x^4 - 48x^2 + 12
    H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1344
    H12(x) = 4096x^12 - 73728x^10 + 552960x^8 - 2150400x^6 + 3675120x^4 - 2764800x^2 + 498960
    
    g(0) = c0*1 + c4*12 + c8*1344 + c12*498960 = 0
    => c12 = -(c0 + 12*c4 + 1344*c8) / 498960
    """
    c12 = -(c0 + 12.0 * c4 + 1344.0 * c8) / 498960.0
    return np.array([c0, c4, c8, c12])


def _compute_variances(c0, c4, c8, c12):
    """Compute <x^2> and <p^2> for the normalized Hermite-Gaussian function.
    
    For g(x) = sum c_n H_n(x) exp(-x^2/2), normalized:
    The inner product <H_m H_n exp(-x^2)> = sqrt(pi) * 2^n * n! * delta_mn
    
    So ||g||^2 = sum |c_n|^2 * sqrt(pi) * 2^n * n!
    
    For <x^2>: use x^2 = (1/2)(H2 + 2) in terms of Hermite polynomials... 
    Actually, let's use the ladder operator approach or direct integration.
    
    Using physicists' Hermite: x H_n = (1/2)H_{n+1} + n H_{n-1}
    So x^2 H_n = (1/4)H_{n+2} + (n+1/2)H_n + (n/4)H_{n-2}
    
    Wait, let me be more careful:
    x H_n(x) = (1/2) H_{n+1}(x) + n H_{n-1}(x)
    
    x^2 H_n = x * [(1/2) H_{n+1} + n H_{n-1}]
           = (1/2)[(1/2)H_{n+2} + (n+1)H_n] + n[(1/2)H_n + (n-1)H_{n-2}]
           = (1/4)H_{n+2} + (n/2 + 1/2)H_n + (n/2)H_n + n(n-1)/2 * H_{n-2}
           = (1/4)H_{n+2} + (n+1)H_n + n(n-1)/2 * H_{n-2}
    
    So <x^2> = sum_{m,n} c_m c_n <H_m x^2 H_n> / ||g||^2
    
    <H_m H_k exp(-x^2)> = sqrt(pi) 2^k k! delta_mk
    
    For x^2 H_n: the H_{n+2} term contributes to m=n+2, H_n to m=n, H_{n-2} to m=n-2.
    
    <H_m x^2 H_n> = sqrt(pi) [
        (1/4) * 2^{n+2}(n+2)! * delta_{m,n+2}
        + (n+1) * 2^n n! * delta_{m,n}
        + n(n-1)/2 * 2^{n-2}(n-2)! * delta_{m,n-2}
    ]
    
    So <x^2> = (1/||g||^2) * sum_n |c_n|^2 sqrt(pi) [
        (1/4)*2^{n+2}(n+2)! * (c_{n+2} conjugate... wait, we need cross terms)
    ]
    
    Actually for real coefficients:
    <x^2> = (1/||g||^2) * sum_{m,n} c_m c_n <H_m x^2 H_n>
    
    The non-zero terms are when m = n+2, m = n, or m = n-2.
    
    Let me just compute this numerically using Gauss-Hermite quadrature for accuracy.
    """
    # Use Gauss-Hermite quadrature with enough points
    # We integrate f(x) exp(-x^2) dx
    n_pts = 200
    x, w = np.polynomial.hermite.hermgauss(n_pts)
    
    # Hermite polynomials at quadrature points
    # H_0, H_4, H_8, H_12
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1344
    H12 = 4096*x**12 - 73728*x**10 + 552960*x**8 - 2150400*x**6 + 3675120*x**4 - 2764800*x**2 + 498960
    
    H = np.column_stack([H0, H4, H8, H12])
    coeffs = np.array([c0, c4, c8, c12])
    
    g = H @ coeffs  # g(x) values at quadrature points (without exp(-x^2/2))
    
    # The actual function is g(x)*exp(-x^2/2), so |g(x)*exp(-x^2/2)|^2 = g(x)^2 * exp(-x^2)
    norm_sq = np.sum(w * g**2)
    
    if norm_sq < 1e-30:
        return 1e10, 1e10
    
    # <x^2>
    x2 = np.sum(w * g**2 * x**2) / norm_sq
    
    # <p^2> where p = -i d/dx, so <p^2> = <(dg/dx)^2> in the appropriate sense
    # Actually for real functions: <p^2> = integral (g'(x))^2 exp(-x^2) dx / integral g^2 exp(-x^2) dx
    # Wait, the momentum operator in this basis... Let me think.
    # If psi(x) = g(x) exp(-x^2/2), then p = -i d/dx
    # <p^2> = integral |psi'(x)|^2 dx = integral |g'(x)exp(-x^2/2) - x g(x) exp(-x^2/2)|^2 dx
    #       = integral [g'(x)^2 - 2x g(x) g'(x) + x^2 g(x)^2] exp(-x^2) dx
    # By integration by parts: integral g' * g * exp(-x^2) dx = -integral g^2 x exp(-x^2) dx
    # So <p^2> = integral g'^2 exp(-x^2) dx + integral x^2 g^2 exp(-x^2) dx - 2*(-integral x g^2 exp(-x^2))... 
    # Hmm, let me just compute g'(x) numerically or analytically.
    
    # g(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # g'(x) = c4*H4' + c8*H8' + c12*H12'  (H0' = 0)
    # H_n'(x) = 2n H_{n-1}(x)
    # H4' = 8*H3, H8' = 16*H7, H12' = 24*H11
    
    # H3 = 8x^3 - 12x
    # H7 = 128x^7 - 1792x^5 + 6720x^3 - 6720x
    # H11 = 2048x^11 - 45056x^9 + 378880x^7 - 1552320x^5 + 2540160x^3 - 1451520x
    
    H3 = 8*x**3 - 12*x
    H7 = 128*x**7 - 1792*x**5 + 6720*x**3 - 6720*x
    H11 = 2048*x**11 - 45056*x**9 + 378880*x**7 - 1552320*x**5 + 2540160*x**3 - 1451520*x
    
    H4p = 8*H3
    H8p = 16*H7
    H12p = 24*H11
    
    gp = c4*H4p + c8*H8p + c12*H12p
    
    p2 = np.sum(w * gp**2) / norm_sq
    
    return x2, p2


def _objective(params):
    """Compute the upper bound C4 ~ sqrt(<x^2> * <p^2>) or related quantity."""
    c0, c4, c8 = params
    c12 = -(c0 + 12.0*c4 + 1344.0*c8) / 498960.0
    x2, p2 = _compute_variances(c0, c4, c8, c12)
    if x2 < 0 or p2 < 0:
        return 1e10
    # The uncertainty product bound is proportional to sqrt(<x^2><p^2>)
    # The actual C4 constant may have a specific form, but minimizing the product
    # of variances is the right objective.
    return np.sqrt(max(x2, 0) * max(p2, 0))


def construct():
    """Find optimal coefficients (c0, c4, c8) minimizing the uncertainty bound."""
    start_time = time.time()
    deadline = 180.0  # safety margin
    
    # Initial guess: the prior state of the art
    x0 = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    best_params = x0.copy()
    best_val = _objective(x0)
    
    # Try multiple starting points
    rng = np.random.RandomState(42)
    
    # First, try the known optimum
    results = [(best_val, best_params)]
    
    # Try Nelder-Mead from several random starting points
    n_random_starts = 50
    for i in range(n_random_starts):
        if time.time() - start_time > deadline - 30:
            break
        
        scale = 1.0
        x_rand = x0 + rng.randn(3) * scale
        res = minimize(_objective, x_rand, method='Nelder-Mead', 
                      options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.success or res.fun < best_val:
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
                results.append((best_val, best_params))
    
    # Try L-BFGS-B with gradient (numerical)
    if time.time() - start_time < deadline - 60:
        for i in range(20):
            if time.time() - start_time > deadline - 30:
                break
            x_rand = x0 + rng.randn(3) * 0.5
            res = minimize(_objective, x_rand, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
    
    # Final polish with high-precision Nelder-Mead
    if time.time() - start_time < deadline - 10:
        res = minimize(_objective, best_params, method='Nelder-Mead',
                      options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-16,
                               'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
