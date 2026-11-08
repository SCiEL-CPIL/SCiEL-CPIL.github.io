"""Program to minimize the upper bound C4 for the uncertainty inequality using Hermite-Gaussian test functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import factorial

_start_time = time.time()
_TIME_LIMIT = 165.0  # safety margin before 180s kill

def _compute_bound_analytical(a0, a4, a8):
    """
    Compute the uncertainty product bound analytically using Hermite-Gaussian
    orthogonality relations.
    
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    
    Using orthogonality: ∫ H_m H_n e^{-x^2} dx = sqrt(pi) * 2^n * n! * delta_{mn}
    
    For x^2 moments, use: x^2 H_n = (1/2)H_{n+2} + n H_n + (n/2)H_{n-2}
    (with appropriate boundary terms for small n)
    
    Specifically:
    x^2 H_0 = H_2 + 0*H_0 + 0 = H_2
    x^2 H_4 = (1/2)H_6 + 4*H_4 + 2*H_2
    x^2 H_8 = (1/2)H_10 + 8*H_8 + 4*H_6
    x^2 H_12 = (1/2)H_14 + 12*H_12 + 6*H_10
    """
    # Enforce g(0) = 0
    # H_n(0) for even n: H_0(0)=1, H_4(0)=3, H_8(0)=105, H_12(0)=10395
    a12 = -(a0 * 1.0 + a4 * 3.0 + a8 * 105.0) / 10395.0
    
    # Norm: ||g||^2 = sum of |c_n|^2 * ||H_n e^{-x^2/2}||^2
    # ||H_n e^{-x^2/2}||^2 = ∫ H_n^2 e^{-x^2} dx = sqrt(pi) * 2^n * n!
    sqrt_pi = np.sqrt(np.pi)
    
    def norm_sq_n(n):
        return sqrt_pi * (2.0 ** n) * factorial(n, exact=False)
    
    c = np.array([a0, a4, a8, a12])
    ns = np.array([0, 4, 8, 12])
    
    # Compute norm squared
    norm_sq = 0.0
    for ci, ni in zip(c, ns):
        norm_sq += ci**2 * norm_sq_n(ni)
    
    if norm_sq < 1e-30:
        return 1e10
    
    # Compute ∫ x^2 |g|^2 dx
    # x^2 H_n = (1/2)H_{n+2} + n*H_n + (n/2)H_{n-2} for n >= 2
    # For n=0: x^2 H_0 = H_2
    # For n=2: x^2 H_2 = (1/2)H_4 + 2*H_2 + H_0
    # General: x^2 H_n = (1/2)H_{n+2} + n*H_n + (n/2)H_{n-2}
    
    # We need ∫ x^2 H_m H_n e^{-x^2} dx
    # = (1/2)∫ H_{m+2} H_n e^{-x^2} dx + m ∫ H_m H_n e^{-x^2} dx + (m/2)∫ H_{m-2} H_n e^{-x^2} dx
    # Using orthogonality, this is nonzero only when indices match.
    
    # Let's compute ∫ x^2 H_m H_n e^{-x^2} dx for our specific m, n in {0,4,8,12}
    # For m=n: 
    #   ∫ x^2 H_n^2 e^{-x^2} dx = (1/2)∫ H_{n+2} H_n e^{-x^2} + n ∫ H_n^2 e^{-x^2} + (n/2)∫ H_{n-2} H_n e^{-x^2}
    #   = n * sqrt(pi) * 2^n * n!  (cross terms vanish by orthogonality)
    # Wait, that's not right. Let me recalculate.
    
    # Actually: x^2 H_n(x) = (1/2) H_{n+2}(x) + n H_n(x) + (n/2) H_{n-2}(x)
    # So: ∫ x^2 H_m H_n e^{-x^2} dx = (1/2)∫ H_{m+2} H_n e^{-x^2} + m ∫ H_m H_n e^{-x^2} + (m/2)∫ H_{m-2} H_n e^{-x^2}
    
    # By orthogonality: ∫ H_a H_b e^{-x^2} dx = sqrt(pi) 2^b b! delta_{a,b} (if a>=b) or similar
    # Actually: ∫ H_m H_n e^{-x^2} dx = sqrt(pi) 2^{min(m,n)} min(m,n)! * delta_{m,n} ... no
    # Correct: ∫ H_m(x) H_n(x) e^{-x^2} dx = sqrt(pi) 2^n n! delta_{m,n} for m=n
    
    # So ∫ x^2 H_m H_n e^{-x^2} dx is nonzero when:
    # m+2 = n, or m = n, or m-2 = n
    # i.e., n = m+2, n = m, or n = m-2
    
    def moment_x2(m, n):
        """Compute ∫ x^2 H_m H_n e^{-x^2} dx"""
        result = 0.0
        # Term 1: (1/2) ∫ H_{m+2} H_n e^{-x^2} dx
        if m + 2 == n:
            result += 0.5 * sqrt_pi * (2.0**n) * factorial(n, exact=False)
        # Term 2: m ∫ H_m H_n e^{-x^2} dx  
        if m == n:
            result += m * sqrt_pi * (2.0**m) * factorial(m, exact=False)
        # Term 3: (m/2) ∫ H_{m-2} H_n e^{-x^2} dx
        if m >= 2 and m - 2 == n:
            result += (m/2.0) * sqrt_pi * (2.0**n) * factorial(n, exact=False)
        return result
    
    moment_x2_sq = 0.0
    for i, (ci, ni) in enumerate(zip(c, ns)):
        for j, (cj, nj) in enumerate(zip(c, ns)):
            moment_x2_sq += ci * cj * moment_x2(ni, nj)
    
    # The uncertainty product bound
    bound = norm_sq * moment_x2_sq
    return bound

def _objective(a):
    return _compute_bound_analytical(a[0], a[1], a[2])

def construct():
    """
    Returns three coefficients [a0, a4, a8] for the Hermite combination.
    a12 is determined by g(0)=0.
    Minimizes the upper bound on C4.
    """
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        return np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    best_a = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _compute_bound_analytical(best_a[0], best_a[1], best_a[2])
    
    # Try multiple initial points for local optimization
    initial_points = [
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([-1.0, -1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([1.0, -1.0, 1.0]),
        best_a.copy(),
    ]
    
    # Also add random points with varied scales
    rng = np.random.default_rng(42)
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(10):
            initial_points.append(rng.uniform(-scale, scale, size=3))
    
    for x0 in initial_points:
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 20:
            break
        try:
            result = minimize(_objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_a = result.x
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    elapsed = time.time() - _start_time
    if elapsed < _TIME_LIMIT - 5:
        try:
            result = minimize(_objective, best_a, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_a = result.x
        except Exception:
            pass
    
    return best_a
# EVOLVE-BLOCK-END
