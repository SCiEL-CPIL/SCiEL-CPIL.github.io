"""Fourier-invariant Hermite-Gaussian test function tuned via full spectral moment computation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from math import factorial


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty bound C4 for the Hermite-Gaussian test function.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so g(0) = 0.
    
    The bound involves the second moments of |g|^2 and |F(g)|^2.
    """
    # Physicists' Hermite polynomials at x=0:
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -47880.0
    # g(0) = a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0 = 0
    # => a12 = -(a0 + a4*H4_0 + a8*H8_0) / H12_0
    a12 = -(a0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    coeffs = np.array([a0, a4, a8, a12])
    degrees = np.array([0, 4, 8, 12])
    
    # Norm: ||g||^2 = sum_i,j coeffs[i]*coeffs[j] * <H_i, H_j>
    # <H_i, H_j> = sqrt(pi) * 2^j * j! * delta_ij (by orthogonality)
    # So ||g||^2 = sqrt(pi) * sum_i coeffs[i]^2 * 2^i * i!
    norms = np.array([2.0**k * factorial(int(k)) for k in degrees])
    norm_sq = np.sum(coeffs**2 * norms)
    
    if norm_sq <= 0:
        return 1e10  # invalid
    
    # x^2 moment: <x^2> = <x^2 |g|^2> / ||g||^2
    # <x^2 H_i, H_j> = integral of x^2 H_i H_j exp(-x^2) dx
    # Using x^2 H_n = (1/4)H_{n+2} + (n+1/2)H_n + n(n-1)/2 H_{n-2}
    # So <x^2 H_i, H_j> = (1/4)*<H_{i+2}, H_j> + (i+1/2)*<H_i, H_j> + i(i-1)/2*<H_{i-2}, H_j>
    # = (1/4)*sqrt(pi)*2^j*j!*delta_{i+2,j} + (i+1/2)*sqrt(pi)*2^i*i!*delta_{i,j} + i(i-1)/2*sqrt(pi)*2^j*j!*delta_{i-2,j}
    
    # Build the x^2 moment matrix
    x2_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            val = 0.0
            # Term 1: (1/4) * <H_{i+2}, H_j>
            if i + 2 == j:
                val += 0.25 * 2.0**j * factorial(int(j))
            # Term 2: (i+1/2) * <H_i, H_j>
            if i == j:
                val += (i + 0.5) * 2.0**i * factorial(int(i))
            # Term 3: i(i-1)/2 * <H_{i-2}, H_j>
            if i - 2 == j and i >= 2:
                val += i * (i - 1) / 2.0 * 2.0**j * factorial(int(j))
            x2_matrix[i, j] = val
    
    x2_moment = np.dot(coeffs, x2_matrix @ coeffs) / norm_sq
    
    # For the Fourier side, F[H_n(x) exp(-x^2/2)] = i^n H_n(x) exp(-x^2/2)
    # So |F(g)|^2 has the same coefficient structure, meaning <xi^2> = <x^2>
    xi2_moment = x2_moment
    
    # The uncertainty bound C4 is related to sqrt(<x^2> * <xi^2>)
    # For the product A(f)*A(F), the bound involves this quantity
    bound = x2_moment * xi2_moment  # This is <x^2>^2 since they're equal
    
    return bound


def construct():
    start = time.time()
    deadline = start + 180.0  # safety margin before 210s kill
    
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best = seed.copy()
    best_bound = _compute_bound(*seed)
    
    # Try to improve via optimization
    try:
        def objective(x):
            if time.time() > deadline - 5.0:
                return 1e10
            return _compute_bound(x[0], x[1], x[2])
        
        # Use Nelder-Mead for derivative-free optimization
        result = minimize(objective, seed, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
        
        if result.success or result.fun < best_bound:
            if time.time() < deadline - 5.0:
                best = result.x
                best_bound = result.fun
    except Exception:
        pass
    
    # Additional refinement with different initial points if time allows
    if time.time() < deadline - 20.0:
        try:
            # Try a few different starting points
            rng = np.random.RandomState(42)
            for _ in range(5):
                if time.time() > deadline - 10.0:
                    break
                init = rng.uniform(-1, 1, size=3)
                result = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
                if result.fun < best_bound:
                    best = result.x
                    best_bound = result.fun
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
