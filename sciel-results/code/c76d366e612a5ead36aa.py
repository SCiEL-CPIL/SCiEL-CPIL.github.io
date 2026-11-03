"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
from math import sqrt, pi

# Precompute constants for analytic moment computation
# Degrees present: 0, 4, 8, 12
_DEGS = [0, 4, 8, 12]

# H_n(0) for even n: H_{2k}(0) = (-1)^k * (2k)! / k!
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

# Norm squared of H_n under e^{-x^2} weight: sqrt(pi) * 2^n * n!
def _norm_sq(n):
    # n! for n in [0, 4, 8, 12]
    # 0! = 1
    # 4! = 24
    # 8! = 40320
    # 12! = 479001600
    fact = {0: 1, 4: 24, 8: 40320, 12: 479001600}
    return sqrt(pi) * (2.0 ** n) * fact[n]

# We need to compute var_x and var_xi.
# g(x) = P(x) e^{-x^2/2}
# g_hat(x) = P_hat(x) e^{-x^2/2}
# where P(x) = sum a_k H_k(x)
# P_hat(x) = sum a_k i^k H_k(x) (assuming unitary FT convention where F[H_n e^{-x^2/2}] = i^n H_n e^{-x^2/2})
# Note: The standard physicist's Hermite polynomial H_n satisfies H_n(x) = (-1)^n e^{x^2/2} d^n/dx^n e^{-x^2/2}.
# The Fourier transform of H_n(x) e^{-x^2/2} depends on the convention.
# With F(k) = 1/sqrt(2pi) integral f(x) e^{-ikx} dx, F[H_n(x) e^{-x^2/2}] = i^n H_n(k) e^{-k^2/2}.
# So P_hat(x) = a0 H0 + a4 i^4 H4 + a8 i^8 H8 + a12 i^12 H12
# i^0 = 1, i^4 = 1, i^8 = 1, i^12 = -1
# So P_hat(x) = a0 H0 + a4 H4 + a8 H8 - a12 H12

def _compute_bound_analytic(a0, a4, a8):
    """Compute the upper bound C4 on A(f)*A(Fourier(f)) analytically.
    
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 chosen so g(0) = 0.
    """
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    
    coeffs = [a0, a4, a8, a12]
    
    # We need integrals of the form:
    # I_{nm} = integral H_n(x) H_m(x) e^{-x^2} dx = sqrt(pi) 2^n n! delta_nm
    # J_{nm} = integral x^2 H_n(x) H_m(x) e^{-x^2} dx
    
    # Var_x = (integral x^2 g^2 dx) / (integral g^2 dx) - (integral x g^2 dx)^2 / (integral g^2 dx)^2
    # Since g^2 is even, integral x g^2 dx = 0.
    # So Var_x = (integral x^2 g^2 dx) / (integral g^2 dx)
    
    # Similarly for Fourier side, but with P_hat.
    
    # Let's compute the quadratic forms.
    # g^2 = P^2 e^{-x^2}
    # integral g^2 dx = sum_i sum_j a_i a_j integral H_i H_j e^{-x^2} dx = sum_i a_i^2 NormSq_i
    
    norm_sq = 0.0
    x2_integral = 0.0
    
    for i, n_i in enumerate(_DEGS):
        c_i = coeffs[i]
        if c_i == 0: continue
        ns_i = _norm_sq(n_i)
        norm_sq += c_i * c_i * ns_i
        
        # integral x^2 H_n^2 e^{-x^2} dx
        # There is a formula for this.
        # x H_n = 1/2 H_{n+1} + n H_{n-1}
        # x^2 H_n = 1/4 H_{n+2} + n H_n + n(n-1)/4 H_{n-2} ? No.
        # Let's use the recurrence:
        # x H_n = 1/2 H_{n+1} + n H_{n-1}
        # x^2 H_n = x (1/2 H_{n+1} + n H_{n-1})
        # = 1/2 (1/2 H_{n+2} + (n+1) H_n) + n (1/2 H_n + (n-1) H_{n-2})
        # = 1/4 H_{n+2} + (n+1)/2 H_n + n/2 H_n + n(n-1)/2 H_{n-2}
        # = 1/4 H_{n+2} + (2n+1)/2 H_n + n(n-1)/2 H_{n-2}
        
        # So x^2 H_n = 0.25 H_{n+2} + (n+0.5) H_n + 0.5*n*(n-1) H_{n-2}
        
        # Integral x^2 H_n^2 e^{-x^2} dx = <H_n, x^2 H_n> = <H_n, 0.25 H_{n+2} + (n+0.5) H_n + 0.5*n*(n-1) H_{n-2}>
        # By orthogonality, only the H_n term survives.
        # = (n + 0.5) * NormSq_n
        
        x2_integral += c_i * c_i * (n_i + 0.5) * ns_i
    
    if norm_sq < 1e-15:
        return 1e10
        
    var_x = x2_integral / norm_sq
    
    # Now for Fourier side.
    # P_hat = a0 H0 + a4 H4 + a8 H8 - a12 H12
    # coeffs_hat = [a0, a4, a8, -a12]
    coeffs_hat = [coeffs[0], coeffs[1], coeffs[2], -coeffs[3]]
    
    norm_sq_hat = 0.0
    x2_integral_hat = 0.0
    
    for i, n_i in enumerate(_DEGS):
        c_i = coeffs_hat[i]
        if c_i == 0: continue
        ns_i = _norm_sq(n_i)
        norm_sq_hat += c_i * c_i * ns_i
        x2_integral_hat += c_i * c_i * (n_i + 0.5) * ns_i
        
    if norm_sq_hat < 1e-15:
        return 1e10
        
    var_xi = x2_integral_hat / norm_sq_hat
    
    bound = 2.0 * sqrt(max(var_x, 0.0) * max(var_xi, 0.0))
    return bound

def _objective_sphere(theta, phi):
    """Objective function on the sphere.
    Since the objective is homogeneous, we can fix the scale.
    Let a0 = sin(phi) cos(theta), a4 = sin(phi) sin(theta), a8 = cos(phi).
    This covers the entire space up to scale.
    """
    a0 = np.sin(phi) * np.cos(theta)
    a4 = np.sin(phi) * np.sin(theta)
    a8 = np.cos(phi)
    return _compute_bound_analytic(a0, a4, a8)

def construct():
    """Optimize the coefficients to minimize the upper bound C4."""
    start_time = time.time()
    deadline = start_time + 190  # Safety margin for 210s limit
    
    # Define bounds for DE
    # theta in [0, 2pi], phi in [-pi/2, pi/2] (or 0 to pi)
    bounds = [(0, 2*np.pi), (-np.pi/2, np.pi/2)]
    
    best_bound = 1e10
    best_x = np.array([0.0, 0.0, 0.0])
    
    # We can also try a few random points first to get a good initial population or just let DE do its thing
    # DE is stochastic, so set seed for reproducibility if needed, but task doesn't require it.
    
    try:
        result = differential_evolution(
            _objective_sphere,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-10,
            popsize=20,
            init='sobol'
        )
        
        if result.fun < best_bound:
            best_bound = result.fun
            theta, phi = result.x
            a0 = np.sin(phi) * np.cos(theta)
            a4 = np.sin(phi) * np.sin(theta)
            a8 = np.cos(phi)
            best_x = np.array([a0, a4, a8])
            
    except Exception as e:
        # Fallback to a simple grid search or random search if DE fails
        pass
    
    # If DE didn't find anything or we have time, do a local polish with Nelder-Mead on the 3D coefficients
    # Or just return the best found.
    # The analytic bound is very fast, so we can do more iterations if time permits.
    
    if time.time() < deadline - 5:
        # Try a few more DE runs with different seeds or pop sizes if time allows
        for seed in [1, 2, 3]:
            if time.time() >= deadline - 5:
                break
            try:
                result = differential_evolution(
                    _objective_sphere,
                    bounds,
                    seed=seed,
                    maxiter=50,
                    tol=1e-10,
                    popsize=15
                )
                if result.fun < best_bound:
                    best_bound = result.fun
                    theta, phi = result.x
                    a0 = np.sin(phi) * np.cos(theta)
                    a4 = np.sin(phi) * np.sin(theta)
                    a8 = np.cos(phi)
                    best_x = np.array([a0, a4, a8])
            except Exception:
                pass

    return best_x
# EVOLVE-BLOCK-END
