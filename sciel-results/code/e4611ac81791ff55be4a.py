"""Spectral energy minimization for covering the sphere with 130 points."""
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 130

# EVOLVE-BLOCK-START
def _spherical_harmonic(l, m, theta, phi):
    """Compute spherical harmonic Y_l^m(theta, phi) using scipy.special."""
    from scipy.special import lpmv
    from scipy.constants import pi
    
    # lpmv gives associated Legendre function P_l^m(cos(theta))
    # Spherical harmonic: Y_l^m = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m(cos(theta)) * e^{im*phi}
    # We'll use a normalized version for numerical stability
    
    if m < 0:
        # Use relation Y_l^{-m} = (-1)^m conj(Y_l^m)
        return _spherical_harmonic(l, -m, theta, phi) * ((-1)**m) * np.exp(-1j * (-m) * phi) / np.exp(-1j * (-m) * phi) * np.exp(1j * m * phi)
    
    P_lm = lpmv(m, l, np.cos(theta))
    norm = np.sqrt((2*l + 1)/(4*np.pi) * np.math.factorial(l - m)/np.math.factorial(l + m))
    return norm * P_lm * np.exp(1j * m * phi)


def _pair_correlation_energy(points, max_l=20):
    """
    Compute a spectral energy based on the variance of the pair correlation function.
    The pair correlation function g(r) measures how uniformly points are distributed.
    We use the spectral representation to compute an energy that promotes uniformity.
    
    For a set of N points on the sphere, the power spectrum of the point density
    at multipole l is related to the sum over pairs of P_l(cos(theta_ij)).
    A uniform distribution has zero power for l >= 1.
    We minimize the weighted sum of powers for l >= 1.
    """
    # Compute pairwise angular distances
    # Use dot product for efficiency
    dots = points @ points.T
    np.clip(dots, -1.0, 1.0, out=dots)
    
    energy = 0.0
    for l in range(1, max_l + 1):
        # Compute P_l(cos(theta_ij)) for all pairs
        P_l = np.polynomial.legendre.legvander(dots, l)[:, -1]
        # Sum over all pairs (i, j) with i != j
        # The diagonal terms are P_l(1) = 1, there are N of them
        pair_sum = np.sum(P_l) - N  # exclude diagonal
        # The power at multipole l is proportional to |sum_{i,j} Y_l^m*(x_i) Y_l^m(x_j)|^2 summed over m
        # For real points, this simplifies. A good proxy is pair_sum^2 / (N^2 - N)
        # Weight by 1/l to give more weight to low-order modes (large-scale non-uniformity)
        weight = 1.0 / l
        energy += weight * (pair_sum / (N - 1))**2
    
    return energy


def _refine_from_spectral(start_points, max_iters=500, tol=1e-10):
    """
    Refine points using spectral energy minimization.
    We parameterize points by (theta, phi) and use gradient-free optimization.
    """
    # Convert to (theta, phi)
    theta = np.arccos(np.clip(start_points[:, 2], -1.0, 1.0))
    phi = np.arctan2(start_points[:, 1], start_points[:, 0])
    
    # Flatten to 1D array of 2N parameters
    x0 = np.concatenate([theta, phi])
    
    def obj(x):
        t = x[:N]
        p = x[N:]
        # Wrap phi to [0, 2pi)
        p = np.mod(p, 2*np.pi)
        # Convert back to Cartesian
        pts = np.column_stack([
            np.sin(t) * np.cos(p),
            np.sin(t) * np.sin(p),
            np.cos(t)
        ])
        return _pair_correlation_energy(pts, max_l=15)
    
    # Use Nelder-Mead for local refinement
    result = minimize(obj, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iters, 'xatol': tol, 'fatol': tol})
    
    t = result.x[:N]
    p = np.mod(result.x[N:], 2*np.pi)
    pts = np.column_stack([
        np.sin(t) * np.cos(p),
        np.sin(t) * np.sin(p),
        np.cos(t)
    ])
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts, result.fun


def construct():
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    start_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine using spectral energy minimization
    refined_pts, energy = _refine_from_spectral(start_pts, max_iters=300)
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(refined_pts, axis=1, keepdims=True)
    refined_pts = refined_pts / norms
    
    return refined_pts
# EVOLVE-BLOCK-END
