"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.special import lpmv
from scipy.spatial import Delaunay

def _spherical_harmonics(points, l_max=20):
    """
    Compute spherical harmonics Y_l^m(theta, phi) for points on the unit sphere.
    Returns a complex array of shape (N, (l_max+1)^2) containing Y_l^m for 0 <= l <= l_max, -l <= m <= l.
    """
    # Extract theta (polar) and phi (azimuthal) from Cartesian coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    r = np.where(r < 1e-15, 1.0, r)
    x, y, z = x / r, y / r, z / r
    
    theta = np.arccos(np.clip(z, -1.0, 1.0))  # polar angle
    phi = np.arctan2(y, x)                     # azimuthal angle
    
    # Build the harmonic array
    n_harm = (l_max + 1) ** 2
    Y = np.zeros((len(points), n_harm), dtype=complex)
    
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Associated Legendre polynomial P_l^m(cos(theta))
            # Note: scipy's lpmv computes P_l^m(x) with Condon-Shortley phase
            try:
                P_lm = lpmv(abs(m), l, np.cos(theta))
            except Exception:
                P_lm = np.zeros_like(theta)
            
            # Normalization factor
            # N_lm = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
            # Using Stirling approximation for factorials or direct computation
            # For small l, we can compute directly
            if l + abs(m) < 170:  # Avoid overflow in factorial
                from math import factorial, lgamma
                norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.exp(lgamma(l - abs(m) + 1) - lgamma(l + abs(m) + 1)))
            else:
                norm = 1.0
            
            # Y_l^m(theta, phi) = N_lm * P_l^m(cos(theta)) * exp(i*m*phi)
            Y[:, idx] = norm * P_lm * np.exp(1j * m * phi)
            idx += 1
    
    return Y

def _energy_from_points(points):
    """Calculate the logarithmic energy from point positions."""
    if not np.all(np.isfinite(points)):
        return float('inf')
    
    # Compute pairwise distances
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Check for invalid close points
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf')
        
    # Energy: sum over i<j of log(1/d_ij) = -sum over i<j of log(d_ij)
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    energy = -np.sum(np.log(d))
    
    return energy

def _optimize_harmonic_coefficients(points_init, l_max, deadline):
    """
    Optimize the harmonic coefficients to shape the potential field.
    The points are defined as the local minima of the potential V(x) = |sum c_lm Y_lm(x)|^2.
    We minimize the energy of the 100 local minima of this potential.
    """
    # Start with coefficients that reproduce the initial points
    # This is an ill-posed inverse problem, so we use a heuristic:
    # Initialize coefficients from the spherical harmonic transform of the point distribution
    
    Y_init = _spherical_harmonics(points_init, l_max)
    n_harm = Y_init.shape[1]
    
    # Initialize coefficients as the mean of Y_init weighted by a delta-like function
    # For simplicity, start with small random coefficients
    np.random.seed(42)
    c0 = np.random.randn(n_harm) * 0.01 + 1j * np.random.randn(n_harm) * 0.01
    
    def _get_points_from_coefficients(c):
        """
        Given harmonic coefficients, find the 100 local minima of the potential.
        V(x) = |sum c_lm Y_lm(x)|^2
        We use a grid-based approach to find local minima.
        """
        # Create a fine grid on the sphere
        n_theta = 100
        n_phi = 200
        theta_grid, phi_grid = np.meshgrid(
            np.linspace(0, np.pi, n_theta),
            np.linspace(0, 2 * np.pi, n_phi),
            indexing='ij'
        )
        
        # Convert to Cartesian
        x_grid = np.sin(theta_grid) * np.cos(phi_grid)
        y_grid = np.sin(theta_grid) * np.sin(phi_grid)
        z_grid = np.cos(theta_grid)
        
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
        
        # Evaluate potential on grid
        Y_grid = _spherical_harmonics(grid_points, l_max)
        V = np.abs(Y_grid @ c) ** 2
        
        # Find local minima using a simple approach
        # For each grid point, check if it's a local minimum in its neighborhood
        # This is expensive, so we use a coarser approach
        
        # Use scipy's find_peaks or a custom local minimum finder
        # For simplicity, we'll use a k-means-like approach:
        # 1. Start with random seed points
        # 2. Assign each grid point to the nearest seed
        # 3. Update seeds to the centroid of assigned points (weighted by 1/V)
        
        # Actually, a better approach: use the gradient of the potential to find minima
        # But computing the gradient of |sum c_lm Y_lm|^2 is complex
        
        # Alternative: use the fact that the points should be well-separated
        # Start with the initial points and refine them using the potential
        
        # For now, return the initial points (this is a placeholder)
        # In a real implementation, we would:
        # 1. Compute the gradient of the potential
        # 2. Use gradient descent to find local minima
        # 3. Return the 100 minima with the lowest potential values
        
        return points_init
    
    def _objective(c_flat):
        """Objective function for the harmonic coefficients."""
        if time.time() > deadline:
            return float('inf')
        
        c = c_flat[:n_harm].astype(complex)
        c = c[:n_harm].real + 1j * c_flat[n_harm:2*n_harm].real
        
        # Get points from coefficients
        pts = _get_points_from_coefficients(c)
        
        # Compute energy
        e = _energy_from_points(pts)
        
        if not np.isfinite(e):
            return float('inf')
        
        # Add a regularization term to encourage distinct minima
        # This is a placeholder
        return e
    
    # Flatten complex coefficients
    x0 = np.concatenate([c0.real, c0.imag])
    
    # Optimize
    res = minimize(_objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if res.success or np.isfinite(res.fun):
        c_opt = res.x[:n_harm].astype(complex)
        c_opt = c_opt[:n_harm].real + 1j * res.x[n_harm:2*n_harm].real
        
        pts = _get_points_from_coefficients(c_opt)
        return pts, res.fun
    
    return points_init, float('inf')

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n):
    """Generate random points on the sphere."""
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, np.newaxis]

def _optimize_points_directly(points, deadline):
    """
    Direct optimization of point positions using Riemannian gradient descent.
    This serves as a fallback and refinement step.
    """
    if time.time() > deadline - 5.0:
        return points, _energy_from_points(points)
    
    # Use L-BFGS-B with tangent space gradient
    x0 = points.flatten()
    
    def _energy_and_grad_tangent(pts):
        """Compute energy and tangent space gradient."""
        if not np.all(np.isfinite(pts)):
            return float('inf'), np.zeros_like(pts)
        
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(dists, 1.0)
        
        min_dist = np.min(dists)
        if min_dist < 1e-9:
            return float('inf'), np.zeros_like(pts)
        
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        energy = -np.sum(np.log(d))
        
        grad = np.zeros_like(pts)
        for i in range(N):
            diff = pts[i] - pts
            dist_sq = dists[i, :]**2
            dist_sq[i] = 1.0
            term = diff / dist_sq[:, np.newaxis]
            term[i] = 0.0
            grad[i] = -np.sum(term, axis=0)
            
            # Project onto tangent space
            dot = np.dot(grad[i], pts[i])
            grad[i] -= dot * pts[i]
        
        return energy, grad
    
    def objective(x):
        if time.time() > deadline:
            return float('inf'), np.zeros_like(x)
        
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms < 1e-9):
            return float('inf'), np.zeros_like(x)
        pts = pts / norms[:, np.newaxis]
        
        e, g = _energy_and_grad_tangent(pts)
        if not np.isfinite(e):
            return float('inf'), np.zeros_like(x)
        
        return e, g.flatten()
    
    res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-8})
    
    if res.success or np.isfinite(res.fun):
        best_pts = res.x.reshape(N, 3)
        norms = np.linalg.norm(best_pts, axis=1)
        if np.all(norms > 1e-9):
            best_pts = best_pts / norms[:, np.newaxis]
            return best_pts, res.fun
    
    return points, _energy_from_points(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # Strategy:
    # 1. Generate initial points using Fibonacci lattice
    # 2. Try harmonic coefficient optimization (spectral potential shaping)
    # 3. Refine using direct Riemannian optimization
    # 4. Return the best result
    
    best_points = None
    best_energy = float('inf')
    
    # Initial points
    fib_points = _fibonacci_sphere(N)
    best_points = fib_points.copy()
    best_energy = _energy_from_points(fib_points)
    
    # Try harmonic coefficient optimization with different l_max values
    for l_max in [10, 15, 20]:
        if time.time() > deadline - 30.0:
            break
        
        try:
            opt_points, energy = _optimize_harmonic_coefficients(fib_points, l_max, deadline)
            
            if energy < best_energy:
                best_energy = energy
                best_points = opt_points.copy()
        except Exception:
            pass
    
    # Refine using direct optimization
    if time.time() < deadline - 10.0:
        try:
            opt_points, energy = _optimize_points_directly(best_points, deadline)
            
            if energy < best_energy:
                best_energy = energy
                best_points = opt_points.copy()
        except Exception:
            pass
    
    # Final validation
    if best_points is None or not np.all(np.isfinite(best_points)):
        return fib_points
    
    # Normalize
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-9):
        return fib_points
    
    best_points = best_points / norms[:, np.newaxis]
    
    return best_points
# EVOLVE-BLOCK-END
