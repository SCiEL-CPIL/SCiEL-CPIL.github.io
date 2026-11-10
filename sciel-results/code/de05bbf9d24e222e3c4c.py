"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math


def _log_energy(pts):
    """Compute log energy: sum over i<j of log(1/d_ij) = -sum log(d_ij)."""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e10
            total += -math.log(d)
    return total


def _log_energy_vec(pts):
    """Vectorized log energy computation."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only upper triangle, exclude diagonal
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.any(d < 1e-15):
        return 1e10
    return float(np.sum(-np.log(d)))


def _log_energy_grad(pts):
    """Gradient of log energy with respect to each point.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                continue
            grad[i] += d / (dist * dist)
    return grad


def _log_energy_and_grad(pts):
    """Return (energy, gradient) for log energy."""
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                return 1e10, grad
            # E += -log(dist)
            energy += -math.log(dist)
            # grad[i] += (x_i - x_j)/dist^2
            # grad[j] += (x_j - x_i)/dist^2 = -(x_i - x_j)/dist^2
            g = d / (dist * dist)
            grad[i] += g
            grad[j] -= g
    return energy, grad


def _spherical_to_cartesian(theta, phi):
    """Convert spherical coords (theta: polar angle from z-axis, phi: azimuthal) to Cartesian."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def _cartesian_to_spherical(pts):
    """Convert Cartesian points on unit sphere to (theta, phi)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return np.column_stack([theta, phi])


def _optimize_from_spherical(sph, max_time=100.0):
    """Optimize starting from spherical coordinates using L-BFGS-B."""
    start_time = time.time()
    
    def objective_and_grad(sph_flat):
        if time.time() - start_time > max_time:
            return 1e10, np.zeros_like(sph_flat)
        theta = sph_flat[0::2]
        phi = sph_flat[1::2]
        pts = _spherical_to_cartesian(theta, phi)
        # Project back to sphere to ensure validity
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        energy, cart_grad = _log_energy_and_grad(pts)
        if energy > 1e9:
            return energy, np.zeros_like(sph_flat)
        # Transform gradient from Cartesian to spherical
        # x = sin(theta)cos(phi), y = sin(theta)sin(phi), z = cos(theta)
        # dx/dtheta = cos(theta)cos(phi), dy/dtheta = cos(theta)sin(phi), dz/dtheta = -sin(theta)
        # dx/dphi = -sin(theta)sin(phi), dy/dphi = sin(theta)cos(phi), dz/dphi = 0
        grad_sph = np.zeros_like(sph_flat)
        for i in range(N):
            gx, gy, gz = cart_grad[i]
            t, p = theta[i], phi[i]
            # dE/dtheta = gx*cos(t)*cos(p) + gy*cos(t)*sin(p) + gz*(-sin(t))
            grad_sph[2*i] = gx * math.cos(t) * math.cos(p) + gy * math.cos(t) * math.sin(p) - gz * math.sin(t)
            # dE/dphi = gx*(-sin(t)*sin(p)) + gy*(sin(t)*cos(p))
            grad_sph[2*i+1] = -gx * math.sin(t) * math.sin(p) + gy * math.sin(t) * math.cos(p)
        return energy, grad_sph
    
    # L-BFGS-B with bounds on theta [0, pi] and phi [0, 2*pi]
    bounds = [(0.0, math.pi), (0.0, 2*math.pi)] * N
    
    result = minimize(
        objective_and_grad,
        sph_flat=sph.ravel(),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 50000}
    )
    
    if result.success or result.fun < 1e9:
        theta = result.x[0::2]
        phi = result.x[1::2]
        pts = _spherical_to_cartesian(theta, phi)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        return pts, float(result.fun)
    return None, 1e10


def _fibonacci_sphere(n, offset=0.5):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=42):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _perturb_sphere(pts, sigma=0.1, seed=0):
    """Add Gaussian perturbation and re-normalize."""
    rng = np.random.RandomState(seed)
    perturbed = pts + rng.randn(*pts.shape) * sigma
    perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
    return perturbed


def construct():
    deadline = 140.0  # Safety margin before 150s kill
    start = time.time()
    
    best_pts = None
    best_energy = 1e10
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Fibonacci spiral
    initial_configs.append(_fibonacci_sphere(N, 0.5))
    
    # 2. Fibonacci with different offsets
    initial_configs.append(_fibonacci_sphere(N, 0.3))
    initial_configs.append(_fibonacci_sphere(N, 0.7))
    
    # 3. Random starts
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        initial_configs.append(_random_sphere(N, seed))
    
    # 4. Perturbed Fibonacci
    base = _fibonacci_sphere(N, 0.5)
    for sigma in [0.05, 0.1, 0.2, 0.3]:
        for seed in [100, 101, 102]:
            initial_configs.append(_perturb_sphere(base, sigma, seed))
    
    # Evaluate and optimize from each config
    for idx, pts in enumerate(initial_configs):
        if time.time() - start > deadline - 10:
            break
        
        # First evaluate the initial energy
        e0, _ = _log_energy_and_grad(pts)
        if e0 < best_energy:
            best_energy = e0
            best_pts = pts.copy()
        
        # Convert to spherical and optimize
        sph = _cartesian_to_spherical(pts)
        
        remaining = deadline - 10 - (time.time() - start)
        if remaining < 5:
            break
        
        opt_pts, opt_energy = _optimize_from_spherical(sph, max_time=min(remaining, 12.0))
        
        if opt_pts is not None and opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
        
        # If we have time, try a second round of optimization from the result
        if time.time() - start < deadline - 15:
            sph2 = _cartesian_to_spherical(opt_pts)
            remaining2 = deadline - 15 - (time.time() - start)
            if remaining2 > 5:
                opt_pts2, opt_energy2 = _optimize_from_spherical(sph2, max_time=min(remaining2, 10.0))
                if opt_pts2 is not None and opt_energy2 < best_energy:
                    best_energy = opt_energy2
                    best_pts = opt_pts2.copy()
    
    # Final fallback: if no valid result, return Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, 0.5)
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts

# EVOLVE-BLOCK-END
