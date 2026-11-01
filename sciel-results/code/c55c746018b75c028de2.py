"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize, L_BFGS_B
from scipy.special import logsumexp
import itertools


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    # Compute all pairwise distances
    # Use broadcasting: pts[i] - pts[j] for i != j
    # To save memory, we can compute in chunks or use a smarter approach
    # For N=150, N^2 = 22500 which is fine
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # We want sum over i<j of log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    return -np.log(dist[iu]).sum()


def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Gradient of -log(d_ij) w.r.t. x_i is (x_i - x_j) / d_ij^2
    # For the full sum, gradient w.r.t. x_i is sum_{j != i} (x_i - x_j) / d_ij^2
    # But we only sum over i<j, so we need to be careful.
    
    # Let's compute the full pairwise gradient and then adjust.
    # E = sum_{i<j} -log(d_ij)
    # dE/dx_i = sum_{j>i} (x_i - x_j)/d_ij^2 + sum_{j<i} (-(x_j - x_i))/d_ij^2
    #         = sum_{j>i} (x_i - x_j)/d_ij^2 + sum_{j<i} (x_i - x_j)/d_ij^2
    #         = sum_{j != i} (x_i - x_j)/d_ij^2
    
    # Wait, let's re-derive.
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j>i} d/dx_i log(d_ij) - sum_{j<i} d/dx_i log(d_ji)
    # d/dx_i log(d_ij) = (1/d_ij) * d/dx_i d_ij = (1/d_ij) * (x_i - x_j)/d_ij = (x_i - x_j)/d_ij^2
    # So dE/dx_i = -sum_{j>i} (x_i - x_j)/d_ij^2 - sum_{j<i} (x_i - x_j)/d_ji^2
    # Note d_ji = d_ij, so:
    # dE/dx_i = -sum_{j>i} (x_i - x_j)/d_ij^2 - sum_{j<i} (x_i - x_j)/d_ij^2
    #         = -sum_{j != i} (x_i - x_j)/d_ij^2
    #         = sum_{j != i} (x_j - x_i)/d_ij^2
    
    # So the gradient is the sum over all j != i of (x_j - x_i) / d_ij^2
    # This is exactly the force in a Coulomb-like potential.
    
    # Compute using matrix operations
    # diff[i,j] = x_i - x_j
    # We want sum_j (x_j - x_i) / d_ij^2 = -sum_j (x_i - x_j) / d_ij^2 = -sum_j diff[i,j] / dist_sq[i,j]
    
    # Create a matrix M where M[i,j] = diff[i,j] / dist_sq[i,j]
    # But we need to handle the diagonal (set to 0)
    
    # dist_sq is (n, n)
    # diff is (n, n, 3)
    
    # To avoid division by zero on diagonal, we already set diagonal of dist_sq to 1.0
    # But we want the diagonal contribution to be 0, so we set the diagonal of the result to 0.
    
    # Let's compute: grad = -sum_j (x_i - x_j) / d_ij^2
    # In matrix form: for each i, grad[i] = -sum_j (diff[i,j] / dist_sq[i,j])
    
    # We can do this with broadcasting
    # inv_dist_sq = 1.0 / dist_sq  (n, n)
    # Set diagonal to 0
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Now grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff is (n, n, 3), inv_dist_sq is (n, n)
    # We need to multiply diff[i,j] by inv_dist_sq[i,j] and sum over j
    # This is like a weighted sum: grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return grad


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 145s out of 150s
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = np.inf
    
    # Strategy:
    # 1. Start with Fibonacci spiral
    # 2. Do L-BFGS-B optimization on the sphere using parameterization
    #    We use spherical coordinates or quaternion-free approach:
    #    Actually, L-BFGS-B works on R^3, so we need to project back to sphere.
    #    Better: use a parameterization that maps R^3 -> S^2, e.g., stereographic projection
    #    or just optimize in R^3 with projection at each step.
    #    
    #    Alternative: use scipy.optimize.minimize with a custom approach where
    #    we optimize in R^3 and normalize after each gradient step.
    #    This is not exactly on the manifold but works in practice.
    #
    #    Better approach: Use the fact that the gradient we computed is already
    #    the gradient in R^3. We can do projected gradient descent:
    #    x_{k+1} = normalize(x_k - alpha * grad(x_k))
    #
    #    But L-BFGS-B doesn't support this directly. Let's use a different approach:
    #    Parameterize each point by 2 angles (theta, phi) and optimize in R^2 per point.
    #    Total variables: 2 * N = 300.
    
    # Let's use spherical coordinates: x = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    # theta in [0, pi], phi in [0, 2*pi)
    
    def _theta_phi_to_xyz(theta_phi):
        """Convert (theta, phi) pairs to 3D points on sphere."""
        theta = theta_phi[0::2]
        phi = theta_phi[1::2]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    def _energy_from_theta_phi(theta_phi):
        pts = _theta_phi_to_xyz(theta_phi)
        return _log_energy(pts)
    
    def _grad_from_theta_phi(theta_phi):
        pts = _theta_phi_to_xyz(theta_phi)
        grad_xyz = _log_energy_grad(pts)
        # Chain rule: dE/dtheta = grad_xyz . dpts/dtheta
        # dpts/dtheta = (cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta))
        theta = theta_phi[0::2]
        phi = theta_phi[1::2]
        
        dpts_dtheta = np.column_stack([
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta)
        ])
        dpts_dphi = np.column_stack([
            -np.sin(theta) * np.sin(phi),
            np.sin(theta) * np.cos(phi),
            np.zeros(len(theta))
        ])
        
        grad_theta = np.sum(grad_xyz * dpts_dtheta, axis=1)
        grad_phi = np.sum(grad_xyz * dpts_dphi, axis=1)
        
        # Interleave
        grad = np.empty(2 * len(theta))
        grad[0::2] = grad_theta
        grad[1::2] = grad_phi
        return grad
    
    # Start with Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    
    # Convert to theta, phi
    theta0 = np.arccos(np.clip(pts0[:, 2], -1.0, 1.0))
    phi0 = np.arctan2(pts0[:, 1], pts0[:, 0])
    
    x0 = np.empty(2 * N)
    x0[0::2] = theta0
    x0[1::2] = phi0
    
    # Optimize with L-BFGS-B
    # We need to handle the periodicity of phi. L-BFGS-B doesn't handle this well.
    # Let's try it and see.
    
    result = minimize(
        _energy_from_theta_phi,
        x0,
        jac=_grad_from_theta_phi,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
    )
    
    if result.success or result.fun < best_energy:
        best_pts = _theta_phi_to_xyz(result.x)
        best_energy = result.fun
    
    # Try multiple random restarts if time permits
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        pts_r = _random_sphere(N, rng)
        theta_r = np.arccos(np.clip(pts_r[:, 2], -1.0, 1.0))
        phi_r = np.arctan2(pts_r[:, 1], pts_r[:, 0])
        
        x_r = np.empty(2 * N)
        x_r[0::2] = theta_r
        x_r[1::2] = phi_r
        
        result_r = minimize(
            _energy_from_theta_phi,
            x_r,
            jac=_grad_from_theta_phi,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        
        if result_r.fun < best_energy:
            best_energy = result_r.fun
            best_pts = _theta_phi_to_xyz(result_r.x)
    
    # Final polish: do a few more iterations with tighter tolerance
    if best_pts is not None and time.time() < deadline - 10.0:
        theta_final = np.arccos(np.clip(best_pts[:, 2], -1.0, 1.0))
        phi_final = np.arctan2(best_pts[:, 1], best_pts[:, 0])
        
        x_final = np.empty(2 * N)
        x_final[0::2] = theta_final
        x_final[1::2] = phi_final
        
        result_final = minimize(
            _energy_from_theta_phi,
            x_final,
            jac=_grad_from_theta_phi,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 20000}
        )
        
        if result_final.fun < best_energy:
            best_energy = result_final.fun
            best_pts = _theta_phi_to_xyz(result_final.x)
    
    # Ensure best_pts is valid
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Normalize to ensure unit sphere
    best_pts = _normalize(best_pts)
    
    # Validate: no points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
