"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def _spherical_to_cartesian(theta, phi):
    sin_t = np.sin(theta)
    return np.column_stack([
        sin_t * np.cos(phi),
        sin_t * np.sin(phi),
        np.cos(theta)
    ])

def _cartesian_to_spherical(xyz):
    # theta is polar angle (0 to pi), phi is azimuthal (0 to 2pi)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    # Ensure phi is in [0, 2pi)
    phi[phi < 0] += 2 * np.pi
    return theta, phi

def _compute_energy_and_grad(theta_phi):
    n = len(theta_phi) // 2
    theta = theta_phi[0::2]
    phi = theta_phi[1::2]
    
    xyz = _spherical_to_cartesian(theta, phi)
    
    # Compute pairwise distances
    # To save memory and time for N=150, we can do it in chunks or just vectorize
    # N=150 is small enough for full NxN matrix
    diff = xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero (should not happen if points are distinct)
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij)
    # We only need upper triangle
    iu = np.triu_indices(n, k=1)
    d_pairs = dist[iu]
    energy = np.sum(np.log(1.0 / d_pairs))
    
    # Gradient calculation
    # grad_x[k] = - sum_{j != k} (x_k - x_j) / dist_kj^2
    # Note: dist_sq is symmetric.
    # We can compute the force on each point.
    # F_k = - sum_{j != k} (x_k - x_j) / dist_kj^2
    # Let's compute the sum of (x_k - x_j) / dist_kj^2 for all j != k.
    
    # dist_sq is NxN. We want to divide each element (i,j) by dist_sq(i,j)^2? No.
    # The term is (x_k - x_j) / dist_kj^2.
    # So we need to scale the difference vectors.
    
    # Create a mask for i != j
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    # dist_sq_inv_sq = 1 / dist_sq^2
    dist_sq_inv_sq = 1.0 / (dist_sq * dist_sq)
    dist_sq_inv_sq[~mask] = 0.0 # Zero out diagonals
    
    # Weighted differences
    # diff[i,j] = x_i - x_j
    # We want sum_j (x_i - x_j) / dist_ij^2
    # This is sum_j diff[i,j] * dist_sq_inv_sq[i,j]
    # Broadcasting: diff is (N, N, 3), dist_sq_inv_sq is (N, N)
    weighted_diff = diff * dist_sq_inv_sq[:, :, np.newaxis]
    
    # Sum over j (axis 1) to get force on i
    # grad_x[i] = - sum_j weighted_diff[i,j]
    grad_x = -np.sum(weighted_diff, axis=1)
    
    # Now project grad_x to theta, phi gradients
    # dE/dtheta_i = grad_x[i] . d(x_i)/d(theta_i)
    # dE/dphi_i = grad_x[i] . d(x_i)/d(phi_i)
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    # d(x_i)/d(theta_i) = (cos_t cos_p, cos_t sin_p, -sin_t)
    d_x_d_theta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
    
    # d(x_i)/d(phi_i) = (-sin_t sin_p, sin_t cos_p, 0)
    d_x_d_phi = np.column_stack([-sin_t * sin_p, sin_t * cos_p, np.zeros(n)])
    
    grad_theta = np.sum(grad_x * d_x_d_theta, axis=1)
    grad_phi = np.sum(grad_x * d_x_d_phi, axis=1)
    
    # Interleave gradients
    grad = np.empty_like(theta_phi)
    grad[0::2] = grad_theta
    grad[1::2] = grad_phi
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial guess: Fibonacci spiral
    xyz_init = _fibonacci_points(N)
    theta_init, phi_init = _cartesian_to_spherical(xyz_init)
    
    # Flatten for optimizer
    x0 = np.empty(2 * N)
    x0[0::2] = theta_init
    x0[1::2] = phi_init
    
    best_energy = float('inf')
    best_x = x0.copy()
    
    # Run L-BFGS-B
    # We might want to try multiple restarts if time permits, but one good run is often enough
    # Let's try to run it with a tight tolerance
    
    try:
        res = minimize(
            _compute_energy_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if res.success:
            if res.fun < best_energy:
                best_energy = res.fun
                best_x = res.x
    except Exception:
        pass
    
    # If time remains, try a second restart with random initialization
    if time.time() < deadline:
        np.random.seed(42)
        # Random points on sphere
        vecs = np.random.normal(size=(N, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        theta_r, phi_r = _cartesian_to_spherical(vecs)
        x_r = np.empty(2 * N)
        x_r[0::2] = theta_r
        x_r[1::2] = phi_r
        
        try:
            res_r = minimize(
                _compute_energy_and_grad, 
                x_r, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if res_r.success:
                if res_r.fun < best_energy:
                    best_energy = res_r.fun
                    best_x = res_r.x
        except Exception:
            pass

    # Convert best_x back to Cartesian
    theta_best = best_x[0::2]
    phi_best = best_x[1::2]
    xyz_best = _spherical_to_cartesian(theta_best, phi_best)
    
    # Normalize to ensure unit sphere (just in case of numerical drift)
    norms = np.linalg.norm(xyz_best, axis=1)
    xyz_best = xyz_best / norms[:, np.newaxis]
    
    return xyz_best

# EVOLVE-BLOCK-END
