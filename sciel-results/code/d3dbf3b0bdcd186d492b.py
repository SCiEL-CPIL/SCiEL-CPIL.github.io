import numpy as np
from scipy.optimize import minimize
import time

N = 150

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral method."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _points_to_spherical(points):
    """Convert Cartesian points to spherical coordinates (theta, phi)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    r[r == 0] = 1e-12
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    # Ensure phi is in [0, 2*pi)
    phi = phi % (2 * np.pi)
    return theta, phi

def _spherical_to_points(theta, phi):
    """Convert spherical coordinates (theta, phi) to Cartesian points."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _log_energy_and_grad(params):
    """
    Compute log energy and its gradient with respect to spherical parameters.
    params: array of shape (2*N,) containing [theta_1, phi_1, theta_2, phi_2, ...]
    """
    theta = params[0::2]
    phi = params[1::2]
    
    # Convert to Cartesian
    points = _spherical_to_points(theta, phi)
    
    # Compute pairwise distances
    # points: (N, 3)
    # diff: (N, N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Add epsilon to avoid log(0)
    eps = 1e-12
    dist_sq_safe = dist_sq + eps
    
    # Energy: sum over i<j of log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij) = -0.25 * sum_{i!=j} log(d_ij^2)
    # We sum over all i!=j and divide by 2 to get i<j sum.
    # log(d_ij) = 0.5 * log(d_ij^2)
    # E = sum_{i<j} log(1/d_ij) = - sum_{i<j} 0.5 * log(d_ij^2) = -0.25 * sum_{i!=j} log(d_ij^2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    log_dist_sq = np.log(dist_sq_safe)
    energy = -0.25 * np.sum(log_dist_sq[mask])
    
    # Gradient calculation
    # grad_u_k = sum_{j != k} (u_k - u_j) / dist_sq_kj
    # We need grad_theta_k and grad_phi_k
    
    # diff is (i, j, 3) where diff[i,j] = u_i - u_j
    # grad_u[i] = sum_j (u_i - u_j) / dist_sq[i,j] for j != i
    
    # We can compute the sum of (u_i - u_j) / dist_sq[i,j] efficiently
    # Note: dist_sq[i,i] = 0, so we must handle the diagonal.
    # Using eps, dist_sq_safe[i,i] = eps. The term (u_i-u_i)/eps = 0. So it's safe.
    
    inv_dist_sq = 1.0 / dist_sq_safe
    # Set diagonal to 0 to be precise, though it's 0 anyway
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_u[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff[i,j] is vector. inv_dist_sq[i,j] is scalar.
    grad_u = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Now project grad_u to theta and phi
    # d u / d theta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
    # d u / d phi   = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    
    d_u_d_theta = np.column_stack([
        cos_t * cos_p,
        cos_t * sin_p,
        -sin_t
    ])
    
    d_u_d_phi = np.column_stack([
        -sin_t * sin_p,
        sin_t * cos_p,
        np.zeros(N)
    ])
    
    # grad_theta[i] = dot(grad_u[i], d_u_d_theta[i])
    grad_theta = np.sum(grad_u * d_u_d_theta, axis=1)
    grad_phi = np.sum(grad_u * d_u_d_phi, axis=1)
    
    # Combine into a single gradient vector
    grad = np.empty(2 * N)
    grad[0::2] = grad_theta
    grad[1::2] = grad_phi
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, leave margin
    
    # Initial guess: Fibonacci spiral
    points_init = _fibonacci_sphere(N)
    theta_init, phi_init = _points_to_spherical(points_init)
    params_init = np.empty(2 * N)
    params_init[0::2] = theta_init
    params_init[1::2] = phi_init
    
    bounds = [(0, np.pi), (0, 2 * np.pi)] * N
    
    best_params = params_init
    best_energy = np.inf
    
    # Try a few restarts to escape local minima
    # 1. Fibonacci
    # 2. Random
    # 3. Perturbed Fibonacci
    
    num_restarts = 3
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:  # Stop if close to deadline
            break
            
        if restart == 0:
            p0 = params_init
        elif restart == 1:
            # Random start
            theta_r = np.random.uniform(0, np.pi, N)
            phi_r = np.random.uniform(0, 2 * np.pi, N)
            p0 = np.empty(2 * N)
            p0[0::2] = theta_r
            p0[1::2] = phi_r
        else:
            # Perturbed Fibonacci
            p0 = params_init + np.random.normal(0, 0.1, 2 * N)
            # Wrap phi
            p0[1::2] = p0[1::2] % (2 * np.pi)
            # Clip theta
            p0[0::2] = np.clip(p0[0::2], 0.01, np.pi - 0.01)
        
        # Optimize
        try:
            res = minimize(
                _log_energy_and_grad,
                p0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_energy:
                best_params = res.x
                best_energy = res.fun
        except Exception:
            pass

    # Convert best params back to points
    theta_best = best_params[0::2]
    phi_best = best_params[1::2]
    points_best = _spherical_to_points(theta_best, phi_best)
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(points_best, axis=1, keepdims=True)
    points_best = points_best / norms
    
    return points_best
