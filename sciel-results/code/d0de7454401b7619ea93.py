import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _energy_and_grad(coords):
    """
    coords: shape (N, 2), [theta, phi]
    Returns: energy, gradient (shape (N, 2))
    """
    theta = coords[:, 0]
    phi = coords[:, 1]
    
    # Compute 3D points
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    
    pts = np.column_stack((x, y, z))
    
    # Compute pairwise distances and dot products (u_ij)
    # diff[i, j, k] = pts[i, k] - pts[j, k]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for self-interaction
    np.fill_diagonal(dists_sq, 1.0)
    
    # u_ij = dot(pts[i], pts[j])
    # We can compute this as 1 - dists_sq / 2
    u = 1.0 - 0.5 * dists_sq
    
    # Energy = sum_{i<j} log(1/sqrt(dists_sq_ij)) = -0.5 * sum_{i<j} log(dists_sq_ij)
    # Using only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d_sq_upper = dists_sq[iu]
    # Clip to avoid log(0) if points are too close, though L-BFGS should avoid this
    d_sq_upper = np.maximum(d_sq_upper, 1e-15)
    energy = -0.5 * np.sum(np.log(d_sq_upper))
    
    # Gradient calculation
    # dE/dtheta_i = sum_{j!=i} (du_ij/dtheta_i) / dists_sq_ij
    # dE/dphi_i   = sum_{j!=i} (du_ij/dphi_i)   / dists_sq_ij
    
    # du_ij/dtheta_i = cos(theta_i)sin(theta_j)cos(phi_i-phi_j) - sin(theta_i)cos(theta_j)
    # du_ij/dphi_i   = -sin(theta_i)sin(theta_j)sin(phi_i-phi_j)
    
    # Vectorized computation
    # We need to sum over j for each i.
    # Let's compute the terms for all pairs (i, j) and then sum.
    
    # theta_i, theta_j
    theta_i = theta[:, np.newaxis] # N x 1
    theta_j = theta[np.newaxis, :] # 1 x N
    phi_i = phi[:, np.newaxis]     # N x 1
    phi_j = phi[np.newaxis, :]     # 1 x N
    
    sin_theta_i = sin_theta[:, np.newaxis]
    sin_theta_j = sin_theta[np.newaxis, :]
    cos_theta_i = cos_theta[:, np.newaxis]
    cos_theta_j = cos_theta[np.newaxis, :]
    
    dphi = phi_i - phi_j
    cos_dphi = np.cos(dphi)
    sin_dphi = np.sin(dphi)
    
    du_dtheta = cos_theta_i * sin_theta_j * cos_dphi - sin_theta_i * cos_theta_j
    du_dphi = -sin_theta_i * sin_theta_j * sin_dphi
    
    # We need to sum over j (axis 1) of (du / dists_sq)
    # dists_sq is N x N. Diagonal is 1.0 (ignored).
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0) # Exclude self
    
    grad_theta = np.sum(du_dtheta * inv_dists_sq, axis=1)
    grad_phi = np.sum(du_dphi * inv_dists_sq, axis=1)
    
    grad = np.column_stack((grad_theta, grad_phi))
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_coords = None
    
    # Initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    theta = np.arccos(z)
    fib_coords = np.column_stack((theta, phi))
    initializations.append(fib_coords)
    
    # 2. Random Uniform
    rng = np.random.default_rng(42)
    for seed in range(5):
        rng_local = np.random.default_rng(seed * 123 + 7)
        # Uniform on sphere: theta = arccos(1 - 2*u), phi = 2*pi*v
        u = rng_local.random(N)
        v = rng_local.random(N)
        theta = np.arccos(1.0 - 2.0 * u)
        phi = 2.0 * np.pi * v
        rand_coords = np.column_stack((theta, phi))
        initializations.append(rand_coords)
        
    # 3. Perturbed Fibonacci
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pert = rng_local.standard_normal((N, 2)) * 0.01
        pert[:, 0] = np.clip(pert[:, 0], -1e-6, 1e-6) # Keep theta away from boundaries if possible, or just let it be
        # Actually, just add small noise
        pert_coords = fib_coords + pert
        # Ensure theta is in [eps, pi-eps]
        eps = 1e-6
        pert_coords[:, 0] = np.clip(pert_coords[:, 0], eps, np.pi - eps)
        initializations.append(pert_coords)
        
    for idx, init_coords in enumerate(initializations):
        if time.time() > deadline - 15:
            break
            
        # Flatten for minimize
        x0 = init_coords.flatten()
        
        # Bounds for theta and phi
        eps = 1e-6
        bounds_theta = (eps, np.pi - eps)
        bounds_phi = (0.0, 2.0 * np.pi)
        bounds = []
        for i in range(N):
            bounds.append(bounds_theta)
            bounds.append(bounds_phi)
            
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.success or res.fun < best_energy:
                current_energy = res.fun
                current_coords = res.x.reshape(N, 2)
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_coords = current_coords.copy()
                    
        except Exception as e:
            pass # Skip if optimization fails
            
    if best_coords is None:
        # Fallback to initial Fibonacci if all failed
        best_coords = initializations[0]
        
    # Convert best_coords (theta, phi) to 3D points
    theta = best_coords[:, 0]
    phi = best_coords[:, 1]
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    
    pts = np.column_stack((x, y, z))
    
    # Normalize to ensure they are exactly on the unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    return pts
