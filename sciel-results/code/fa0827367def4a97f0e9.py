"""Minimize logarithmic energy of 150 points on the unit sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds


def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    pts: (N, 3) array of unit vectors.
    Energy = sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient with respect to x_i (constrained to sphere, but we compute unconstrained grad
    and project, or use the fact that on the sphere the gradient of the energy restricted
    to the tangent space is what matters).
    
    For point i:
    dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    (since d/dx_i log(1/|x_i-x_j|) = -d/dx_i log|x_i-x_j| = -(x_i-x_j)/|x_i-x_j|^2)
    
    Wait, let me re-derive:
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    dE/dx_i = -sum_{j>i} d/dx_i log(d_ij) - sum_{j<i} d/dx_i log(d_ij)
            = -sum_{j!=i} (x_i - x_j) / (2 * d_ij^2) * 2  ... 
    
    Actually, d/dx_i log(d_ij) = d/dx_i log(|x_i - x_j|) = (x_i - x_j) / |x_i - x_j|^2
    
    So dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    # Compute pairwise distances
    # pts: (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero
    dist_safe = np.where(dist < 1e-12, 1e-12, dist)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist_safe[iu]))
    
    # Gradient: grad_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad_i = -sum_j (diff[i,j] / dist_safe[i,j]^2)
    # But we need to be careful: dist_safe[i,i] = 0, so we mask the diagonal
    
    inv_dist_sq = np.zeros_like(dist_safe)
    mask = dist_safe > 1e-12
    inv_dist_sq[mask] = 1.0 / (dist_safe[mask] ** 2)
    
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff[i,j] = x_i - x_j
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def construct():
    start_time = time.time()
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Use gradient descent with Barzilai-Borwein step sizes
    # First, compute initial gradient
    energy, grad = _energy_and_grad(pts)
    
    # BB step size parameters
    s_prev = None
    y_prev = None
    step = 0.01
    
    max_iters = 5000
    prev_grad = grad.copy()
    prev_pts = pts.copy()
    
    for it in range(max_iters):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN:
            break
        
        # Compute BB step size
        if s_prev is not None and y_prev is not None:
            sy = np.dot(s_prev, y_prev)
            if abs(sy) > 1e-12:
                step = np.dot(s_prev, s_prev) / sy
                step = np.clip(step, 1e-6, 0.1)
        
        # Gradient descent step (unconstrained, then project)
        new_pts = pts - step * grad
        
        # Project to sphere
        new_pts = _project_to_sphere(new_pts)
        
        # Compute new energy and gradient
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        # Accept if energy decreased
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        
        if new_energy < energy:
            # Update BB quantities
            s_prev = new_pts - pts
            y_prev = new_grad - grad
            pts = new_pts
            grad = new_grad
            energy = new_energy
        else:
            # Reduce step size
            step *= 0.5
            # Don't update BB quantities on rejected step
    
    # Final high-precision polish using scipy's L-BFGS-B on spherical coordinates
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN - 5.0:
        # Convert to spherical coordinates for optimization
        def _cartesian_to_spherical(pts):
            """Convert (x,y,z) to (theta, phi) where theta is polar, phi is azimuthal."""
            r = np.linalg.norm(pts, axis=1)
            theta = np.arccos(np.clip(pts[:, 2] / r, -1.0, 1.0))  # polar angle
            phi = np.arctan2(pts[:, 1], pts[:, 0])  # azimuthal angle
            return np.column_stack([theta, phi])
        
        def _spherical_to_cartesian(sp):
            """Convert (theta, phi) to (x,y,z) on unit sphere."""
            theta = sp[:, 0]
            phi = sp[:, 1]
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            return np.column_stack([x, y, z])
        
        def _spherical_energy_grad(sp):
            """Energy and gradient in spherical coordinates."""
            pts = _spherical_to_cartesian(sp)
            e, g_cart = _energy_and_grad(pts)
            
            # Transform gradient from cartesian to spherical
            # x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = cos(theta)
            # dx/dtheta = cos(theta)*cos(phi), dx/dphi = -sin(theta)*sin(phi)
            # dy/dtheta = cos(theta)*sin(phi), dy/dphi = sin(theta)*cos(phi)
            # dz/dtheta = -sin(theta), dz/dphi = 0
            theta = sp[:, 0]
            phi = sp[:, 1]
            
            st = np.sin(theta)
            ct = np.cos(theta)
            cp = np.cos(phi)
            sp_ = np.sin(phi)
            
            # grad_theta_i = g_cart_x * dx/dtheta + g_cart_y * dy/dtheta + g_cart_z * dz/dtheta
            grad_theta = g_cart[:, 0] * ct * cp + g_cart[:, 1] * ct * sp_ + g_cart[:, 2] * (-st)
            grad_phi = g_cart[:, 0] * (-st * sp_) + g_cart[:, 1] * (st * cp)
            
            g_sph = np.column_stack([grad_theta, grad_phi])
            
            return e, g_sph.flatten()
        
        # Start from best points found so far
        sp_init = _cartesian_to_spherical(best_pts)
        
        try:
            res = minimize(_spherical_energy_grad, sp_init, method='L-BFGS-B',
                          jac=True, options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if res.success or res.fun < best_energy:
                final_pts = _spherical_to_cartesian(res.x)
                final_pts = _project_to_sphere(final_pts)
                final_energy = _energy_and_grad(final_pts)[0]
                if final_energy < best_energy:
                    best_pts = final_pts
                    best_energy = final_energy
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: no points too close, all finite
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9 or not np.all(np.isfinite(best_pts)):
        # Fallback to initial Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
