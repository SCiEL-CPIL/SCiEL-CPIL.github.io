"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = pts.shape[0]
    # Compute pairwise distances vectorized
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    dist = np.sqrt(dist_sq)  # (N, N)
    
    # Upper triangle indices
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    # Use math.fsum for accuracy as specified
    return -math.fsum(math.log(x) for x in d)


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    pts: (N, 3) array
    Returns: (energy, gradient)
    """
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    dist = np.sqrt(dist_sq)  # (N, N)
    
    # Energy: sum_{i<j} log(1/d_{ij}) = -sum_{i<j} log(d_{ij})
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    # Vectorized gradient computation
    # grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # dist_sq[i,j] = |x_i - x_j|^2
    # We want grad[i] = sum_j (pts[i] - pts[j]) / dist_sq[i,j] for j != i
    
    # Create inverse distance squared matrix, zero out diagonal
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = ~np.eye(n, dtype=bool)
    # Avoid division by zero
    safe_dist_sq = np.where(dist_sq > 1e-24, dist_sq, 1e24)
    inv_dist_sq[mask] = 1.0 / safe_dist_sq[mask]
    
    # grad[i] = sum_j (pts[i] - pts[j]) * inv_dist_sq[i,j]
    #         = pts[i] * sum_j inv_dist_sq[i,j] - sum_j pts[j] * inv_dist_sq[i,j]
    #         = pts[i] * row_sums - pts @ inv_dist_sq.T
    
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    grad = pts * row_sums[:, np.newaxis] - pts @ inv_dist_sq.T  # (N, 3)
    
    # Project gradient onto tangent space: grad_tan = grad - (grad . x_i) * x_i
    dots = np.sum(grad * pts, axis=1)  # (N,)
    grad = grad - dots[:, np.newaxis] * pts
    
    return energy, grad


def _spherical_energy_and_grad(params):
    """Unpack spherical coordinates, compute energy and gradient.
    params: (2N,) array of (theta_i, phi_i) for each point
    """
    thetas = params[0::2]
    phis = params[1::2]
    
    # Convert to Cartesian
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    pts = np.column_stack([x, y, z])
    
    # Compute energy
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    # Vectorized Cartesian gradient
    n = N
    mask = ~np.eye(n, dtype=bool)
    safe_dist_sq = np.where(dist_sq > 1e-24, dist_sq, 1e24)
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / safe_dist_sq[mask]
    
    row_sums = np.sum(inv_dist_sq, axis=1)
    grad_cart = pts * row_sums[:, np.newaxis] - pts @ inv_dist_sq.T
    
    # Convert gradient to spherical coordinates
    # dE/dtheta = grad_cart . dpts/dtheta
    # dE/dphi = grad_cart . dpts/dphi
    grad_params = np.zeros_like(params)
    
    # Vectorized computation of dtheta and dphi components
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    cos_phis = np.cos(phis)
    sin_phis = np.sin(phis)
    
    # dpts/dtheta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
    # dpts/dphi = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
    
    # dE/dtheta_i = grad_cart[i] . dpts_i/dtheta
    # = grad_cart[i,0]*cos(th_i)*cos(ph_i) + grad_cart[i,1]*cos(th_i)*sin(ph_i) + grad_cart[i,2]*(-sin(th_i))
    grad_params[0::2] = (grad_cart[:, 0] * cos_thetas * cos_phis + 
                         grad_cart[:, 1] * cos_thetas * sin_phis - 
                         grad_cart[:, 2] * sin_thetas)
    
    # dE/dphi_i = grad_cart[i] . dpts_i/dphi
    # = grad_cart[i,0]*(-sin(th_i)*sin(ph_i)) + grad_cart[i,1]*sin(th_i)*cos(ph_i) + 0
    grad_params[1::2] = (-grad_cart[:, 0] * sin_thetas * sin_phis + 
                         grad_cart[:, 1] * sin_thetas * cos_phis)
    
    return energy, grad_params


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _momentum_descent(pts, max_iters=2000, lr=0.01, momentum=0.9, tol=1e-12):
    """Momentum-based projected gradient descent on sphere."""
    pts = pts.copy()
    n = pts.shape[0]
    E = _log_energy(pts)
    vel = np.zeros_like(pts)
    
    for it in range(max_iters):
        E_vec, g = _energy_and_grad(pts)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            break
        
        # Update velocity with momentum
        vel = momentum * vel - lr * g
        
        # Step
        new_pts = pts + vel
        new_pts = _project_to_sphere(new_pts)
        
        new_E = _log_energy(new_pts)
        
        if new_E < E:
            pts = new_pts
            E = new_E
        else:
            # Reduce learning rate
            lr *= 0.5
            vel = 0.5 * vel
            if lr < 1e-10:
                break
    
    return pts, E


def construct():
    # Golden-angle (Fibonacci) spiral points on the unit sphere: a good but improvable start.
    pts = _fibonacci_sphere(N)
    pts = _project_to_sphere(pts)
    
    # Time limit: 150s, keep safety margin of 10s
    start_time = time.time()
    time_limit = 140.0  # safety margin
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Try to improve with gradient-based optimization using spherical coordinates
    # Initial spherical coordinates
    thetas = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phis = np.arctan2(pts[:, 1], pts[:, 0])
    params0 = np.zeros(2 * N)
    params0[0::2] = thetas
    params0[1::2] = phis
    
    # Optimize with L-BFGS-B
    try:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 10:
            result = minimize(_spherical_energy_and_grad, params0, method='L-BFGS-B',
                              jac=True, options={'maxiter': 5000, 'maxfun': 50000,
                                                'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.success or result.fun < best_energy:
                thetas_opt = result.x[0::2]
                phis_opt = result.x[1::2]
                x = np.sin(thetas_opt) * np.cos(phis_opt)
                y = np.sin(thetas_opt) * np.sin(phis_opt)
                z = np.cos(thetas_opt)
                pts_opt = np.column_stack([x, y, z])
                pts_opt = _project_to_sphere(pts_opt)
                
                e_opt = _log_energy(pts_opt)
                if e_opt < best_energy:
                    best_pts = pts_opt
                    best_energy = e_opt
    except Exception:
        pass
    
    # Check time and try momentum descent if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        try:
            pts_md, E_md = _momentum_descent(best_pts, max_iters=1000, lr=0.005, momentum=0.9)
            if E_md < best_energy:
                best_pts = pts_md
                best_energy = E_md
        except Exception:
            pass
    
    # Final polish with smaller steps if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 3:
        try:
            pts_polish, E_polish = _momentum_descent(best_pts, max_iters=500, lr=0.001, momentum=0.9, tol=1e-14)
            if E_polish < best_energy:
                best_pts = pts_polish
                best_energy = E_polish
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Check for near-duplicate points and nudge apart if needed
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                perp = best_pts[i] - np.dot(best_pts[i], best_pts[j]) * best_pts[j]
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 1e-15:
                    best_pts[j] = best_pts[j] + 1e-8 * perp / perp_norm
                else:
                    # Use a different direction
                    best_pts[j] = best_pts[j] + 1e-8 * np.array([1.0, 0.0, 0.0])
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    # Final check
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
