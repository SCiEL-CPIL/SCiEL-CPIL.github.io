"""
Strategy: Stereographic Projection + L-BFGS-B
1. Parameterize the sphere using stereographic projection from the South Pole.
   This maps the sphere (minus one point) to R^2. Since N=100 points are unlikely
   to cluster exactly at the South Pole in a good configuration, this provides
   a valid, smooth, unconstrained parameterization of the sphere.
2. Use scipy.optimize.minimize with method='L-BFGS-B' on the 2N parameters.
   L-BFGS-B is a quasi-Newton method that converges much faster than GD/Adam
   for smooth functions and handles the large dimensionality (200 vars) well.
3. Initialize with a Sobol sequence for better uniform coverage than Fibonacci
   or Random, reducing the likelihood of starting in a bad local minimum.
4. Multi-start with different Sobol offsets and random rotations to escape local minima.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
DEADLINE_MARGIN = 5.0

# EVOLVE-BLOCK-START

def _stereo_to_sphere(u):
    """
    Convert 2D parameters u (shape N, 2) to points on the unit sphere (shape N, 3).
    Stereographic projection from South Pole (0, 0, -1).
    x = 2*u_x / (1 + |u|^2)
    y = 2*u_y / (1 + |u|^2)
    z = (1 - |u|^2) / (1 + |u|^2)
    """
    u_sq = u[:, 0]**2 + u[:, 1]**2
    denom = 1.0 + u_sq
    x = 2.0 * u[:, 0] / denom
    y = 2.0 * u[:, 1] / denom
    z = (1.0 - u_sq) / denom
    return np.column_stack([x, y, z])

def _sphere_to_stereo(pts):
    """
    Convert points on the sphere (shape N, 3) to 2D parameters (shape N, 2).
    Inverse of stereographic projection from South Pole.
    u_x = x / (1 + z)
    u_y = y / (1 + z)
    Note: If z is close to -1, this blows up. We assume our configs avoid this.
    """
    # Clip z to avoid division by zero if a point is exactly at the South Pole
    z = np.clip(pts[:, 2], -0.999999, 0.999999)
    denom = 1.0 + z
    u_x = pts[:, 0] / denom
    u_y = pts[:, 1] / denom
    return np.column_stack([u_x, u_y])

def _energy_and_grad_sphere(pts):
    """
    Compute logarithmic energy and gradient for points on the sphere.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid log(0)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist_sq[mask] < 1e-18):
        return float('inf'), None
        
    dists = np.sqrt(dist_sq[mask])
    energy = -np.sum(np.log(dists))
    
    # Gradient on the sphere: sum_{j!=i} (x_i - x_j) / d_ij^2
    # This is the gradient of the energy with respect to the position vector x_i
    # projected onto the tangent space? 
    # Actually, the gradient of the energy E = sum log(1/d) with respect to x_i in R^3
    # is sum_{j!=i} (x_i - x_j) / d_ij^2.
    # Since we are optimizing on the sphere, we need the Riemannian gradient, which is
    # the Euclidean gradient projected onto the tangent space:
    # grad_tangent_i = grad_euclidean_i - (grad_euclidean_i . x_i) x_i
    # However, since x_i is a unit vector, and the force is central, 
    # let's check: (x_i - x_j).x_i = 1 - x_i.x_j.
    # The projection is necessary for the Riemannian flow.
    
    dist_sq_safe = dist_sq.copy()
    dist_sq_safe[dist_sq_safe == 0] = 1.0
    
    # Euclidean gradient
    terms = diff / dist_sq_safe[:, :, np.newaxis]
    idx = np.arange(n)
    terms[idx, idx] = 0.0
    grad_euclidean = np.sum(terms, axis=1)
    
    # Project to tangent space: grad_tan = grad_euclidean - (grad_euclidean . x_i) x_i
    dot = np.sum(grad_euclidean * pts, axis=1, keepdims=True)
    grad_tangent = grad_euclidean - dot * pts
    
    return energy, grad_tangent

def _objective_and_jac(u_flat):
    """
    Objective and Jacobian for the unconstrained parameters u (flattened).
    """
    u = u_flat.reshape(N, 2)
    pts = _stereo_to_sphere(u)
    
    # Check for invalid points (shouldn't happen with stereo, but safe)
    if not np.all(np.isfinite(pts)):
        return float('inf'), np.zeros_like(u_flat)
        
    energy, grad_tangent = _energy_and_grad_sphere(pts)
    
    if not np.isfinite(energy):
        return float('inf'), np.zeros_like(u_flat)
        
    # Chain rule: dE/du = (dE/dx) . (dx/du)
    # dx/du is the Jacobian of the stereographic mapping.
    # x = 2u_x / D, y = 2u_y / D, z = (1-u^2)/D, where D = 1+u^2
    # Let u^2 = u_x^2 + u_y^2
    # dx/du_x = (2D - 2u_x(2u_x))/D^2 = 2(D-u_x^2)/D^2 = 2(1+u_y^2)/D^2
    # dx/du_y = -2u_x(2u_y)/D^2 = -4u_x u_y / D^2
    # dy/du_x = -4u_x u_y / D^2
    # dy/du_y = 2(D-u_y^2)/D^2 = 2(1+u_x^2)/D^2
    # dz/du_x = -2u_x(2u_x)/D^2 = -4u_x^2/D^2  ... wait, derivative of (1-u^2)/D
    # dz/du_x = [ -2u_x D - (1-u^2)2u_x ] / D^2 = -2u_x [ D + 1 - u^2 ] / D^2 = -2u_x [ 1+u^2+1-u^2 ]/D^2 = -4u_x/D^2
    # dz/du_y = -4u_y/D^2
    
    u_sq = u[:, 0]**2 + u[:, 1]**2
    D = 1.0 + u_sq
    D2 = D**2
    
    # Jacobian of pts wrt u
    # J[i, 0, 0] = dx/du_x, J[i, 0, 1] = dx/du_y
    # J[i, 1, 0] = dy/du_x, J[i, 1, 1] = dy/du_y
    # J[i, 2, 0] = dz/du_x, J[i, 2, 1] = dz/du_y
    
    J_xx = 2.0 * (1.0 + u[:, 1]**2) / D2
    J_xy = -4.0 * u[:, 0] * u[:, 1] / D2
    J_yx = -4.0 * u[:, 0] * u[:, 1] / D2
    J_yy = 2.0 * (1.0 + u[:, 0]**2) / D2
    J_zx = -4.0 * u[:, 0] / D2
    J_zy = -4.0 * u[:, 1] / D2
    
    # grad_u = grad_tan_x * J_xx + grad_tan_y * J_yx + grad_tan_z * J_zx  (for u_x)
    # grad_u = grad_tan_x * J_xy + grad_tan_y * J_yy + grad_tan_z * J_zy  (for u_y)
    
    gt_x = grad_tangent[:, 0]
    gt_y = grad_tangent[:, 1]
    gt_z = grad_tangent[:, 2]
    
    grad_u_x = gt_x * J_xx + gt_y * J_yx + gt_z * J_zx
    grad_u_y = gt_x * J_xy + gt_y * J_yy + gt_z * J_zy
    
    grad_flat = np.column_stack([grad_u_x, grad_u_y]).flatten()
    
    return energy, grad_flat

def _sobol_sphere(n, offset=0.0):
    """Generate points on the sphere using a Sobol sequence."""
    # Sobol sequence in 3D
    sampler = qmc.Sobol(d=3, scramble=True)
    # We need n points. Sobol requires power of 2, so we generate next_power_of_2
    m = 1
    while m < n:
        m *= 2
    samples = sampler.random(m)
    
    # Map [0,1]^3 to sphere
    # Use spherical coordinates
    # theta (azimuth) in [0, 2pi], phi (polar) in [0, pi]
    # To avoid clustering at poles, use sin(phi) uniform? 
    # Actually, standard uniform on sphere:
    # z = 2*u1 - 1
    # theta = 2*pi*u2
    # x = sqrt(1-z^2) * cos(theta)
    # y = sqrt(1-z^2) * sin(theta)
    
    # Apply offset to shift the sequence
    samples = (samples + offset) % 1.0
    
    z = 2.0 * samples[:, 0] - 1.0
    theta = 2.0 * math.pi * samples[:, 1]
    
    r = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    pts = np.column_stack([x, y, z])
    return pts[:n]

def _random_rotation(pts, rng):
    """Apply a random rotation to points on the sphere."""
    # Generate random rotation matrix
    # Using QR decomposition of a random Gaussian matrix
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det=1)
    d = np.sign(np.linalg.det(Q))
    Q[:, -1] *= d
    return Q @ pts.T

def _optimize_stereo(u0, time_limit):
    """
    Optimize using L-BFGS-B on the stereographic parameters.
    """
    start_time = time.time()
    
    # Flatten u0
    u0_flat = u0.flatten()
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            return True
        return False
    
    try:
        res = minimize(
            _objective_and_jac,
            u0_flat,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        # Check if it converged or timed out
        # res.x is the result
        u_opt = res.x.reshape(N, 2)
        pts_opt = _stereo_to_sphere(u_opt)
        energy_opt, _ = _energy_and_grad_sphere(pts_opt)
        
        if not np.isfinite(energy_opt):
            return u0, _energy_and_grad_sphere(_stereo_to_sphere(u0))[0]
            
        return u_opt, energy_opt
        
    except Exception:
        return u0, _energy_and_grad_sphere(_stereo_to_sphere(u0))[0]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 - DEADLINE_MARGIN # 115s limit
    
    best_pts = None
    best_energy = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate multiple diverse starting points
    # 1. Sobol sequence (no rotation)
    # 2. Sobol sequence (rotated)
    # 3. Random uniform (rotated)
    
    starts = []
    
    # Sobol base
    sobol_pts = _sobol_sphere(N, offset=0.0)
    starts.append(sobol_pts)
    
    # Sobol with different offsets
    for offset in [0.1, 0.3, 0.7]:
        starts.append(_sobol_sphere(N, offset=offset))
        
    # Random starts
    for _ in range(4):
        rand_pts = rng.normal(size=(N, 3))
        rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
        starts.append(rand_pts)
        
    # Rotated Sobol
    for i in range(4):
        rot_pts = _random_rotation(_sobol_sphere(N, offset=0.5), rng)
        starts.append(rot_pts)
        
    # Sort starts by initial energy? Maybe not, just run them.
    # But running the best initial energy first might find a good baseline quickly.
    
    # Convert to stereographic parameters
    # If a point is at the south pole (z=-1), stereo fails.
    # Our generators avoid this, but just in case, shift them slightly.
    valid_starts_u = []
    for pts in starts:
        # Check for south pole
        if np.any(pts[:, 2] < -0.99999):
            # Perturb
            pts = pts.copy()
            pts[pts[:, 2] < -0.99999, 0] += 1e-5
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        try:
            u = _sphere_to_stereo(pts)
            if np.all(np.isfinite(u)):
                valid_starts_u.append(u)
        except:
            continue
            
    # Optimize each start
    num_starts = len(valid_starts_u)
    time_per_start = (deadline - time.time()) / max(1, num_starts)
    
    for i, u0 in enumerate(valid_starts_u):
        if time.time() > deadline:
            break
            
        # Allocate time for this start
        remaining_starts = num_starts - i
        t_limit = min(time_per_start, (deadline - time.time()) / remaining_starts)
        t_limit = max(t_limit, 2.0) # Min 2s per start
        
        u_opt, energy = _optimize_stereo(u0, t_limit)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = _stereo_to_sphere(u_opt)
            
    # If no valid start, fallback to Fibonacci
    if best_pts is None:
        best_pts = _sobol_sphere(N, offset=0.0)
        best_energy, _ = _energy_and_grad_sphere(best_pts)
        
    # Final polish with a very tight tolerance if time remains
    if time.time() < deadline - 10.0:
        u_best = _sphere_to_stereo(best_pts)
        u_final, energy_final = _optimize_stereo(u_best, time_limit=deadline - time.time() - 5.0)
        if energy_final < best_energy:
            best_pts = _stereo_to_sphere(u_final)
            best_energy = energy_final
            
    # Ensure output is on the sphere
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts

# EVOLVE-BLOCK-END
