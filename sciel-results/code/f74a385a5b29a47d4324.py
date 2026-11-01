"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: 0.5 * sum_{i!=j} -log(dist)
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _stereo_to_sphere(p):
    """
    Map 2D stereographic coordinates to unit sphere.
    p: (N, 2) array.
    Uses inverse stereographic projection from the North Pole (0,0,1).
    x = 2u / (1+u^2+v^2)
    y = 2v / (1+u^2+v^2)
    z = (1-u^2-v^2) / (1+u^2+v^2)
    """
    u, v = p[:, 0], p[:, 1]
    denom = 1.0 + u**2 + v**2
    x = 2.0 * u / denom
    y = 2.0 * v / denom
    z = (1.0 - u**2 - v**2) / denom
    return np.column_stack([x, y, z])

def _sphere_to_stereo(x):
    """
    Map unit sphere to 2D stereographic coordinates (inverse of above).
    x: (N, 3) array.
    u = x / (1 - z)
    v = y / (1 - z)
    """
    x, y, z = x[:, 0], x[:, 1], x[:, 2]
    # Avoid division by zero if z is close to 1
    # If z is very close to 1, the point is near the north pole.
    # We assume initialization avoids this.
    denom = 1.0 - z
    # Clip to avoid inf/nan
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    u = x / denom
    v = y / denom
    return np.column_stack([u, v])

def _energy_and_grad_stereo(p):
    """
    Computes energy and gradient with respect to stereographic parameters p.
    p: (N, 2)
    """
    x = _stereo_to_sphere(p)
    energy, grad_sphere = _log_energy_and_grad_sphere(x)
    
    # Jacobian of x wrt p
    # x = 2u/D, y = 2v/D, z = (1-u^2-v^2)/D where D = 1+u^2+v^2
    # dx/du = 2/D - 2u*(2u)/D^2 = 2(D-2u^2)/D^2 = 2(1-u^2+v^2)/D^2
    # dx/dv = -2u*(2v)/D^2 = -4uv/D^2
    # dy/du = -4uv/D^2
    # dy/dv = 2(1+u^2-v^2)/D^2
    # dz/du = -4u(1-z^2)/D^2 ... wait, simpler to use chain rule numerically or analytically.
    # Let's derive analytically.
    # D = 1 + u^2 + v^2
    # x = 2u/D
    # dx/du = (2D - 2u*2u)/D^2 = 2(1 - u^2 + v^2)/D^2
    # dx/dv = (0 - 2u*2v)/D^2 = -4uv/D^2
    # y = 2v/D
    # dy/du = -4uv/D^2
    # dy/dv = 2(1 + u^2 - v^2)/D^2
    # z = (1 - u^2 - v^2)/D
    # dz/du = (-2u*D - (1-u^2-v^2)*2u)/D^2 = -2u(1+u^2+v^2 + 1-u^2-v^2)/D^2 = -4u/D^2
    # dz/dv = -4v/D^2
    
    u, v = p[:, 0], p[:, 1]
    D = 1.0 + u**2 + v**2
    D2 = D**2
    
    # Jacobian matrix J of shape (N, 3, 2) where J[i, k, l] = dx_k/du_l
    J = np.zeros((N, 3, 2))
    J[:, 0, 0] = 2.0 * (1.0 - u**2 + v**2) / D2
    J[:, 0, 1] = -4.0 * u * v / D2
    J[:, 1, 0] = -4.0 * u * v / D2
    J[:, 1, 1] = 2.0 * (1.0 + u**2 - v**2) / D2
    J[:, 2, 0] = -4.0 * u / D2
    J[:, 2, 1] = -4.0 * v / D2
    
    # grad_p = J^T grad_x
    # grad_p[i, l] = sum_k J[i, k, l] * grad_sphere[i, k]
    grad_p = np.einsum('ikl,ik->il', J, grad_sphere)
    
    return energy, grad_p

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Rotate x0 so that no point is near the North Pole (0,0,1)
    # The Fibonacci spiral has a point near z=1. We rotate around y-axis by 90 degrees?
    # Or just pick a random rotation to ensure points are spread out in stereographic coords.
    # A simple rotation: swap x and z? No, let's rotate by 45 deg around x-axis.
    # Actually, the worst case for stereographic is z=1.
    # Let's rotate the initial configuration by a random rotation matrix to be safe.
    # Or simpler: use the South Pole projection? No, standard is North Pole.
    # Let's just rotate the sphere so that the "hole" is at a random point far from all initial points.
    # Or, even simpler: use a different initial distribution?
    # The Fibonacci spiral is good. The point with z closest to 1 is i=0.5/N -> z~1.
    # Let's rotate the whole set by 90 degrees around the y-axis.
    # This maps (x,y,z) -> (z, y, -x). The old north pole (0,0,1) becomes (1,0,0).
    # The new north pole is (0,0,1) which was (0,0,-1) in old coords (South Pole).
    # The Fibonacci spiral covers the sphere uniformly, so there is a point near the South Pole.
    # So this rotation might still have a point near the new North Pole.
    # Better: Use a random rotation.
    
    # Generate random rotation
    rng = np.random.default_rng(42)
    # Random rotation matrix
    # Use QR decomposition of a random Gaussian matrix
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det=1)
    d = np.sign(np.linalg.det(Q))
    Q[:, 0] *= d
    if np.linalg.det(Q) < 0:
        Q[:, 1] *= -1
        
    x0_rotated = x0 @ Q.T
    
    # Check if any point is too close to North Pole (0,0,1)
    # If z > 0.99, it's risky.
    # If so, rotate again or use a different chart.
    # For robustness, let's try multiple random rotations and pick the one with max min distance to North Pole.
    best_rot = None
    best_min_dist = -1
    for _ in range(10):
        A = rng.normal(size=(3, 3))
        Q, R = np.linalg.qr(A)
        d = np.sign(np.linalg.det(Q))
        Q[:, 0] *= d
        if np.linalg.det(Q) < 0:
            Q[:, 1] *= -1
        x_rot = x0 @ Q.T
        dist_to_np = np.linalg.norm(x_rot - np.array([0,0,1]), axis=1)
        min_dist = np.min(dist_to_np)
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_rot = Q
            
    x0_opt = x0 @ best_rot.T
    
    # Convert to stereographic parameters
    p0 = _sphere_to_stereo(x0_opt)
    
    best_energy, _ = _log_energy_and_grad_sphere(x0_opt)
    best_p = p0.copy()
    
    # Optimize using L-BFGS-B
    # Flatten p0
    p_flat0 = p0.flatten()
    
    def obj_flat(p_flat):
        p = p_flat.reshape(N, 2)
        e, g = _energy_and_grad_stereo(p)
        return e, g.flatten()
        
    # Run optimization
    # We might need to handle cases where points fly off to infinity (z->1)
    # L-BFGS-B should handle this if bounds are set, but we don't have strict bounds.
    # However, if a point goes to infinity, energy goes to infinity (log(1/d) -> -inf? No, log(1/d) -> -inf as d->0? 
    # Wait. Energy is sum log(1/d). If d->0, log(1/d)->+inf. So energy blows up.
    # So the optimizer will naturally avoid collisions.
    
    try:
        res = minimize(obj_flat, p_flat0, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'maxfun': 500})
        
        if time.time() < deadline:
            p_res = res.x.reshape(N, 2)
            x_res = _stereo_to_sphere(p_res)
            e_res, _ = _log_energy_and_grad_sphere(x_res)
            
            if e_res < best_energy:
                best_energy = e_res
                best_x = x_res
            else:
                best_x = x0_opt
                
    except Exception:
        best_x = x0_opt

    # If we have time left, try a second optimization from a different start or polish
    if time.time() < deadline - 5:
        # Try a second run with a different random rotation or initial perturbation
        # Or just run L-BFGS-B again on the result to ensure convergence
        p_curr = _sphere_to_stereo(best_x)
        p_flat_curr = p_curr.flatten()
        
        try:
            res2 = minimize(obj_flat, p_flat_curr, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'maxfun': 200})
            if res2.success or res2.fun < best_energy:
                p_res2 = res2.x.reshape(N, 2)
                x_res2 = _stereo_to_sphere(p_res2)
                e_res2, _ = _log_energy_and_grad_sphere(x_res2)
                if e_res2 < best_energy:
                    best_x = x_res2
        except Exception:
            pass

    # Final check
    if time.time() > deadline - 2:
         return best_x

    # Ensure points are on sphere
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    
    return best_x

# EVOLVE-BLOCK-END
