import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius in radians for points on the unit sphere.
    The covering radius is the maximum circumradius of the convex hull facets.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi # Fallback for degenerate cases
        
    max_radius = 0.0
    # hull.simplices contains the indices of the vertices of each facet
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        # Normal to the plane of the triangle
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        
        if norm_normal < 1e-10:
            continue
            
        # Distance from origin to the plane
        # d = |normal . p0| / ||normal||
        d = np.abs(np.dot(normal, p0)) / norm_normal
        
        # The circumradius (angular) alpha satisfies cos(alpha) = d
        # Since d is in [0, 1], alpha is in [0, pi/2] for valid hull facets on unit sphere
        # Actually, for a triangle on the unit sphere, the circumcenter is on the sphere
        # and the distance from origin to the plane is cos(alpha).
        # So alpha = arccos(d).
        
        # Clip d to [0, 1] to avoid numerical errors in arccos
        d = np.clip(d, 0.0, 1.0)
        alpha = np.arccos(d)
        
        if alpha > max_radius:
            max_radius = alpha
            
    return max_radius

def _repulsive_forces(points, k=1.0):
    """
    Vectorized repulsive forces between all pairs of points.
    Force on point i from point j: proportional to (pi - pj) / |pi - pj|^3
    """
    # Compute all pairwise differences
    # diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    
    # Compute squared distances
    dist_sq = np.sum(diff ** 2, axis=2)  # (N, N)
    
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    # Force magnitude: 1 / dist^2 (Coulomb-like)
    # Force vector: diff / dist^3 = (diff / dist) * (1 / dist^2)
    # diff / dist^3 = diff / (dist_sq * sqrt(dist_sq))
    inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))  # (N, N)
    
    # Set diagonal to zero (no self-force)
    np.fill_diagonal(inv_dist_cubed, 0.0)
    
    # Force on each point: sum over j of (pi - pj) / |pi - pj|^3
    forces = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)  # (N, 3)
    
    return k * forces

def _density_scaled_forces(points, k=1.0, gamma=1.0):
    """
    Repulsive forces scaled by local density.
    Points in sparse regions (low density) experience stronger repulsion,
    causing them to flow into empty regions.
    """
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff ** 2, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Local density: number of neighbors within a certain radius, or inverse of nearest neighbor distance
    # Use inverse of nearest neighbor distance as a density proxy
    nn_dist_sq = np.min(dist_sq, axis=1)  # (N,)
    nn_dist = np.sqrt(nn_dist_sq)
    
    # Density proxy: 1 / nn_dist (higher when points are closer)
    # Scale forces by density: points with low density (high nn_dist) get stronger push
    # Actually, we want points in sparse regions to be pushed more strongly
    # So scale by 1/density = nn_dist, or use a power law
    scale = nn_dist ** gamma  # (N,)
    
    # Compute standard repulsive forces
    forces = _repulsive_forces(points, k=1.0)
    
    # Scale each point's force by its local density factor
    forces *= scale[:, np.newaxis]
    
    return forces

def _repulsive_relaxation(points, steps=2000, alpha=0.05, use_density_scaling=False):
    """
    Relax points using repulsive forces to spread them out.
    Vectorized for speed.
    """
    points = points.copy()
    for _ in range(steps):
        if use_density_scaling:
            forces = _density_scaled_forces(points, k=1.0, gamma=1.0)
        else:
            forces = _repulsive_forces(points, k=1.0)
        
        # Update positions
        points += alpha * forces
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
        
    return points

def _spherical_to_cartesian(params):
    """
    Convert spherical coordinates (theta, phi) to Cartesian coordinates.
    theta: polar angle [0, pi]
    phi: azimuthal angle [0, 2*pi]
    """
    theta = params[0::2]
    phi = params[1::2]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(points):
    """
    Convert Cartesian coordinates to spherical coordinates (theta, phi).
    """
    theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])
    params = np.zeros(2 * len(points))
    params[0::2] = theta
    params[1::2] = phi
    return params

def _local_optimize_nelder_mead(points, deadline, maxiter=500):
    """
    Local optimization using Nelder-Mead on spherical coordinates.
    This handles the sphere constraint naturally.
    """
    def objective(params):
        if time.time() > deadline:
            return 1e6
        pts = _spherical_to_cartesian(params)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _covering_radius(pts)
    
    params0 = _cartesian_to_spherical(points)
    
    try:
        result = minimize(objective, params0, method='Nelder-Mead', 
                          options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8})
        if result.fun < _covering_radius(points):
            return _spherical_to_cartesian(result.x)
    except Exception:
        pass
        
    return points

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (10s buffer before 150s kill)
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # 2. Repulsive Relaxation with density scaling
    # Use a moderate number of steps to get a good starting point for Nelder-Mead
    points = _repulsive_relaxation(points, steps=2000, alpha=0.03, use_density_scaling=True)
    current_radius = _covering_radius(points)
    if current_radius < best_radius:
        best_points = points.copy()
        best_radius = current_radius
    
    # 3. Nelder-Mead Optimization on Spherical Coordinates
    # This is the key refinement step from the inspiration
    if time.time() < deadline - 30:
        optimized = _local_optimize_nelder_mead(best_points, deadline - 5, maxiter=800)
        opt_radius = _covering_radius(optimized)
        if opt_radius < best_radius:
            best_points = optimized
            best_radius = opt_radius
    
    # 4. Multiple restarts with perturbations if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        np.random.seed(42)
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            # Different perturbation strategies
            if restart % 3 == 0:
                # Small Gaussian noise
                noise = np.random.normal(0, 0.05, size=best_points.shape)
            elif restart % 3 == 1:
                # Larger noise
                noise = np.random.normal(0, 0.15, size=best_points.shape)
            else:
                # Random rotation
                theta = np.random.uniform(0, 2 * np.pi)
                axis = np.random.normal(size=3)
                axis /= np.linalg.norm(axis)
                # Rodrigues' rotation formula
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
                R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
                noise = best_points @ R.T - best_points
            
            perturbed = best_points + noise
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            perturbed /= norms
            
            # Quick relaxation
            relaxed = _repulsive_relaxation(perturbed, steps=500, alpha=0.03, use_density_scaling=True)
            rad = _covering_radius(relaxed)
            
            if rad < best_radius:
                best_points = relaxed
                best_radius = rad
                
                # Refine with Nelder-Mead if we have time
                if time.time() < deadline - 15:
                    optimized = _local_optimize_nelder_mead(best_points, deadline - 5, maxiter=300)
                    opt_radius = _covering_radius(optimized)
                    if opt_radius < best_radius:
                        best_points = optimized
                        best_radius = opt_radius
    
    # Final check: ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points /= norms
    
    return best_points
