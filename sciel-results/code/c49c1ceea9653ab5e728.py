"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius_fast(points):
    """
    Compute the covering radius in degrees for points on the unit sphere.
    Uses vectorized circumradius computation for triangular facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_circumradius = 0.0
        
        # Handle triangular facets (most common in 3D convex hull)
        tri_facets = [f for f in facets if len(f) == 3]
        
        if len(tri_facets) > 0:
            tri_facets = np.array(tri_facets)
            a = points[tri_facets[:, 0]]
            b = points[tri_facets[:, 1]]
            c = points[tri_facets[:, 2]]
            
            # Compute cross products
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            
            # Triple product: a . (b x c)
            triple = np.einsum('ij,ij->i', a, np.cross(b, c))
            denom = 4.0 * np.abs(triple)
            
            # Avoid division by zero
            valid = denom > 1e-15
            if np.any(valid):
                norms_ab = np.linalg.norm(ab[valid], axis=1)
                norms_bc = np.linalg.norm(bc[valid], axis=1)
                norms_ca = np.linalg.norm(ca[valid], axis=1)
                
                sin_R = (norms_ab * norms_bc * norms_ca) / denom[valid]
                sin_R = np.clip(sin_R, 0.0, 1.0)
                R = np.arcsin(sin_R)
                max_circumradius = max(max_circumradius, np.max(R))
        
        # Handle non-triangular facets (rare, but possible)
        non_tri_facets = [f for f in facets if len(f) != 3]
        for simplex in non_tri_facets:
            pts = points[simplex]
            n_pts = len(pts)
            if n_pts < 3:
                continue
            if n_pts == 4:
                a, b, c, d = pts[0], pts[1], pts[2], pts[3]
                v1 = a - b
                v2 = a - c
                v3 = a - d
                x = np.cross(v1, np.cross(v2, v3))
                nx = np.linalg.norm(x)
                if nx < 1e-15:
                    continue
                x = x / nx
                cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
                R = np.arccos(cos_R)
                max_circumradius = max(max_circumradius, R)
            else:
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-15:
                    continue
                centroid = centroid / nc
                for p in pts:
                    cos_R = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    R = np.arccos(cos_R)
                    max_circumradius = max(max_circumradius, R)
        
        return np.degrees(max_circumradius)
    except Exception:
        return np.inf

def _energy_and_grad(points, alpha=2.0):
    """
    Compute repulsive energy and gradient on the sphere.
    Energy: sum over pairs of 1 / (1 - cos(theta))^alpha
    This penalizes close points heavily.
    """
    # points: (N, 3)
    # Compute pairwise dot products
    dots = points @ points.T
    # Mask diagonal
    np.fill_diagonal(dots, 1.0)
    
    # 1 - dots is proportional to chord length squared / 2
    # We want to maximize minimum distance, so minimize energy of close points
    # Let u = 1 - dots. Small u means close points.
    # Potential V(u) = u^{-alpha}
    # dV/du = -alpha * u^{-alpha-1}
    # du/dp_i = -2 * p_j (for i != j)
    # grad_i = sum_{j != i} dV/du * du/dp_i = sum_{j != i} (-alpha * u^{-alpha-1}) * (-2 * p_j)
    # grad_i = 2 * alpha * sum_{j != i} (1 - dots[i,j])^{-alpha-1} * p_j
    
    u = 1.0 - dots
    # Clip to avoid division by zero if points coincide
    u = np.clip(u, 1e-10, None)
    
    weights = u ** (-alpha - 1.0)
    np.fill_diagonal(weights, 0.0)
    
    # grad_i = 2 * alpha * (weights @ points)
    grad = 2.0 * alpha * (weights @ points)
    
    # Project gradient onto tangent space
    # grad_tangent = grad - (grad . p) * p
    grad_proj = np.einsum('ij,ij->i', grad, points)
    grad_tangent = grad - grad_proj[:, np.newaxis] * points
    
    energy = np.sum(u ** (-alpha))
    return energy, grad_tangent

def _optimize_energy(points, max_time=10.0):
    """
    Gradient descent on the sphere with adaptive step size.
    """
    start_time = time.time()
    current = points.copy()
    energy, grad = _energy_and_grad(current)
    
    step_size = 0.01
    patience = 0
    best_energy = energy
    best_points = current.copy()
    
    while time.time() - start_time < max_time:
        # Normalize step direction
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
            
        direction = -grad / grad_norm
        
        # Project direction to tangent space (it should already be, but ensure)
        proj = np.einsum('ij,ij->i', direction, current)
        direction = direction - proj[:, np.newaxis] * current
        
        # Update
        candidate = current + step_size * direction
        candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
        
        new_energy, new_grad = _energy_and_grad(candidate)
        
        if new_energy < energy:
            current = candidate
            energy = new_energy
            grad = new_grad
            step_size *= 1.05 # Increase step slightly
            patience = 0
            if energy < best_energy:
                best_energy = energy
                best_points = current.copy()
        else:
            step_size *= 0.5
            patience += 1
            if patience > 100:
                break
                
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    np.random.seed(42)
    
    # 1. Generate a good initial configuration using Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = fib_points
    best_radius = _covering_radius_fast(best_points)
    
    # 2. Run energy minimization from Fibonacci start
    opt_points = _optimize_energy(fib_points, max_time=20.0)
    opt_radius = _covering_radius_fast(opt_points)
    if opt_radius < best_radius:
        best_points = opt_points
        best_radius = opt_radius
        
    # 3. Try a few random restarts if time permits
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # Random start
        rand_points = np.random.randn(N, 3)
        rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
        
        # Optimize
        opt_points_r = _optimize_energy(rand_points, max_time=15.0)
        opt_radius_r = _covering_radius_fast(opt_points_r)
        
        if opt_radius_r < best_radius:
            best_points = opt_points_r
            best_radius = opt_radius_r
            
    # 4. Final polish on the best configuration found
    if time.time() < deadline - 5.0:
        final_points = _optimize_energy(best_points, max_time=10.0)
        final_radius = _covering_radius_fast(final_points)
        if final_radius < best_radius:
            best_points = final_points
            best_radius = final_radius
            
    return best_points
# EVOLVE-BLOCK-END
