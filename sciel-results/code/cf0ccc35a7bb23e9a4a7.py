"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def _covering_radius(pts):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This equals the maximum circumradius of the facets
    of the convex hull (for points in general position on the sphere).
    """
    try:
        hull = ConvexHull(pts)
        max_rad = 0.0
        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Circumcenter of triangle on unit sphere
            # The circumradius of a spherical triangle is the angular distance from the
            # circumcenter to each vertex. For a triangle on the unit sphere, we can
            # compute the circumcenter as the normalized cross product of the differences.
            
            # More robust: find the point on the sphere equidistant from a, b, c.
            # This is the intersection of the sphere with the plane through a, b, c 
            # (if the triangle is not degenerate), but actually the circumcenter on the
            # sphere is the normalized vector that is equidistant from all three.
            
            # Method: The circumcenter p satisfies |p-a|^2 = |p-b|^2 = |p-c|^2 with |p|=1.
            # |p|^2 - 2p.a + |a|^2 = |p|^2 - 2p.b + |b|^2
            # Since |a|=|b|=|c|=1, this simplifies to p.a = p.b = p.c.
            # So p is in the null space of [a-b; b-c] and |p|=1.
            
            # Let's use the formula for circumcenter of a spherical triangle.
            # For a triangle with vertices a, b, c on the unit sphere, the circumcenter
            # can be computed as the normalized vector:
            # p = (|a-b|^2 * (b x c) + |b-c|^2 * (c x a) + |c-a|^2 * (a x b)) / |...|
            # This is the planar circumcenter formula adapted. Let me verify.
            
            # Actually, the standard formula for the circumcenter of a spherical triangle
            # with vertices a, b, c (unit vectors) is:
            # p = normalize( (|a-b|^2 * (b x c) + |b-c|^2 * (c x a) + |c-a|^2 * (a x b)) )
            # But I need to be careful about orientation.
            
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)
            
            # Cross products
            bc_cross = np.cross(b, c)
            ca_cross = np.cross(c, a)
            ab_cross = np.cross(a, b)
            
            p = ab * ab * bc_cross + bc * bc * ca_cross + ca * ca * ab_cross
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-12:
                # Degenerate triangle, use the largest edge
                continue
            p = p / p_norm
            
            # The circumradius is the angular distance from p to any vertex
            # Use a for stability
            dot_val = np.clip(np.dot(p, a), -1.0, 1.0)
            rad = np.arccos(dot_val)
            if rad > max_rad:
                max_rad = rad
        return np.degrees(max_rad)
    except Exception:
        return 180.0  # Invalid configuration

def _objective(pts_flat):
    """Objective function for optimization: the covering radius in radians."""
    pts = pts_flat.reshape(N, 3)
    # Normalize each point
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    return _covering_radius(pts)

def _initial_points_fibonacci():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_points_random(seed=42):
    """Generate random initial points."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _local_optimization(pts, max_iters=200, tol=1e-10, time_limit=None):
    """
    Local optimization to minimize the covering radius.
    Uses subgradient method on the nonsmooth objective.
    """
    start_time = time.time()
    pts = pts.copy()
    best_pts = pts.copy()
    best_val = _covering_radius(pts)
    
    lr = 0.05
    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        if best_val < tol:
            break
            
        # Compute subgradient
        # The subgradient of the max circumradius is the circumcenter of the facet
        # that achieves the maximum, projected to the tangent space.
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        max_rad = 0.0
        max_facet_idx = -1
        max_circumcenter = None
        for fi, simplex in enumerate(hull.simplices):
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)
            bc_cross = np.cross(b, c)
            ca_cross = np.cross(c, a)
            ab_cross = np.cross(a, b)
            p = ab * ab * bc_cross + bc * bc * ca_cross + ca * ca * ab_cross
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-12:
                continue
            p = p / p_norm
            dot_val = np.clip(np.dot(p, a), -1.0, 1.0)
            rad = np.arccos(dot_val)
            if rad > max_rad:
                max_rad = rad
                max_facet_idx = fi
                max_circumcenter = p
                max_facet = simplex
                
        if max_circumcenter is None:
            break
            
        # The subgradient for each vertex in the max facet is the circumcenter
        # projected to the tangent space at that vertex.
        grad = np.zeros_like(pts)
        for idx in max_facet:
            v = pts[idx]
            # Project circumcenter to tangent space at v
            grad[idx] = max_circumcenter - np.dot(max_circumcenter, v) * v
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad /= grad_norm
        
        # Update: move vertices in the direction that decreases the covering radius
        # For the vertices in the max facet, move them away from the circumcenter
        # to shrink the facet.
        for idx in max_facet:
            v = pts[idx]
            # Move v away from circumcenter in tangent space
            direction = v - np.dot(max_circumcenter, v) * max_circumcenter
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-12:
                direction /= direction_norm
                pts[idx] = pts[idx] + lr * direction
            else:
                # Perturb slightly
                rng = np.random.RandomState(int(it) + int(idx))
                pert = rng.randn(3)
                pert = pert - np.dot(pert, v) * v
                pert_norm = np.linalg.norm(pert)
                if pert_norm > 1e-12:
                    pts[idx] = pts[idx] + lr * pert / pert_norm
        
        # Renormalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        
        # Check improvement
        current_val = _covering_radius(pts)
        if current_val < best_val:
            best_val = current_val
            best_pts = pts.copy()
        else:
            # Revert and reduce learning rate
            pts = best_pts.copy()
            lr *= 0.9
            if lr < 1e-8:
                break
    
    return best_pts, best_val

def _polish(pts, max_iters=100, time_limit=None):
    """
    Final polish using a smooth surrogate.
    We minimize a soft-max of the circumradii.
    """
    start_time = time.time()
    pts = pts.copy()
    best_pts = pts.copy()
    best_val = _covering_radius(pts)
    
    def soft_max_obj(pts_flat, beta=50.0):
        pts = pts_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        try:
            hull = ConvexHull(pts)
        except Exception:
            return 1.0
        radii = []
        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)
            bc_cross = np.cross(b, c)
            ca_cross = np.cross(c, a)
            ab_cross = np.cross(a, b)
            p = ab * ab * bc_cross + bc * bc * ca_cross + ca * ca * ab_cross
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-12:
                continue
            p = p / p_norm
            dot_val = np.clip(np.dot(p, a), -1.0, 1.0)
            rad = np.arccos(dot_val)
            radii.append(rad)
        if not radii:
            return 1.0
        radii = np.array(radii)
        # Soft max
        m = np.max(radii)
        log_sum = np.log(np.sum(np.exp(beta * (radii - m))))
        return m + np.log(radii.size) / beta + log_sum / beta
    
    # Use L-BFGS-B on the flat representation
    x0 = pts.flatten()
    result = minimize(soft_max_obj, x0, method='L-BFGS-B', options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if result.success or result.fun < best_val:
        pts_opt = result.x.reshape(N, 3)
        norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts_opt = pts_opt / norms
        val = _covering_radius(pts_opt)
        if val < best_val:
            best_val = val
            best_pts = pts_opt
    
    return best_pts, best_val

def construct():
    deadline = 120.0  # seconds
    start_time = time.time()
    
    # Generate multiple initial configurations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_initial_points_fibonacci())
    
    # Random initializations with different seeds
    for seed in [1, 2, 3, 4, 5]:
        initializations.append(_initial_points_random(seed))
    
    best_pts = None
    best_val = 180.0
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() - start_time > deadline - 30:
            break
        
        # Local optimization
        remaining = deadline - 30 - (time.time() - start_time)
        if remaining < 5:
            break
        opt_pts, opt_val = _local_optimization(init_pts, max_iters=100, time_limit=remaining)
        
        if opt_val < best_val:
            best_val = opt_val
            best_pts = opt_pts.copy()
    
    # Final polish
    if best_pts is not None:
        remaining = deadline - 5 - (time.time() - start_time)
        if remaining > 2:
            polished_pts, polished_val = _polish(best_pts, max_iters=50, time_limit=remaining)
            if polished_val < best_val:
                best_val = polished_val
                best_pts = polished_pts
    
    if best_pts is None:
        best_pts = _initial_points_fibonacci()
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_pts = best_pts / norms
    
    return best_pts

# EVOLVE-BLOCK-END
