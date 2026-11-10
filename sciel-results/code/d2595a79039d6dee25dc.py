"""Rewritten program for covering the sphere with 50 equal caps."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize


def _covering_radius_degrees(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    # Ensure points are normalized
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Compute convex hull to get facets
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0  # Fallback if hull fails
    
    # For each facet, compute the circumcenter and circumradius
    # The covering radius is the max over all facets of the circumradius
    # (since each facet defines a Voronoi cell, and the farthest point in that cell
    # from its nearest point is the circumcenter of the facet's circumcircle)
    
    # Get facet indices
    facets = hull.simplices  # shape (n_facets, 3)
    
    max_circumradius = 0.0
    
    for facet in facets:
        p1, p2, p3 = pts[facet[0]], pts[facet[1]], pts[facet[2]]
        
        # Compute circumcenter of the triangle on the sphere
        # The circumcenter is the point equidistant from all three vertices
        # On the unit sphere, this is the normalized vector in the direction of
        # the cross products
        
        # Method: find the point c on the unit sphere such that c.p1 = c.p2 = c.p3
        # This means c is in the plane defined by the three points, and |c| = 1
        # The circumcenter direction is proportional to (p1 x p2) x (p2 x p3)
        # or equivalently, we can solve the linear system
        
        # Using the formula: circumcenter = (a*(b x c) + b*(c x a) + c*(a x b)) / |...|
        # where a, b, c are the points, and the scalar a = |b|^2|c|^2 - (b.c)^2, etc.
        
        a = p1
        b = p2
        c = p3
        
        # Compute the circumcenter using vector formula
        # cc = (|b|^2|c|^2 - (b·c)^2)(a) + (|c|^2|a|^2 - (c·a)^2)(b) + (|a|^2|b|^2 - (a·b)^2)(c)
        # All points are on unit sphere, so |a|=|b|=|c|=1
        ab = np.dot(a, b)
        bc = np.dot(b, c)
        ca = np.dot(c, a)
        
        coeff_a = 1.0 - bc * bc
        coeff_b = 1.0 - ca * ca
        coeff_c = 1.0 - ab * ab
        
        cc_vec = coeff_a * a + coeff_b * b + coeff_c * c
        cc_norm = np.linalg.norm(cc_vec)
        
        if cc_norm < 1e-10:
            continue
            
        cc = cc_vec / cc_norm
        
        # Check if cc is on the correct side (inside the cap)
        # The circumradius is the angle between cc and any of the vertices
        cos_theta = np.dot(cc, a)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Also check that cc is in the same hemisphere as the facet normal
        # The facet normal points outward
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        # The circumcenter should be on the same side as the facet normal
        # (i.e., the cap center)
        if np.dot(cc, n) < 0:
            cc = -cc
            cos_theta = -cos_theta
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
        
        max_circumradius = max(max_circumradius, theta)
    
    return np.degrees(max_circumradius)


def _repulsive_force(pts, k=5, strength=1.0):
    """
    Compute repulsive forces on each point from its k nearest neighbors.
    The force is scaled by the inverse of the local covering density.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Compute pairwise distances
    # dist[i, j] = angular distance between pts[i] and pts[j]
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(dots)
    
    np.fill_diagonal(dists, np.inf)
    
    for i in range(n):
        # Find k nearest neighbors
        nn_idx = np.argsort(dists[i])[:k]
        
        local_force = np.zeros(3)
        
        for j in nn_idx:
            if j == i:
                continue
            d = dists[i, j]
            if d < 1e-10:
                continue
            
            # Repulsive force direction: from j to i (push i away from j)
            # On the sphere, the direction is the tangent component of (pts[i] - pts[j])
            diff = pts[i] - pts[j]
            # Project onto tangent plane at pts[i]
            proj = diff - np.dot(diff, pts[i]) * pts[i]
            proj_norm = np.linalg.norm(proj)
            
            if proj_norm < 1e-10:
                continue
            
            # Force magnitude: strong repulsion from nearest neighbors,
            # scaled by inverse of local covering density
            # Local density ~ 1 / d^2 (area of cap)
            # Inverse density ~ d^2
            # But we want strong repulsion when points are close
            # Use 1/d^2 scaling for the force magnitude
            force_mag = strength / (d * d)
            
            local_force += force_mag * (proj / proj_norm)
        
        forces[i] = local_force
    
    return forces


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius
    using repulsive force-based dynamics.
    """
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Run repulsive dynamics
    # Use small steps to avoid instability
    dt = 0.01
    num_steps = 2000
    k = 6  # Number of nearest neighbors to consider
    
    best_pts = pts.copy()
    best_radius = _covering_radius_degrees(pts)
    
    for step in range(num_steps):
        # Decay step size
        current_dt = dt * (1.0 - step / num_steps)
        if current_dt < 0.001:
            current_dt = 0.001
        
        # Compute forces
        forces = _repulsive_force(pts, k=k, strength=1.0)
        
        # Update positions: move along the force direction on the sphere
        new_pts = pts.copy()
        for i in range(N):
            # Project force onto tangent plane
            f = forces[i]
            f_tangent = f - np.dot(f, pts[i]) * pts[i]
            
            # Move in the direction of the tangent force
            new_pts[i] = pts[i] + current_dt * f_tangent
            
            # Renormalize to unit sphere
            norm = np.linalg.norm(new_pts[i])
            if norm > 1e-10:
                new_pts[i] = new_pts[i] / norm
            else:
                new_pts[i] = pts[i]
        
        pts = new_pts
        
        # Check improvement
        if step % 100 == 0:
            current_radius = _covering_radius_degrees(pts)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
    
    # Final check
    final_radius = _covering_radius_degrees(pts)
    if final_radius < best_radius:
        best_pts = pts.copy()
    
    # Normalize final points
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
