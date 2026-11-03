"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull approach."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: grid
        return _covering_radius_grid(pts)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter of spherical triangle a,b,c
        # u is parallel to (a-b) x (b-c)
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            # Degenerate
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            max_cr = max(max_cr, cr)
            continue
        
        u = v / norm_v
        
        # Determine correct orientation
        # The circumcenter should be on the side of the face away from the origin?
        # No, the circumcenter of the Voronoi cell is the point on the sphere equidistant to a,b,c.
        # There are two such points: u and -u.
        # The one that is a vertex of the Voronoi diagram is the one "outside" the triangle relative to the other points?
        # Actually, for a convex hull face, the outward normal n points away from the origin.
        # The Voronoi vertex corresponding to this face is the point on the sphere that is furthest from the origin in the direction of the face? No.
        
        # Let's check the dot product with the face normal.
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        # Ensure n points away from origin (origin is inside hull)
        if np.dot(n, a) < 0:
            n = -n
            
        # The Voronoi vertex is the one that aligns with the outward normal?
        # If we have a face on the convex hull, the "hole" is on the outside.
        # The circumcenter of the spherical triangle formed by a,b,c that is part of the Voronoi diagram
        # is the one that is "farther" from the interior.
        # Let's test both u and -u. The correct one is the one that is NOT inside the convex hull of the other points?
        # Simpler heuristic: The circumcenter of the facet in the convex hull context is the point c such that
        # c is the intersection of the sphere and the plane through a,b,c? No, that's the circumcenter of the planar triangle.
        # On the sphere, the circumcenter is the normalized vector of the solution to A x = b.
        
        # Let's just compute the distance for both u and -u and pick the one that is "valid".
        # A valid Voronoi vertex must be closer to a,b,c than to any other point.
        # But for the covering radius calculation, we just need the circumradius of the cell.
        # The cell is the set of points closer to a,b,c than others.
        # The circumradius is the distance from the circumcenter to the vertices.
        
        # Let's use the property: The circumcenter u satisfies u·a = u·b = u·c.
        # We have two candidates u and -u.
        # The one that is part of the Voronoi diagram is the one that is "exposed".
        # For a face of the convex hull, the corresponding Voronoi cell vertex is the one on the "outside".
        # Since the hull contains the origin, the "outside" of the face is the direction of the normal n.
        # So we pick u if dot(u, n) > 0, else -u.
        
        if np.dot(u, n) > 0:
            circum_center = u
        else:
            circum_center = -u
            
        # Calculate circumradius
        d_a = np.arccos(np.clip(np.dot(circum_center, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(circum_center, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(circum_center, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        
        max_cr = max(max_cr, cr)
        
    return max_cr

def _covering_radius_grid(pts, n_grid=100):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_facet_shrink(pts, max_iter=2000, step_size=0.1):
    """
    Greedy optimization: Find the facet with the largest circumradius,
    and move its vertices towards the circumcenter of that facet.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # Find facet with max circumradius
        max_cr = 0.0
        max_facet = None
        max_circum_center = None
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            u = v / norm_v
            
            n_face = np.cross(b - a, c - a)
            n_face = n_face / np.linalg.norm(n_face)
            if np.dot(n_face, a) < 0:
                n_face = -n_face
                
            if np.dot(u, n_face) > 0:
                cc = u
            else:
                cc = -u
                
            d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
            cr = max(d_a, d_b, d_c)
            
            if cr > max_cr:
                max_cr = cr
                max_facet = simplex
                max_circum_center = cc
                
        if max_facet is None:
            break
            
        # Move vertices of the max facet towards the circumcenter
        # Use a step size that is a fraction of the angular distance
        # To avoid overshooting, we project the movement onto the tangent plane
        for idx in max_facet:
            p = pts[idx]
            # Vector from p to cc (chordal)
            vec = max_circum_center - p
            # Project onto tangent space at p: vec - (vec·p)p
            vec_tan = vec - np.dot(vec, p) * p
            # Normalize and scale
            norm_vec = np.linalg.norm(vec_tan)
            if norm_vec > 1e-10:
                # Step size: move a fraction of the way
                # Adaptive step: smaller if cr is small
                step = step_size * min(1.0, max_cr)
                pts[idx] = p + step * (vec_tan / norm_vec)
                
        pts = _normalize(pts)
        
        # Convergence check
        if max_cr < 1e-6:
            break
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    np.random.seed(42)
    
    # Multiple restarts with perturbations
    for restart in range(50):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = _fibonacci_points(N)
        else:
            # Perturb best solution
            noise = 0.5 * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
            
        # Optimize
        pts = _optimize_facet_shrink(pts, max_iter=500, step_size=0.5)
        
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
        # Check if we are in a hemisphere (cr > pi/2)
        # If so, we need to force expansion
        if best_cr > np.pi / 2:
             # Apply a strong repulsive force to push points apart
             for _ in range(100):
                 forces = np.zeros_like(best_pts)
                 for i in range(N):
                     for j in range(N):
                         if i == j: continue
                         d = np.arccos(np.clip(np.dot(best_pts[i], best_pts[j]), -1, 1))
                         if d < 1e-6: continue
                         f = (best_pts[j] - best_pts[i]) / (d * d)
                         forces[i] += f
                 best_pts = best_pts + 0.1 * forces
                 best_pts = _normalize(best_pts)
             best_cr = _covering_radius(best_pts)
             if best_cr < cr:
                 cr = best_cr
                 pts = best_pts.copy()

    return _normalize(best_pts)
# EVOLVE-BLOCK-END
