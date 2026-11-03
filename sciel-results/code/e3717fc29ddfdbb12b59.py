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
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            max_cr = max(max_cr, cr)
            continue
        
        u = v / norm_v
        
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        if np.dot(n, a) < 0:
            n = -n
            
        if np.dot(u, n) > 0:
            circum_center = u
        else:
            circum_center = -u
            
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

def _spherical_lloyd(pts, max_iter=1000, tol=1e-8):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves each point to the centroid of its Voronoi cell.
    The Voronoi cell vertices are the outward normals of the convex hull facets incident to the point.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # Map point index to list of incident facets
        incident_facets = [[] for _ in range(n)]
        
        for simplex in hull.simplices:
            # Compute outward normal for this facet
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            normal = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            # Ensure normal points away from origin
            if np.dot(normal, a) < 0:
                normal = -normal
                
            for idx in simplex:
                incident_facets[idx].append(normal)
                
        # Compute new positions
        new_pts = np.zeros_like(pts)
        max_move = 0.0
        
        for i in range(n):
            if len(incident_facets[i]) == 0:
                new_pts[i] = pts[i]
                continue
                
            # Centroid is the normalized sum of the incident facet normals
            # This is the "vertex centroid" of the dual cell (Voronoi cell)
            centroid = np.sum(incident_facets[i], axis=0)
            norm_c = np.linalg.norm(centroid)
            
            if norm_c < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = centroid / norm_c
                
            # Track max movement for convergence
            move = np.linalg.norm(new_pts[i] - pts[i])
            max_move = max(max_move, move)
            
        pts = _normalize(new_pts)
        
        if max_move < tol:
            break
            
    return pts

def _optimize_facet_shrink(pts, max_iter=500, step_size=0.1):
    """
    Greedy optimization: Find the facet with the largest circumradius,
    and move its vertices towards the circumcenter of that facet.
    Used as a final polish step.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
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
            
        for idx in max_facet:
            p = pts[idx]
            vec = max_circum_center - p
            vec_tan = vec - np.dot(vec, p) * p
            norm_vec = np.linalg.norm(vec_tan)
            if norm_vec > 1e-10:
                step = step_size * min(1.0, max_cr)
                pts[idx] = p + step * (vec_tan / norm_vec)
                
        pts = _normalize(pts)
        
        if max_cr < 1e-6:
            break
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    np.random.seed(42)
    
    # Strategy: 
    # 1. Start with Fibonacci.
    # 2. Run Lloyd's algorithm to find a good local minimum.
    # 3. Polish with facet shrink.
    # 4. Repeat with perturbations.
    
    for restart in range(30):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = _fibonacci_points(N)
        else:
            # Perturb best solution
            # Use smaller noise for later restarts to refine
            noise_scale = 0.3 / (1 + restart * 0.1)
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
            
        # Run Lloyd's
        # We allow enough iterations for Lloyd's to converge
        # Since N=50 is small, this is fast.
        pts = _spherical_lloyd(pts, max_iter=500)
        
        # Polish with facet shrink to minimize max circumradius specifically
        pts = _optimize_facet_shrink(pts, max_iter=200, step_size=0.1)
        
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
