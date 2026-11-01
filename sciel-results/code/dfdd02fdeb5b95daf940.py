import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _compute_hull_data(points):
    """Compute Delaunay triangulation and facet circumcenters/radii."""
    try:
        hull = Delaunay(points)
        simplices = hull.simplices
        n_faces = len(simplices)
        
        # Compute circumcenters and radii for each triangle
        circumcenters = np.zeros((n_faces, 3))
        radii = np.zeros(n_faces)
        
        for i, s in enumerate(simplices):
            a, b, c = points[s[0]], points[s[1]], points[s[2]]
            # Circumcenter vector for spherical triangle
            v = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                circumcenters[i] = (a + b + c) / 3.0
                radii[i] = np.pi
                continue
            m = v / norm_v
            circumcenters[i] = m
            d_a = np.arccos(np.clip(np.dot(m, a), -1.0, 1.0))
            d_b = np.arccos(np.clip(np.dot(m, b), -1.0, 1.0))
            d_c = np.arccos(np.clip(np.dot(m, c), -1.0, 1.0))
            radii[i] = np.max([d_a, d_b, d_c])
        
        return simplices, circumcenters, radii
    except Exception:
        return None, None, None

def _covering_radius(points):
    """Compute the covering radius of points on unit sphere."""
    if not np.all(np.isfinite(points)):
        return np.pi
    
    pts = _normalize(points)
    simplices, circumcenters, radii = _compute_hull_data(pts)
    
    if simplices is None or len(radii) == 0:
        return np.pi
    
    return np.max(radii)

def _lloyd_step(points, max_iter=50, tol=1e-8):
    """
    Lloyd's algorithm for spherical centroidal Voronoi tessellation.
    Each point is moved to the centroid of its Voronoi cell.
    """
    pts = _normalize(points.copy())
    
    for iteration in range(max_iter):
        # Compute Delaunay triangulation
        try:
            hull = Delaunay(pts)
            simplices = hull.simplices
        except Exception:
            break
        
        n_points = len(pts)
        
        # For each point, compute the weighted centroid of its Voronoi cell
        # The Voronoi cell of point i is the union of triangles (i, j, k) 
        # where the circumcenter is on the same side as i
        # We use the dual relationship: each triangle (i,j,k) contributes 
        # its circumcenter to the Voronoi cells of i, j, k with weights
        
        # Alternative approach: use the fact that in a CVT, each point should be
        # at the spherical centroid of its Voronoi region.
        # We approximate by moving each point toward the average of adjacent circumcenters.
        
        # For each vertex, collect the circumcenters of triangles containing that vertex
        vertex_circumcenters = [[] for _ in range(n_points)]
        
        for s in simplices:
            a, b, c = s[0], s[1], s[2]
            va, vb, vc = pts[a], pts[b], pts[c]
            v = np.cross(va, vb) + np.cross(vb, vc) + np.cross(vc, va)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                cc = (va + vb + vc) / 3.0
            else:
                cc = v / norm_v
            
            # Add this circumcenter to the Voronoi cells of a, b, c
            vertex_circumcenters[a].append(cc)
            vertex_circumcenters[b].append(cc)
            vertex_circumcenters[c].append(cc)
        
        # Compute new positions as spherical centroids of Voronoi cells
        new_pts = np.zeros_like(pts)
        for i in range(n_points):
            if len(vertex_circumcenters[i]) == 0:
                new_pts[i] = pts[i]
            else:
                # Weighted average: weight by solid angle (approximate by 1 for simplicity)
                # Better: weight by the area of the triangle's contribution
                avg = np.mean(vertex_circumcenters[i], axis=0)
                norm_avg = np.linalg.norm(avg)
                if norm_avg > 1e-10:
                    new_pts[i] = avg / norm_avg
                else:
                    new_pts[i] = pts[i]
        
        # Check convergence
        max_move = np.max(np.linalg.norm(new_pts - pts, axis=1))
        pts = _normalize(new_pts)
        
        if max_move < tol:
            break
    
    return pts

def _soft_max_objective(x, beta):
    """Soft-max covering radius objective with penalty for hemisphere."""
    p = x.reshape(-1, 3)
    p = _normalize(p)
    
    # Hemisphere penalty
    centroid = np.mean(p, axis=0)
    c_norm = np.linalg.norm(centroid)
    penalty = 0.0
    if c_norm > 0.4:
        penalty = (c_norm - 0.4) ** 2 * 5.0
    
    try:
        hull = Delaunay(p)
        simplices = hull.simplices
        radii = []
        
        for s in simplices:
            a, b, c = p[s[0]], p[s[1]], p[s[2]]
            v = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                radii.append(np.pi)
                continue
            m = v / norm_v
            d_a = np.arccos(np.clip(np.dot(m, a), -1.0, 1.0))
            d_b = np.arccos(np.clip(np.dot(m, b), -1.0, 1.0))
            d_c = np.arccos(np.clip(np.dot(m, c), -1.0, 1.0))
            radii.append(np.max([d_a, d_b, d_c]))
        
        if not radii:
            return penalty + np.pi
        
        radii = np.array(radii)
        max_r = np.max(radii)
        if beta > 0:
            soft_max = max_r + np.log(np.sum(np.exp(beta * (radii - max_r)))) / beta
        else:
            soft_max = max_r
        
        return penalty + soft_max
    except Exception:
        return penalty + np.pi

def _fibonacci_spiral(n, offset=0.0):
    """Generate points using Fibonacci spiral with optional offset."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere_points(n, seed=42):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s limit
    
    best_pts = None
    best_score = np.pi
    
    def try_improve(pts):
        nonlocal best_pts, best_score
        pts = _normalize(pts.copy())
        score = _covering_radius(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
        return score
    
    # Phase 1: Multi-start with Lloyd's algorithm
    n_starts = 8
    for start_idx in range(n_starts):
        if time.time() > deadline - 30:  # Keep 30s for final polish
            break
        
        if start_idx == 0:
            pts = _fibonacci_spiral(N)
        else:
            pts = _random_sphere_points(N, seed=100 + start_idx * 7)
        
        # Apply Lloyd's algorithm
        lloyd_pts = _lloyd_step(pts, max_iter=100, tol=1e-9)
        try_improve(lloyd_pts)
        
        # If time allows, do a second round of Lloyd from a perturbed version
        if time.time() < deadline - 20:
            rng = np.random.RandomState(200 + start_idx)
            perturbed = _normalize(lloyd_pts + rng.randn(N, 3) * 0.05)
            lloyd_pts2 = _lloyd_step(perturbed, max_iter=50, tol=1e-9)
            try_improve(lloyd_pts2)
    
    # Phase 2: L-BFGS polishing from best Lloyd result
    if best_pts is not None and time.time() < deadline - 10:
        # Try different beta values for soft-max
        for beta in [10.0, 20.0, 50.0]:
            if time.time() > deadline - 5:
                break
            
            x0 = best_pts.flatten()
            
            try:
                res = minimize(
                    _soft_max_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 100}
                )
                
                if res.fun < best_score:
                    new_pts = _normalize(res.x.reshape(-1, 3))
                    try_improve(new_pts)
            except Exception:
                pass
    
    # Phase 3: Final local search - perturb and Lloyd
    if best_pts is not None and time.time() < deadline - 5:
        rng = np.random.RandomState(12345)
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            
            # Small perturbation
            perturbed = _normalize(best_pts + rng.randn(N, 3) * 0.02)
            lloyd_pts = _lloyd_step(perturbed, max_iter=30, tol=1e-9)
            try_improve(lloyd_pts)
    
    # Return best found
    if best_pts is None:
        # Fallback: Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
