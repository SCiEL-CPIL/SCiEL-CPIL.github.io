"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull method."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        # Get the facets (simplices) of the convex hull
        # Each facet is a set of indices
        facets = hull.simplices
        
        # For each facet, compute the circumcenter and check if it's a valid cap
        max_radius = 0.0
        for simplex in facets:
            # Get the points defining this facet
            facet_pts = pts[simplex]
            
            # Compute the circumcenter of the simplex in 3D
            # For a triangle (simplex of size 3), we can compute it directly
            # For a tetrahedron (simplex of size 4), we need a different approach
            # But on a sphere, all facets should be triangles (for a convex polyhedron)
            
            if len(simplex) == 3:
                # Triangle circumcenter
                a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
                
                # The circumcenter of a triangle on the unit sphere
                # is the normalized cross product of (b-a) and (c-a)
                # Actually, for points on the unit sphere, the circumcenter
                # is in the direction of the normal to the plane through a,b,c
                # that points outward (away from origin)
                
                # Compute the normal to the plane
                v1 = b - a
                v2 = c - a
                normal = np.cross(v1, v2)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm
                
                # The circumcenter should be on the unit sphere in the direction
                # of the outward normal. But we need to check which direction is outward.
                # The midpoint of the triangle is (a+b+c)/3
                mid = (a + b + c) / 3.0
                # The outward normal should point away from the origin
                if np.dot(normal, mid) < 0:
                    normal = -normal
                
                # The circumradius is the angle between the circumcenter and any vertex
                # But we need to verify this is actually a covering point
                # i.e., the cap centered at 'normal' with radius = angle to vertices
                # should contain no other points
                
                cos_angles = np.clip(pts @ normal, -1.0, 1.0)
                angles = np.arccos(cos_angles)
                
                # The radius of this cap is the max angle to the facet vertices
                facet_angles = angles[simplex]
                cap_radius = np.max(facet_angles)
                
                # Check if any other point is outside this cap
                # (i.e., has angle > cap_radius to the circumcenter)
                if np.max(angles) > cap_radius + 1e-10:
                    # This facet's circumcenter is not the covering point
                    # because another point is farther
                    continue
                
                max_radius = max(max_radius, cap_radius)
            else:
                # For higher-dimensional simplices (shouldn't happen for 3D hull)
                # Just use the max distance to any vertex as a fallback
                # This is conservative
                pass
        
        # Also check the vertices themselves as potential covering centers
        # Actually, the covering radius is determined by the facets' circumcenters
        # that are "empty" (no other points in the cap)
        
        return max_radius
    except Exception:
        # Fallback: use the max nearest-neighbor distance
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        # This is not quite right, but as a fallback
        # The covering radius is at least the max nearest neighbor distance / 2
        # but this is a lower bound, not the actual covering radius
        # For a fallback, let's just return a large value to indicate failure
        return np.pi  # Very large, will be penalized

def _covering_radius_fast(pts):
    """Faster covering radius computation using Delaunay triangulation."""
    pts = _normalize(pts)
    try:
        # Use Delaunay triangulation on the sphere
        # The Delaunay triangulation of points on a sphere corresponds to
        # the convex hull facets
        tri = Delaunay(pts)
        max_radius = 0.0
        
        for simplex in tri.simplices:
            if len(simplex) != 3:
                continue
            facet_pts = pts[simplex]
            a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
            
            v1 = b - a
            v2 = c - a
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-12:
                continue
            normal = normal / normal_norm
            
            mid = (a + b + c) / 3.0
            if np.dot(normal, mid) < 0:
                normal = -normal
            
            cos_angles = np.clip(pts @ normal, -1.0, 1.0)
            angles = np.arccos(cos_angles)
            
            facet_angles = angles[simplex]
            cap_radius = np.max(facet_angles)
            
            if np.max(angles) > cap_radius + 1e-10:
                continue
            
            max_radius = max(max_radius, cap_radius)
        
        return max_radius
    except Exception:
        return np.pi

def _objective(pts_flat):
    """Objective function: covering radius."""
    pts = pts_flat.reshape(-1, 3)
    pts = _normalize(pts)
    return _covering_radius_fast(pts)

def _gradient_descent(pts, max_iter=1000, lr=0.01):
    """Gradient descent to minimize the covering radius."""
    pts = _normalize(pts)
    best_pts = pts.copy()
    best_obj = _covering_radius_fast(best_pts)
    
    for i in range(max_iter):
        # Compute subgradient
        # The covering radius is determined by the worst facet
        # We can approximate the gradient by perturbing each point
        
        # Use numerical gradient
        grad = np.zeros_like(pts)
        eps = 1e-5
        
        for j in range(len(pts)):
            for k in range(3):
                pts_plus = pts.copy()
                pts_plus[j, k] += eps
                pts_plus = _normalize(pts_plus)
                obj_plus = _covering_radius_fast(pts_plus)
                
                pts_minus = pts.copy()
                pts_minus[j, k] -= eps
                pts_minus = _normalize(pts_minus)
                obj_minus = _covering_radius_fast(pts_minus)
                
                grad[j, k] = (obj_plus - obj_minus) / (2 * eps)
        
        # Update
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        obj = _covering_radius_fast(pts)
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    return best_pts, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_score = _covering_radius_fast(best_pts)
    
    # Try multiple random restarts with local optimization
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random perturbation
        if restart == 0:
            current_pts = best_pts.copy()
        else:
            # Start from a random configuration
            current_pts = np.random.randn(N, 3)
            current_pts = _normalize(current_pts)
        
        # Local optimization using coordinate descent
        for coord_iter in range(500):
            if time.time() > deadline:
                break
            
            # For each point, try to move it to reduce the covering radius
            improved = False
            for j in range(N):
                if time.time() > deadline:
                    break
                
                # Try small perturbations in different directions
                best_local = current_pts[j].copy()
                best_local_obj = _covering_radius_fast(current_pts)
                
                # Generate candidate directions
                # 1. Random directions
                for _ in range(20):
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    
                    for step in [0.01, 0.05, 0.1, 0.2]:
                        new_pt = current_pts[j] + step * direction
                        new_pt = new_pt / np.linalg.norm(new_pt)
                        
                        test_pts = current_pts.copy()
                        test_pts[j] = new_pt
                        obj = _covering_radius_fast(test_pts)
                        
                        if obj < best_local_obj - 1e-10:
                            best_local_obj = obj
                            best_local = new_pt
                            improved = True
                
                current_pts[j] = best_local
            
            if not improved:
                break
        
        # Check if we improved
        score = _covering_radius_fast(current_pts)
        if score < best_score:
            best_score = score
            best_pts = current_pts.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        # Use scipy minimize with L-BFGS-B
        try:
            x0 = best_pts.flatten()
            result = minimize(_objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 100, 'ftol': 1e-12})
            if result.success:
                polished = _normalize(result.x.reshape(-1, 3))
                score = _covering_radius_fast(polished)
                if score < best_score:
                    best_score = score
                    best_pts = polished
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
