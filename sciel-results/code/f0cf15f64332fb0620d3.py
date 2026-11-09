"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius_degrees(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # invalid: return large value
    
    # Get facet normals and planes
    # For each facet, find the circumcenter (point on sphere equidistant to facet vertices)
    # The covering radius is the max over facets of the angular distance from facet center to vertices
    
    max_radius = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        # For a spherical triangle, find the circumcenter
        # The circumcenter of a spherical triangle is the point on the sphere equidistant to all 3 vertices
        # It lies in the plane spanned by the 3 vertices, so circumcenter = normalize(a + b + c) for equilateral
        # For general triangles, we need to solve
        
        # Method: find point x on sphere such that |x - a| = |x - b| = |x - c|
        # This means x · a = x · b = x · c (since |x|=|a|=|b|=|c|=1)
        # So x is orthogonal to (a-b) and (a-c), and on the unit sphere
        
        a, b, c = verts[0], verts[1], verts[2]
        # x is parallel to cross(a-b, a-c) or its negative
        v1 = a - b
        v2 = a - c
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # Degenerate facet
            continue
        n = n / n_norm
        
        # Check which direction gives the circumcenter (the one that is "outside" the triangle)
        # The circumcenter should be on the same side as the facet normal
        # Actually, we need to check both n and -n
        for sign in [1.0, -1.0]:
            x = sign * n
            # Verify: x · a should equal x · b should equal x · c
            da = np.dot(x, a)
            db = np.dot(x, b)
            dc = np.dot(x, c)
            if abs(da - db) > 1e-8 or abs(da - dc) > 1e-8:
                continue
            # Angular distance from x to any vertex
            angle = np.arccos(np.clip(da, -1.0, 1.0))
            if angle > max_radius:
                max_radius = angle
    
    return np.degrees(max_radius)

def _objective_and_gradient(pts):
    """Compute soft-max weighted covering radius objective and gradient."""
    pts = _normalize(pts)
    
    # Compute all pairwise dot products
    dots = pts @ pts.T
    np.fill_diagonal(dots, -2.0)  # Exclude self
    
    # For each point, find its 3 nearest neighbors (by angular distance)
    # These define the local Voronoi cell
    # The covering radius is related to the largest empty circle in the Voronoi diagram
    
    # Simpler approach: use the convex hull method but with soft weighting
    # Let's use a different approach: minimize the max over all pairs of the circumradius
    
    # Actually, let's use the facet-based approach with soft-max
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1.0, np.zeros_like(pts)
    
    # Compute circumradii for each facet
    radii = np.zeros(len(hull.simplices))
    for i, simplex in enumerate(hull.simplices):
        verts = pts[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        v1 = a - b
        v2 = a - c
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            radii[i] = np.pi
            continue
        n = n / n_norm
        
        best_angle = 0.0
        for sign in [1.0, -1.0]:
            x = sign * n
            da = np.dot(x, a)
            db = np.dot(x, b)
            dc = np.dot(x, c)
            if abs(da - db) > 1e-6 or abs(da - dc) > 1e-6:
                continue
            angle = np.arccos(np.clip(da, -1.0, 1.0))
            best_angle = max(best_angle, angle)
        radii[i] = best_angle
    
    # Soft-max to get a smooth objective
    beta = 20.0
    exp_radii = np.exp(beta * radii)
    obj = np.log(np.sum(exp_radii)) / beta
    
    # Gradient via finite differences would be too slow
    # Use analytical gradient: for each facet, the gradient points away from the circumcenter
    # Sum weighted contributions
    
    grad = np.zeros_like(pts)
    weights = exp_radii / np.sum(exp_radii)
    
    for i, simplex in enumerate(hull.simplices):
        w = weights[i]
        if w < 1e-10:
            continue
        verts = pts[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        v1 = a - b
        v2 = a - c
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        for sign in [1.0, -1.0]:
            x = sign * n
            da = np.dot(x, a)
            db = np.dot(x, b)
            dc = np.dot(x, c)
            if abs(da - db) > 1e-6 or abs(da - dc) > 1e-6:
                continue
            # Gradient: move vertices away from x to increase the circumradius
            # The circumcenter x should move toward the facet, so vertices should move away
            for j, vert in enumerate(simplex):
                # Gradient of arccos(x·a) w.r.t. a is -x / sqrt(1-(x·a)^2)
                cos_val = np.dot(x, vert)
                sin_val = np.sqrt(max(1e-12, 1.0 - cos_val * cos_val))
                grad[vert] += w * beta * (-x / sin_val)
            break
    
    return obj, grad

def _gradient_descent(pts, max_iters=500, lr=0.01):
    """Perform gradient descent to minimize covering radius."""
    pts = _normalize(pts).copy()
    best_pts = pts.copy()
    best_obj = _covering_radius_degrees(pts)
    
    for it in range(max_iters):
        obj, grad = _objective_and_gradient(pts)
        
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
        
        # Gradient step
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            lr *= 0.8
    
    return best_pts, best_obj

def _random_start():
    """Generate a random point on the unit sphere."""
    v = np.random.randn(N, 3)
    return _normalize(v)

def _fibonacci_start():
    """Golden-angle spiral starting points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin
    
    best_pts = _fibonacci_start()
    best_score = _covering_radius_degrees(best_pts)
    
    # Try multiple random restarts
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = _fibonacci_start()
        else:
            pts = _random_start()
        
        # Gradient descent
        pts, obj = _gradient_descent(pts, max_iters=200, lr=0.05)
        
        score = _covering_radius_degrees(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Polish with smaller steps
        if time.time() > deadline:
            break
        pts, obj = _gradient_descent(pts, max_iters=100, lr=0.01)
        score = _covering_radius_degrees(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
