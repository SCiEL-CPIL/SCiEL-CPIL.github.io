import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return points / norms

def _planar_circumcenter(a, b, c):
    # a, b, c are 3D points
    A = b - a
    B = c - a
    n = np.cross(A, B)
    norm_n_sq = np.dot(n, n)
    if norm_n_sq < 1e-12:
        return (a + b + c) / 3.0 # Degenerate, return centroid
    
    # Formula for circumcenter of triangle defined by a, b, c
    # C = a + (||B||^2 * (A x n) + ||A||^2 * (n x B)) / (2 * ||n||^2)
    term1 = np.dot(B, B) * np.cross(A, n)
    term2 = np.dot(A, A) * np.cross(n, B)
    C = a + (term1 + term2) / (2 * norm_n_sq)
    return C

def _covering_radius(points):
    """
    Computes the covering radius in radians.
    Returns (radius, worst_facet_indices)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.inf, []
    
    facets = hull.simplices
    max_rad = 0.0
    worst_idx = -1
    
    # Precompute circumcenters for all facets
    # To save time, we only need the max radius.
    # But for refinement, we need the circumcenter for each facet.
    # We will return the radius and the circumcenters if needed, 
    # but for the score, just radius.
    
    # Optimization: We can compute radius for each facet.
    # facets shape (M, 3)
    # points shape (N, 3)
    
    # Vectorized circumcenter calculation is complex due to cross products.
    # Loop is fine for N=50, M~100.
    
    radii = np.zeros(len(facets))
    for i, f in enumerate(facets):
        a = points[f[0]]
        b = points[f[1]]
        c = points[f[2]]
        
        C_p = _planar_circumcenter(a, b, c)
        norm_C = np.linalg.norm(C_p)
        if norm_C < 1e-12:
            radii[i] = np.pi # Degenerate
            continue
        C_s = C_p / norm_C
        # Distance is arccos(C_s . a)
        dot = np.dot(C_s, a)
        dot = np.clip(dot, -1.0, 1.0)
        radii[i] = np.arccos(dot)
        
        if radii[i] > max_rad:
            max_rad = radii[i]
            worst_idx = i
            
    return max_rad, worst_idx

def _refine(points, max_iters=100, eta=0.5):
    """
    Local refinement using Voronoi/Circumcenter logic.
    """
    points = _normalize(points.copy())
    best_points = points.copy()
    best_radius, _ = _covering_radius(points)
    
    for i in range(max_iters):
        # Check time
        if time.time() > _deadline:
            break
            
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        facets = hull.simplices
        
        # For each point, accumulate the average circumcenter of incident facets
        # We need to map point index to list of facets
        point_to_facets = [[] for _ in range(N)]
        for fi, f in enumerate(facets):
            for p_idx in f:
                point_to_facets[p_idx].append(fi)
                
        # Compute circumcenters for all facets once
        facet_centers = np.zeros((len(facets), 3))
        facet_radii = np.zeros(len(facets))
        
        for fi, f in enumerate(facets):
            a = points[f[0]]
            b = points[f[1]]
            c = points[f[2]]
            C_p = _planar_circumcenter(a, b, c)
            norm_C = np.linalg.norm(C_p)
            if norm_C < 1e-12:
                facet_centers[fi] = (a+b+c)/3.0
                facet_radii[fi] = 0
            else:
                C_s = C_p / norm_C
                facet_centers[fi] = C_s
                dot = np.dot(C_s, a)
                facet_radii[fi] = np.arccos(np.clip(dot, -1.0, 1.0))
                
        # Update points
        new_points = points.copy()
        for p_idx in range(N):
            if not point_to_facets[p_idx]:
                continue
            # Weight by radius? Or just average?
            # Moving towards the worst facet is better.
            # Let's move towards the circumcenter of the facet with the max radius incident to this point.
            # Or average of all, weighted by radius.
            # Simple average of circumcenters of incident facets is a good gradient step.
            centers = facet_centers[point_to_facets[p_idx]]
            avg_center = np.mean(centers, axis=0)
            
            # Step
            new_points[p_idx] = points[p_idx] + eta * (avg_center - points[p_idx])
            
        new_points = _normalize(new_points)
        
        # Check if improved
        current_radius, _ = _covering_radius(new_points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
        else:
            # If no improvement, reduce eta or break
            # Simple: break if no improvement for a few steps? 
            # Or just keep old points and try smaller eta?
            # For simplicity, if no improvement, we might be stuck.
            # Let's just break to save time if we don't improve for 5 steps.
            # We don't track non-improvement steps here easily without extra state.
            # Let's just assume the step is small enough.
            pass

    return best_points

def construct():
    global _deadline
    _deadline = time.time() + 140.0 # 10s margin
    
    best_points = None
    best_score = np.inf
    
    # Start 1: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    starts = [pts_fib]
    
    # Start 2-10: Random rotations of Fibonacci
    rng = np.random.default_rng(42)
    for _ in range(9):
        # Random rotation matrix
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        starts.append(pts_fib @ Q.T)
        
    # Start 11-15: Random points
    for _ in range(5):
        pts_rand = rng.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        starts.append(pts_rand)
        
    for s in starts:
        if time.time() > _deadline:
            break
            
        # Refine
        refined = _refine(s, max_iters=200, eta=0.5)
        score, _ = _covering_radius(refined)
        
        if score < best_score:
            best_score = score
            best_points = refined
            
    if best_points is None:
        return pts_fib
        
    return best_points
