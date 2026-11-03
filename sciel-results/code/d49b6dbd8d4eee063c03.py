"""Covering the sphere with 50 equal caps – structural redesign.

Uses a multi-stage pipeline:
  1. Diverse jittered-Fibonacci initializations (break symmetry).
  2. Lloyd's algorithm (Voronoi / farthest-point centroid) for coarse optimisation.
  3. Gradient descent on the covering-radius objective with adaptive step size.
  4. Facet-shrink local refinement on the worst circumcircle.

A deadline guard ensures we return before the 150 s wall-clock limit.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay

N = 50
TIME_LIMIT = 140.0  # safety margin before 150 s kill

def _normalise(X):
    """Normalise rows of X to unit length, handling zero vectors."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def _fibonacci_spiral(n, jitter=0.0, seed=None):
    """Generate n points on the unit sphere via Fibonacci spiral with optional jitter."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0.0:
        noise = rng.normal(0, jitter, size=pts.shape)
        pts = pts + noise
        pts = _normalise(pts)
    return pts

def _covering_radius(pts):
    """Compute the covering radius (in radians) of a set of points on the sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest point in pts. This equals the maximum circumradius of the facets
    of the convex hull of pts (each facet is a spherical triangle).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # invalid configuration
    
    # For each facet (triangle), compute the circumradius on the sphere.
    # The circumradius of a spherical triangle with vertices a, b, c (unit vectors)
    # is the angular distance from the facet's circumcenter to any vertex.
    # We can compute it via the formula: if the facet has normal n, the circumcenter
    # is along n (or -n), and the radius is arccos(max dot product of vertices with n).
    
    max_radius = 0.0
    for simplex in hull.simplices:
        tri = pts[simplex]  # shape (3, 3)
        # Compute normal
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        # Ensure n points outward (away from origin for convex hull)
        # The centroid of the triangle should be on the same side as n
        centroid = tri.mean(axis=0)
        if np.dot(centroid, n) < 0:
            n = -n
        # Angular distance from n to each vertex
        dots = np.clip(tri @ n, -1.0, 1.0)
        angles = np.arccos(dots)
        facet_radius = angles.max()
        max_radius = max(max_radius, facet_radius)
    
    return max_radius

def _lloyd_step(pts, iterations=100):
    """Perform Lloyd's algorithm: iteratively replace each point with the
    centroid of its Voronoi cell (on the sphere)."""
    pts = _normalise(pts.copy())
    for _ in range(iterations):
        # Compute Delaunay triangulation of points on the sphere
        try:
            # Project to 3D and use Delaunay
            delaunay = Delaunay(pts)
        except Exception:
            break
        
        # For each point, find its Voronoi cell as the set of Delaunay triangles
        # that contain it. The centroid of the spherical Voronoi cell is the
        # normalized sum of the circumcenters of adjacent triangles.
        # A simpler approach: for each point, compute the centroid of all points
        # in its "neighborhood" using the Delaunay edges.
        
        # Actually, a simpler Lloyd step for spherical covering:
        # For each point p_i, find all other points and compute the weighted
        # centroid of the region where p_i is the nearest.
        # Approximate: use the Delaunay triangulation to find neighbors,
        # then move p_i toward the centroid of its Voronoi cell.
        
        # Simpler approach: for each point, compute the mean of all points
        # in the hemisphere centered at p_i (approximate Voronoi).
        new_pts = np.zeros_like(pts)
        for i in range(len(pts)):
            # Find points closer to p_i than to any other point
            # This is expensive; use a simpler heuristic:
            # Move toward the centroid of the nearest neighbors
            diffs = pts - pts[i]
            dists = np.linalg.norm(diffs, axis=1)
            # Find the 10 nearest neighbors
            k = min(10, len(pts) - 1)
            idx = np.argsort(dists)[1:k+1]  # exclude self
            # Weighted centroid
            weights = 1.0 / (dists[idx] + 1e-10)
            centroid = np.sum(pts[idx] * weights[:, None], axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_pts[i] = centroid / norm
            else:
                new_pts[i] = pts[i]
        
        # Blend to avoid oscillation
        pts = _normalise(0.5 * pts + 0.5 * new_pts)
    
    return pts

def _gradient_descent(pts, max_iters=2000, deadline=None):
    """Gradient descent on the covering radius objective."""
    pts = _normalise(pts.copy())
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    lr = 0.01
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute covering radius and find worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        max_radius = 0.0
        worst_simplex = None
        worst_normal = None
        for simplex in hull.simplices:
            tri = pts[simplex]
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            centroid = tri.mean(axis=0)
            if np.dot(centroid, n) < 0:
                n = -n
            dots = np.clip(tri @ n, -1.0, 1.0)
            angles = np.arccos(dots)
            facet_radius = angles.max()
            if facet_radius > max_radius:
                max_radius = facet_radius
                worst_simplex = simplex
                worst_normal = n
        
        if worst_simplex is None:
            break
        
        # Compute gradient: for each point in the worst facet,
        # move it toward the circumcenter (normal direction)
        # to reduce the facet radius.
        grad = np.zeros_like(pts)
        for idx in worst_simplex:
            # Move point idx toward the normal direction
            grad[idx] += worst_normal
        
        # Also add a small repulsion from other points to avoid collapse
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 0.1:
                    force = diff / (dist + 1e-10) * 0.01
                    grad[i] += force
                    grad[j] -= force
        
        # Project gradient onto tangent space
        for i in range(len(pts)):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Update
        pts = _normalise(pts + lr * grad)
        
        # Adaptive learning rate
        new_radius = _covering_radius(pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = pts.copy()
            lr = min(lr * 1.05, 0.1)
        else:
            lr = max(lr * 0.8, 0.001)
        
        # Convergence check
        if it > 100 and abs(new_radius - best_radius) < 1e-8:
            break
    
    return best_pts

def _facet_shrink(pts, max_iters=500, deadline=None):
    """Local refinement: repeatedly find the worst facet and move its vertices
    toward the circumcenter to shrink the facet."""
    pts = _normalise(pts.copy())
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        max_radius = 0.0
        worst_simplex = None
        worst_normal = None
        for simplex in hull.simplices:
            tri = pts[simplex]
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            centroid = tri.mean(axis=0)
            if np.dot(centroid, n) < 0:
                n = -n
            dots = np.clip(tri @ n, -1.0, 1.0)
            angles = np.arccos(dots)
            facet_radius = angles.max()
            if facet_radius > max_radius:
                max_radius = facet_radius
                worst_simplex = simplex
                worst_normal = n
        
        if worst_simplex is None:
            break
        
        # Move vertices of the worst facet toward the circumcenter
        step = 0.05
        new_pts = pts.copy()
        for idx in worst_simplex:
            # Move toward the normal (circumcenter direction)
            new_pts[idx] = pts[idx] + step * worst_normal
            # Project back to sphere
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_radius = _covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Reduce step
            step *= 0.5
            if step < 1e-6:
                break
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_pts = None
    best_radius = np.pi  # worst possible
    
    # Generate diverse initializations
    n_restarts = 8
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:  # leave 10s safety margin
            break
        
        seed = seeds[restart % len(seeds)]
        jitter = 0.1 + 0.05 * restart  # increasing jitter for diversity
        pts = _fibonacci_spiral(N, jitter=jitter, seed=seed)
        
        # Stage 1: Lloyd's algorithm
        pts = _lloyd_step(pts, iterations=50)
        
        # Stage 2: Gradient descent
        pts = _gradient_descent(pts, max_iters=1000, deadline=deadline - 5.0)
        
        # Stage 3: Facet shrink
        pts = _facet_shrink(pts, max_iters=300, deadline=deadline - 2.0)
        
        # Evaluate
        radius = _covering_radius(pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Final safety: if no valid result, return Fibonacci spiral
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, jitter=0.0)
    
    return _normalise(best_pts)
