"""Improved program for sphere covering with 50 points using geometric facet shrinking."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math


def _normalize(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Handle zero vectors to avoid division by zero
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _compute_facets_and_radii(points):
    """
    Compute the convex hull and the circumradius of each facet.
    Returns:
        hull: ConvexHull object
        facets: list of simplices
        radii: list of circumradii in radians
        centers: list of circumcenters (normalized)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, [], [], []

    facets = hull.simplices
    radii = []
    centers = []
    
    for simplex in facets:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # The circumcenter of the spherical triangle is the normalized sum of the vertices
        # provided the triangle does not contain the origin in its interior (which is true for hull facets)
        center = p1 + p2 + p3
        center_norm = np.linalg.norm(center)
        if center_norm < 1e-15:
            # Degenerate case, skip or handle
            radii.append(0.0)
            centers.append(np.array([1.0, 0.0, 0.0]))
            continue
            
        center = center / center_norm
        
        # Circumradius is the angular distance from any vertex to the center
        cos_theta = np.dot(p1, center)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        radii.append(theta)
        centers.append(center)
        
    return hull, facets, radii, centers


def _covering_radius(points):
    """Compute the covering radius in degrees."""
    _, facets, radii, _ = _compute_facets_and_radii(points)
    if not radii:
        return 180.0
    max_radius = max(radii)
    return np.degrees(max_radius)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # 2. Optimization Loop: Geometric Facet Shrinkage
    # We iteratively find the facet with the largest circumradius and move its vertices
    # towards the facet's circumcenter to shrink that specific empty cap.
    
    step_size = 0.05
    min_step = 1e-6
    shrink_factor = 0.95
    
    for iter in range(5000):
        if time.time() > deadline:
            break
            
        _, facets, radii, centers = _compute_facets_and_radii(points)
        if not radii:
            break
            
        max_idx = np.argmax(radii)
        max_rad = radii[max_idx]
        
        if max_rad < 0.32: # ~18.3 degrees in radians is 0.319 rad
            # Check if we can improve further
            pass
            
        # If the radius is very small, we might be stuck in a local minimum or converged
        if max_rad < 0.1: # 5.7 degrees, very good
            break

        # Get the worst facet
        worst_simplex = facets[max_idx]
        worst_center = centers[max_idx]
        
        # Move the vertices of the worst facet towards the center
        # This shrinks the triangle and reduces its circumradius
        for idx in worst_simplex:
            # Move towards center
            direction = worst_center - points[idx]
            points[idx] = points[idx] + step_size * direction
            points[idx] = _normalize(points[idx])
            
        # To prevent points from clumping too much or creating new large facets elsewhere,
        # we can apply a small repulsive force to all points from their nearest neighbors
        # or simply rely on the hull structure. 
        # A simple global repulsion helps maintain distribution.
        # Let's add a light repulsion step every 10 iterations to keep points spread out.
        if iter % 10 == 0:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(points[i] - points[j])
                    if d < 0.5: # Only repel if very close
                        # Repel i from j and j from i
                        diff = points[i] - points[j]
                        diff_norm = np.linalg.norm(diff)
                        if diff_norm > 1e-6:
                            diff = diff / diff_norm
                            # Force proportional to 1/d^2
                            force = 0.01 / (d * d)
                            points[i] += force * diff
                            points[j] -= force * diff
                points[i] = _normalize(points[i])
        
        # Check improvement
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            # If we didn't improve, reduce step size
            step_size *= shrink_factor
            if step_size < min_step:
                # Try a random restart or perturb
                # Perturb the worst facet vertices randomly
                for idx in worst_simplex:
                    noise = np.random.randn(3) * 0.01
                    points[idx] = points[idx] + noise
                    points[idx] = _normalize(points[idx])
                step_size = 0.02

    # Final Polish: Try a few Nelder-Mead steps on the best configuration found
    # This can help fine-tune the last bit
    if time.time() < deadline - 10:
        def objective(x):
            pts = x.reshape(N, 3)
            pts = _normalize(pts)
            return _covering_radius(pts)
            
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.fun < best_radius:
                best_points = result.x.reshape(N, 3)
                best_radius = result.fun
        except Exception:
            pass

    # Final normalization and return
    best_points = _normalize(best_points)
    return best_points
# EVOLVE-BLOCK-END
