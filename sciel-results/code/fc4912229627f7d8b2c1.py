"""Program for covering sphere with 130 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(v):
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return v
    return v / norm

def _covering_radius(pts):
    """Compute the covering radius in degrees."""
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        for simplex in hull.simplices:
            # Get the three points
            p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Compute circumcenter on the sphere
            # For three points on unit sphere, circumcenter is normalized cross product
            # Actually, we need to find point equidistant from all three
            # Use the fact that for unit vectors, circumcenter direction is:
            # (p1 x p2) x (p2 x p3) + ... or use linear algebra
            # Simpler: solve for u such that u·p1 = u·p2 = u·p3
            # This is equivalent to u perpendicular to (p1-p2) and (p2-p3)
            v1 = p2 - p1
            v2 = p3 - p1
            # Circumcenter direction is cross(v1, v2)
            u_dir = np.cross(v1, v2)
            u = _normalize(u_dir)
            # The circumradius is the angle between u and any of the points
            cos_r = np.dot(u, p1)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            radius = np.degrees(np.arccos(cos_r))
            max_radius = max(max_radius, radius)
        return max_radius
    except Exception:
        return 90.0  # Default to hemisphere if hull fails

def _get_worst_facet(pts, hull):
    """Find the facet with the largest circumradius and its circumcenter."""
    max_radius = 0.0
    worst_simplex = None
    worst_circumcenter = None
    
    for simplex in hull.simplices:
        p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        u_dir = np.cross(v1, v2)
        u = _normalize(u_dir)
        cos_r = np.clip(np.dot(u, p1), -1.0, 1.0)
        radius = np.degrees(np.arccos(cos_r))
        
        if radius > max_radius:
            max_radius = radius
            worst_simplex = simplex
            worst_circumcenter = u
            
    return worst_simplex, worst_circumcenter, max_radius

def _local_optimize(pts, max_time=60.0):
    """Local optimization using greedy facet reduction."""
    start_time = time.time()
    pts = pts.copy()
    best_radius = _covering_radius(pts)
    best_pts = pts.copy()
    
    step_size = 0.02
    decay = 0.995
    min_step = 1e-5
    
    # Limit iterations to ensure speed, but rely on time check primarily
    max_iters = 2000 
    
    for iteration in range(max_iters):
        # Check time
        if time.time() - start_time > max_time - 2.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        worst_simplex, circumcenter, current_worst_radius = _get_worst_facet(pts, hull)
        
        if worst_simplex is None:
            break
            
        # Move vertices of the worst facet towards the circumcenter
        # This reduces the circumradius of this specific facet
        for p_idx in worst_simplex:
            p = pts[p_idx]
            # Direction from p to circumcenter on the sphere
            # Project circumcenter onto tangent plane at p? 
            # Or just move in direction of (circumcenter - p) normalized?
            # Since both are unit vectors, the geodesic direction is circumcenter - (circumcenter.dot(p))*p
            v = circumcenter - np.dot(circumcenter, p) * p
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-10:
                v = v / norm_v
                pts[p_idx] = p + step_size * v
                pts[p_idx] = _normalize(pts[p_idx])
        
        # Occasionally check global radius to update best
        # Doing it every step is too slow, every 10 steps is okay
        if iteration % 10 == 0:
            current_radius = _covering_radius(pts)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
        
        # Decay step size
        step_size *= decay
        if step_size < min_step:
            break
    
    # Final evaluation
    final_radius = _covering_radius(pts)
    if final_radius < best_radius:
        best_radius = final_radius
        best_pts = pts.copy()
    
    return best_pts, best_radius

def _fibonacci_spiral(n, seed=0):
    """Generate Fibonacci spiral points on unit sphere."""
    rng = np.random.RandomState(seed)
    # Add small perturbation for variety
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Small random perturbation
    pts += rng.normal(0, 0.01, size=(n, 3))
    pts = np.array([_normalize(p) for p in pts])
    return pts

def _random_initial(n, seed=0):
    """Generate random initial points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.normal(0, 1, size=(n, 3))
    pts = np.array([_normalize(p) for p in pts])
    return pts

def construct():
    """Main function: construct 130 points on unit sphere minimizing covering radius."""
    global_start = time.time()
    deadline = 110.0  # Total time limit (safety margin for 120s kill)
    
    best_pts = None
    best_radius = float('inf')
    
    # Strategy 1: Fibonacci spiral with different seeds
    # Reduce number of restarts to fit in time budget
    for seed in range(3):
        if time.time() - global_start > deadline - 20:
            break
        pts = _fibonacci_spiral(N, seed)
        remaining = deadline - (time.time() - global_start) - 5
        if remaining < 5:
            break
        pts, radius = _local_optimize(pts, max_time=remaining)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Strategy 2: Random initializations
    for seed in range(3):
        if time.time() - global_start > deadline - 20:
            break
        pts = _random_initial(N, seed)
        remaining = deadline - (time.time() - global_start) - 5
        if remaining < 5:
            break
        pts, radius = _local_optimize(pts, max_time=remaining)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Final polish with more iterations if time allows
    if best_pts is not None and time.time() - global_start < deadline - 10:
        remaining_time = deadline - (time.time() - global_start) - 5
        if remaining_time > 5:
            best_pts, final_radius = _local_optimize(best_pts, max_time=remaining_time)
            if final_radius < best_radius:
                best_radius = final_radius
    
    # Ensure validity
    if best_pts is None:
        # Fallback: Fibonacci spiral
        best_pts = _fibonacci_spiral(N, 0)
    
    # Final normalization and validation
    best_pts = np.array([_normalize(p) for p in best_pts])
    
    # Check for coincident points and hemisphere issues
    # If invalid, use fallback
    if not np.all(np.isfinite(best_pts)) or len(set(map(tuple, best_pts.round(10)))) < N:
        best_pts = _fibonacci_spiral(N, 0)
    
    return best_pts

# EVOLVE-BLOCK-END
