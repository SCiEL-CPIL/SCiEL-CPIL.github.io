import numpy as np
import scipy.optimize as opt
import time
from scipy.spatial import ConvexHull

N = 50

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _facet_circumradius(a, b, c):
    """
    Compute the angular circumradius of the spherical triangle (a, b, c).
    Returns (radius, circumcenter) where circumcenter is the unit vector
    equidistant from a, b, c (the one with smaller angular distance).
    """
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.pi, np.zeros(3)
    n = n / nn
    
    best_r = np.pi
    best_c = n
    for s in (1.0, -1.0):
        p = s * n
        da = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
        db = np.arccos(np.clip(np.dot(p, b), -1.0, 1.0))
        dc = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
        # Check if p is the circumcenter (equidistant)
        # Due to numerical errors, we might not get exact equality, 
        # but the circumcenter of the great circle triangle is along the normal.
        # The distance to vertices should be equal for the true circumcenter.
        # We pick the one with smaller radius.
        if da < best_r:
            # Verify it's roughly equidistant? 
            # For a triangle on a sphere, the circumcenter is the intersection of perpendicular bisectors.
            # The normal to the plane containing a,b,c is the axis. 
            # The circumcenter lies on this axis.
            # The condition for p to be the circumcenter is that |p-a| = |p-b| = |p-c| (chord length)
            # which implies angular distance equality.
            # If the triangle is obtuse (contains the pole), the circumcenter is the other pole? 
            # No, the circumcenter of the *spherical* triangle is the point on the sphere.
            # If the triangle contains the center of the sphere (i.e., it's a "large" triangle > hemisphere), 
            # the circumcenter is the antipode of the normal? 
            # Actually, for any 3 non-collinear points, there are two points equidistant from them: 
            # the circumcenter of the small triangle and the circumcenter of the large triangle (antipode).
            # We want the one that is the "center" of the cap, i.e., the one closer to the points.
            # So we just pick the one with smaller distance.
            best_r = da
            best_c = p
    return best_r, best_c

def _covering_radius_hull(pts):
    """
    Compute the covering radius (in radians) via convex hull facet circumradii.
    Returns (radius, worst_circumcenter) or (pi, None) if invalid.
    """
    pts = _to_sphere(pts)
    
    # Check for coincident points
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    if np.min(d2[np.triu_indices(len(pts), k=1)]) < 1e-14:
        return np.pi, None
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, None
    
    if len(hull.simplices) == 0:
        return np.pi, None
    
    max_r = 0.0
    worst_c = None
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        r, cc = _facet_circumradius(a, b, c)
        if r > max_r:
            max_r = r
            worst_c = cc
    
    return max_r, worst_c

def _fibonacci_spiral(n, jitter=0.0, seed=42):
    """Generate n points on the sphere using Fibonacci spiral with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    if jitter > 0:
        # Add small random noise to break symmetry
        noise = rng.randn(n, 3) * jitter
        pts = pts + noise
        pts = _to_sphere(pts)
        
    return pts

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for safety
    
    # 1. Initialization: Jittered Fibonacci
    # Jitter helps escape the specific local minima of the pure spiral
    P = _fibonacci_spiral(N, jitter=0.05, seed=123)
    
    # 2. Grid-based optimization for fast convergence
    # Use a dense grid for the objective function to guide the optimizer
    M = 5000
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(np.maximum(0.0, 1.0 - z_grid * z_grid))
    G = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    def objective(flat_P):
        P = flat_P.reshape(N, 3)
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        P = P / norms
        
        # Dot product
        dots = G @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        
        # High power to focus on worst case (High-Beta effect)
        # Using power 8 or 10 creates a steeper gradient for the worst points
        return np.sum(min_dists ** 8)
    
    best_P = P.copy()
    best_val = objective(P.flatten())
    
    # Optimize with Powell (robust for non-smooth) or L-BFGS-B
    # Powell is slower but more robust to the non-smoothness of min(dists)
    try:
        if time.time() - start_time < deadline - 30:
            res = opt.minimize(objective, P.flatten(), method='Powell', 
                               options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_val:
                best_P = res.x.reshape(N, 3)
                best_val = res.fun
    except Exception:
        pass
    
    # 3. Refinement using Exact Convex Hull Metric (Lloyd's style)
    # This corrects for grid discretization errors and sharpens the solution
    # We perform a few iterations of moving points towards the worst circumcenter
    
    current_P = _to_sphere(best_P.copy())
    current_cr, worst_c = _covering_radius_hull(current_P)
    
    # If hull calculation failed, stick with grid result
    if worst_c is not None:
        # Local Refinement Loop
        step_size = 0.1
        for _ in range(20):
            if time.time() - start_time > deadline - 5:
                break
            
            if worst_c is None:
                break
            
            # Find the point nearest to the worst circumcenter
            # Actually, in Lloyd's algorithm for covering, we move the vertices 
            # of the worst facet towards the circumcenter.
            
            # Re-evaluate hull to get the specific worst facet vertices
            # _covering_radius_hull only returns the center, not the facet indices.
            # We need to modify it or recompute.
            
            # Let's implement a quick check: move the point closest to worst_c?
            # Or better, use the facet vertices.
            
            # For efficiency, let's just nudge the closest point to the worst_c 
            # in the direction of worst_c. This is a simple heuristic.
            d = np.sum((current_P - worst_c[None, :]) ** 2, axis=1)
            k = int(np.argmin(d))
            
            # Move point k towards worst_c
            new_pt = current_P[k] + step_size * (worst_c - current_P[k])
            new_pt = _to_sphere(new_pt.reshape(1, 3))[0]
            
            cand_P = current_P.copy()
            cand_P[k] = new_pt
            
            new_cr, new_wc = _covering_radius_hull(cand_P)
            
            if new_cr < current_cr - 1e-8:
                current_P = cand_P
                current_cr = new_cr
                worst_c = new_wc
                step_size = min(step_size * 1.05, 0.3)
            else:
                step_size *= 0.8
                if step_size < 1e-4:
                    break
        best_P = current_P
    
    # Final normalization
    best_P = _to_sphere(best_P)
    
    return best_P
