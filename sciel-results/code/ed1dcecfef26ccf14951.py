"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, LinearConstraint

N = 50
DEADLINE = 115.0  # safety margin before 120s kill


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n


def _covering_radius(pts):
    """Compute covering radius in degrees using convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_ang = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]  # shape (k, 3), k=3 typically
        
        a, b, c = verts[0], verts[1], verts[2]
        
        # Circumcenter of spherical triangle
        v1 = a - b
        v2 = a - c
        x = np.cross(v1, v2)
        x = _normalize(x)
        
        centroid = (a + b + c) / 3.0
        outward = _normalize(centroid)
        
        if np.dot(x, outward) < 0:
            x = -x
        
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        ang = np.degrees(np.arccos(dot_val))
        
        if ang > max_ang:
            max_ang = ang
    
    return max_ang


def _covering_radius_grid(pts, n_grid=200):
    """Fallback covering radius using a grid on the sphere."""
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = grid @ pts.T  # (n_grid, N)
    max_dots = np.max(dots, axis=1)
    min_ang = np.degrees(np.arccos(np.clip(max_dots, -1.0, 1.0)))
    return np.max(min_ang)


def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _compute_worst_facet(pts):
    """Find the facet with the largest circumradius and its circumcenter."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, 0.0
    
    worst_facet = None
    worst_ang = 0.0
    worst_circ = None
    
    for simplex in hull.simplices:
        verts = pts[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        
        v1 = a - b
        v2 = a - c
        x = np.cross(v1, v2)
        x = _normalize(x)
        
        centroid = (a + b + c) / 3.0
        outward = _normalize(centroid)
        
        if np.dot(x, outward) < 0:
            x = -x
        
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        ang = np.degrees(np.arccos(dot_val))
        
        if ang > worst_ang:
            worst_ang = ang
            worst_facet = simplex
            worst_circ = x
    
    return worst_facet, worst_circ, worst_ang


def _local_optimize(pts, time_limit=10.0):
    """Local optimization using gradient-free approach on the sphere."""
    start_time = time.time()
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    step_size = 0.05
    n_iter = 0
    
    while time.time() - start_time < time_limit:
        # Find the worst facet
        worst_facet, worst_circ, worst_ang = _compute_worst_facet(pts)
        
        if worst_facet is None:
            break
        
        # Move the vertices of the worst facet toward the circumcenter
        verts = pts[worst_facet].copy()
        for idx in worst_facet:
            pts[idx] = _normalize(pts[idx] + step_size * 0.3 * worst_circ)
        
        # Also try to move other points to improve overall coverage
        cr = _covering_radius(pts)
        n_iter += 1
        
        if cr < best_cr - 1e-9:
            best_cr = cr
            best_pts = pts.copy()
            step_size = max(0.001, step_size * 0.95)
        else:
            # Revert and try different perturbation
            pts = best_pts.copy()
            step_size = max(0.001, step_size * 0.7)
            
            # Random perturbation to escape local minimum
            for idx in range(N):
                random_dir = np.random.randn(3)
                random_dir = _normalize(random_dir)
                tangent = random_dir - np.dot(random_dir, pts[idx]) * pts[idx]
                tangent = _normalize(tangent)
                pts[idx] = _normalize(pts[idx] + step_size * 0.1 * tangent)
            
            cr = _covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts.copy()
    
    return best_pts, best_cr


def _gradient_based_optimize(pts, time_limit=15.0):
    """Gradient-based optimization using smooth surrogate."""
    start_time = time.time()
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    def objective(x_flat):
        x = x_flat.reshape(N, 3)
        # Normalize to sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        x_norm = x / norms
        return _covering_radius(x_norm)
    
    # Use Nelder-Mead for derivative-free optimization
    x0 = pts.flatten()
    
    while time.time() - start_time < time_limit:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            break
        
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 50, 'fatol': 1e-8, 'xatol': 1e-8}
        )
        
        x_opt = result.x.reshape(N, 3)
        norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        x_opt = x_opt / norms
        
        cr = _covering_radius(x_opt)
        if cr < best_cr - 1e-9:
            best_cr = cr
            best_pts = x_opt.copy()
            x0 = x_opt.flatten()
        else:
            # Perturb and try again
            perturbation = 0.01 * np.random.randn(N, 3)
            x0 = (x_opt + perturbation).flatten()
    
    return best_pts, best_cr


def _restart_with_perturbation(base_pts, strength=0.1):
    """Create a perturbed version of base_pts."""
    pts = base_pts.copy()
    for idx in range(N):
        random_dir = np.random.randn(3)
        random_dir = _normalize(random_dir)
        tangent = random_dir - np.dot(random_dir, pts[idx]) * pts[idx]
        tangent = _normalize(tangent)
        pts[idx] = _normalize(pts[idx] + strength * tangent)
    return pts


def _final_polish(pts, time_limit=10.0):
    """High-precision polish pass with tight tolerance."""
    start_time = time.time()
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    step_size = 0.005
    
    while time.time() - start_time < time_limit:
        worst_facet, worst_circ, worst_ang = _compute_worst_facet(pts)
        
        if worst_facet is None:
            break
        
        # Small step toward circumcenter
        for idx in worst_facet:
            pts[idx] = _normalize(pts[idx] + step_size * 0.2 * worst_circ)
        
        cr = _covering_radius(pts)
        
        if cr < best_cr - 1e-10:
            best_cr = cr
            best_pts = pts.copy()
            step_size = max(0.0001, step_size * 0.9)
        else:
            pts = best_pts.copy()
            step_size = max(0.0001, step_size * 0.8)
            
            # Tiny random perturbation
            for idx in range(N):
                random_dir = np.random.randn(3)
                random_dir = _normalize(random_dir)
                tangent = random_dir - np.dot(random_dir, pts[idx]) * pts[idx]
                tangent = _normalize(tangent)
                pts[idx] = _normalize(pts[idx] + step_size * 0.05 * tangent)
            
            cr = _covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts.copy()
    
    return best_pts, best_cr


def construct():
    global DEADLINE
    t_start = time.time()
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Phase 1: Local optimization from initial point
    if time.time() - t_start < DEADLINE - 40:
        remaining = DEADLINE - (time.time() - t_start) - 30
        best_pts, best_cr = _local_optimize(best_pts, time_limit=min(15.0, remaining))
    
    # Phase 2: Gradient-based optimization
    if time.time() - t_start < DEADLINE - 20:
        remaining = DEADLINE - (time.time() - t_start) - 15
        new_pts, new_cr = _gradient_based_optimize(best_pts, time_limit=min(20.0, remaining))
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = new_pts.copy()
    
    # Phase 3: Multiple restarts with perturbation
    n_restarts = 0
    while time.time() - t_start < DEADLINE - 15:
        strength = 0.03 + 0.04 * np.random.random()
        perturbed = _restart_with_perturbation(best_pts, strength)
        
        remaining = DEADLINE - (time.time() - t_start) - 10
        if remaining < 5:
            break
        time_limit = min(8.0, remaining)
        new_pts, new_cr = _local_optimize(perturbed, time_limit=time_limit)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = new_pts.copy()
        
        n_restarts += 1
    
    # Phase 4: Final polish
    if time.time() - t_start < DEADLINE - 5:
        remaining = DEADLINE - (time.time() - t_start) - 3
        best_pts, best_cr = _final_polish(best_pts, time_limit=min(10.0, remaining))
    
    # Final check: make sure points are on the unit sphere and valid
    best_pts = np.array([_normalize(p) for p in best_pts])
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                best_pts[j] = _normalize(best_pts[j] + 1e-4 * np.random.randn(3))
    
    return best_pts
# EVOLVE-BLOCK-END
