"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # safety margin before 120s kill


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n


def _compute_facet_data(pts):
    """Compute hull and per-facet circumcenter and circumradius."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None
    
    n_facets = len(hull.simplices)
    circumcenters = np.zeros((n_facets, 3))
    circumradii = np.zeros(n_facets)
    
    for i, simplex in enumerate(hull.simplices):
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1 = a - b
        v2 = a - c
        x = np.cross(v1, v2)
        x = _normalize(x)
        centroid = (a + b + c) / 3.0
        outward = _normalize(centroid)
        if np.dot(x, outward) < 0:
            x = -x
        circumcenters[i] = x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        circumradii[i] = np.arccos(dot_val)  # in radians
    
    return hull, circumcenters, circumradii


def _covering_radius(pts):
    """Compute covering radius in degrees."""
    _, _, circumradii = _compute_facet_data(pts)
    if circumradii is None:
        return 180.0
    return np.degrees(np.max(circumradii))


def _soft_covering_radius(pts, temperature=0.01):
    """Smooth surrogate using log-sum-exp over facet circumradii (radians)."""
    _, _, circumradii = _compute_facet_data(pts)
    if circumradii is None:
        return 180.0 * np.pi / 180.0
    
    # Log-sum-exp for numerical stability
    max_c = np.max(circumradii)
    log_sum = max_c + np.log(np.sum(np.exp(circumradii - max_c)))
    return log_sum / temperature


def _soft_covering_radius_and_grad(pts, temperature=0.01):
    """Compute soft covering radius and its gradient w.r.t. point positions.
    
    The gradient of the circumradius of a facet w.r.t. a vertex can be derived.
    For facet with circumcenter x and vertices a, b, c:
    r = arccos(x · a)
    dr/da = -x / sqrt(1 - (x·a)^2) (in the direction of a)
    
    But we need to be careful: moving a point affects all facets containing it.
    """
    hull, circumcenters, circumradii = _compute_facet_data(pts)
    if circumradii is None:
        return 180.0 * np.pi / 180.0, np.zeros((N, 3))
    
    # Soft max weights
    max_c = np.max(circumradii)
    exp_vals = np.exp(circumradii - max_c)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient accumulator
    grad = np.zeros((N, 3))
    
    for i, simplex in enumerate(hull.simplices):
        w = weights[i]
        if w < 1e-10:
            continue
        
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        x = circumcenters[i]
        
        # r = arccos(x · a), dr/da = -x / sin(r)
        # But we need the component tangent to the sphere
        for idx, vert in zip(simplex, [a, b, c]):
            dot_val = np.dot(x, vert)
            sin_r = np.sqrt(np.maximum(0, 1.0 - dot_val * dot_val))
            if sin_r < 1e-10:
                continue
            # Gradient of r w.r.t. vert (before projection)
            dr_dvert = -x / sin_r
            # Project to tangent space: remove component along vert
            dr_dvert_tangent = dr_dvert - np.dot(dr_dvert, vert) * vert
            grad[idx] += w * dr_dvert_tangent
    
    # Scale by 1/temperature for the soft-max
    grad /= temperature
    
    value = max_c + np.log(np.sum(exp_vals)) / temperature
    return value, grad


def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    return np.array([_normalize(p) for p in pts])


def _local_optimize_smooth(pts, time_limit=20.0, temperature=0.01):
    """Optimize using smooth surrogate with L-BFGS in tangent space."""
    start_time = time.time()
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    def objective_and_grad(x_flat):
        if time.time() - start_time > time_limit:
            return best_cr * np.pi / 180.0, np.zeros(3 * N)
        p = x_flat.reshape(N, 3)
        val, grad = _soft_covering_radius_and_grad(p, temperature)
        return val, grad.flatten()
    
    # Use L-BFGS-B with the smooth objective
    # We'll do a few iterations of projected gradient descent
    step_size = 0.01
    n_iter = 0
    
    while time.time() - start_time < time_limit:
        # Compute gradient
        val, grad = _soft_covering_radius_and_grad(pts, temperature)
        
        # Project gradient to tangent space
        grad_tangent = np.zeros((N, 3))
        for i in range(N):
            grad_tangent[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
        
        # Take a step
        pts_new = pts - step_size * grad_tangent
        pts_new = _project_to_sphere(pts_new)
        
        # Check improvement
        cr_new = _covering_radius(pts_new)
        
        if cr_new < best_cr - 1e-10:
            best_cr = cr_new
            best_pts = pts_new.copy()
            pts = pts_new
            step_size = min(0.05, step_size * 1.02)
        else:
            # Reduce step size and try again
            step_size *= 0.7
            if step_size < 1e-6:
                break
        
        n_iter += 1
    
    return best_pts, best_cr


def _local_optimize_facet(pts, time_limit=15.0):
    """Direct facet-based optimization: move worst facet vertices toward circumcenter."""
    start_time = time.time()
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    step_size = 0.03
    
    while time.time() - start_time < time_limit:
        hull, circumcenters, circumradii = _compute_facet_data(pts)
        if hull is None:
            break
        
        # Find worst facet
        worst_idx = np.argmax(circumradii)
        worst_facet = hull.simplices[worst_idx]
        worst_x = circumcenters[worst_idx]
        worst_r = np.degrees(circumradii[worst_idx])
        
        if worst_r < best_cr - 0.01:
            break
        
        # Move vertices of worst facet toward circumcenter
        for idx in worst_facet:
            pts[idx] = _normalize(pts[idx] + step_size * worst_x)
        
        cr = _covering_radius(pts)
        if cr < best_cr - 1e-9:
            best_cr = cr
            best_pts = pts.copy()
            step_size = max(0.001, step_size * 0.97)
        else:
            pts = best_pts.copy()
            step_size *= 0.7
            if step_size < 1e-5:
                break
    
    return best_pts, best_cr


def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


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


def construct():
    global DEADLINE
    t_start = time.time()
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Phase 1: Smooth optimization from initial point
    remaining = DEADLINE - (time.time() - t_start) - 10
    if remaining > 10:
        best_pts, best_cr = _local_optimize_smooth(best_pts, time_limit=min(25.0, remaining))
    
    # Phase 2: Facet-based polish
    remaining = DEADLINE - (time.time() - t_start) - 8
    if remaining > 5:
        best_pts, best_cr = _local_optimize_facet(best_pts, time_limit=min(15.0, remaining))
    
    # Phase 3: Multiple restarts with perturbation
    n_restarts = 0
    while time.time() - t_start < DEADLINE - 10:
        strength = 0.03 + 0.04 * np.random.random()
        perturbed = _restart_with_perturbation(best_pts, strength)
        
        remaining = DEADLINE - (time.time() - t_start) - 5
        if remaining < 8:
            break
        
        # Smooth optimize
        time_limit1 = min(12.0, remaining * 0.5)
        new_pts, new_cr = _local_optimize_smooth(perturbed, time_limit=time_limit1)
        
        # Facet polish
        remaining2 = DEADLINE - (time.time() - t_start) - 5
        if remaining2 > 3:
            time_limit2 = min(8.0, remaining2)
            new_pts, new_cr = _local_optimize_facet(new_pts, time_limit=time_limit2)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = new_pts.copy()
        
        n_restarts += 1
    
    # Final facet polish
    remaining = DEADLINE - (time.time() - t_start) - 3
    if remaining > 2:
        best_pts, best_cr = _local_optimize_facet(best_pts, time_limit=min(5.0, remaining))
    
    # Final check: make sure points are on the unit sphere and valid
    best_pts = np.array([_normalize(p) for p in best_pts])
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                best_pts[j] = _normalize(best_pts[j] + 1e-4 * np.random.randn(3))
    
    return best_pts
# EVOLVE-BLOCK-END
