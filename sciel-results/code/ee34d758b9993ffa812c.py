import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _fibonacci_spiral(n, seed=42):
    """Generate n points on unit sphere using Fibonacci spiral with optional jitter."""
    np.random.seed(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Add small random perturbation
    for idx in range(n):
        p = pts[idx]
        rand_vec = np.random.randn(3)
        tangent = rand_vec - np.dot(rand_vec, p) * p
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-10:
            pts[idx] = p + 0.01 * tangent / tangent_norm
            pts[idx] = pts[idx] / np.linalg.norm(pts[idx])
    return _normalize(pts)

def _random_sphere(n, seed=0):
    """Generate n random points on the unit sphere."""
    np.random.seed(seed)
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _circumcenter_robust(p1, p2, p3):
    """Compute circumcenter of triangle on unit sphere, robust to degeneracy."""
    # Try multiple edge pairs for cross product to handle near-degenerate triangles
    edges = [
        (p2 - p1, p3 - p1),
        (p1 - p2, p3 - p2),
        (p1 - p3, p2 - p3)
    ]
    
    for v1, v2 in edges:
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        
        n = n / n_norm
        # The circumcenter should be along the normal direction
        # For points on unit sphere, circumcenter is the normalized normal
        # if the triangle is acute, otherwise it's on the opposite side
        center = n.copy()
        # Check if the center is on the correct side (same side as the triangle's interior)
        # The center should be such that the angle from center to each vertex is <= 90 degrees
        # or we can check that the center is in the same hemisphere as the triangle
        # A simple check: the center should be closer to the centroid than the antipode
        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(center, centroid) < 0:
            center = -center
        return center
    return None

def _compute_circumradii(pts):
    """
    Compute the covering radius using convex hull facets.
    For each triangular facet, compute the circumcenter and the angular distance
    from the circumcenter to each of the three vertices. The maximum of these
    distances over all facets is the covering radius.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback to grid-based computation if hull fails
        n_grid = 5000
        grid_pts = _fibonacci_spiral(n_grid, seed=12345)
        dots = grid_pts @ pts.T
        max_dots = np.max(dots, axis=1)
        distances = np.arccos(np.clip(max_dots, -1.0, 1.0))
        return np.max(distances)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]
        
        center = _circumcenter_robust(p1, p2, p3)
        if center is None:
            continue
        
        # Compute angular distances from center to each vertex
        d1 = np.arccos(np.clip(np.dot(center, p1), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.dot(center, p2), -1.0, 1.0))
        d3 = np.arccos(np.clip(np.dot(center, p3), -1.0, 1.0))
        
        facet_radius = max(d1, d2, d3)
        if facet_radius > max_radius:
            max_radius = facet_radius
    
    return max_radius

def _covering_radius_deg(pts):
    """Compute covering radius in degrees."""
    return np.degrees(_compute_circumradii(pts))

def _soft_max_obj(pts, grid_pts, beta):
    """
    Compute a differentiable approximation of the covering radius using soft-max.
    For each grid point, compute the distance to the nearest config point,
    then take a soft-max over all grid points.
    """
    dots = grid_pts @ pts.T  # (n_grid, n_config)
    max_dots = np.max(dots, axis=1)  # (n_grid,)
    distances = np.arccos(np.clip(max_dots, -1.0, 1.0))
    
    # Soft-max approximation of max
    # Use log-sum-exp for numerical stability
    max_d = np.max(distances)
    if max_d < 1e-10:
        return 0.0, np.zeros_like(pts)
    
    shifted = distances - max_d
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    soft_max_val = max_d + np.log(np.sum(weights)) / beta
    
    # Gradient computation
    # d(soft_max)/d(distances_i) = weights_i
    # d(distances_i)/d(pts_j) where j is the nearest config point to grid_i
    # d(arccos(x))/d(x) = -1/sqrt(1-x^2)
    # d(x)/d(p_j) = -g_i (where x = g_i . p_j)
    
    max_idx = np.argmax(dots, axis=1)  # (n_grid,)
    
    grad = np.zeros_like(pts)
    for i in range(len(grid_pts)):
        if weights[i] < 1e-10:
            continue
        j = max_idx[i]
        g = grid_pts[i]
        p = pts[j]
        dot_val = np.clip(np.dot(g, p), -1.0 + 1e-10, 1.0 - 1e-10)
        denom = np.sqrt(1.0 - dot_val * dot_val)
        # Gradient contribution: weight_i * (-g / denom)
        grad[j] += weights[i] * (-g / denom)
    
    return soft_max_val, grad

def _soft_max_optimize(pts, deadline):
    """
    Optimize using soft-max objective with dynamic temperature (beta) schedule.
    Start with low beta (smooth) and increase to high beta (sharp, approaches true max).
    """
    n_grid = 3000
    grid_pts = _fibonacci_spiral(n_grid, seed=12345)
    
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_radius = _compute_circumradii(current_pts)
    
    # Beta schedule: start smooth, get sharper
    beta_start = 5.0
    beta_end = 50.0
    total_iters = 200
    
    for it in range(total_iters):
        if time.time() > deadline - 10.0:
            break
        
        # Dynamic beta
        progress = it / total_iters
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Compute objective and gradient
        obj_val, grad = _soft_max_obj(current_pts, grid_pts, beta)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad = grad / grad_norm
        
        # Adaptive step size
        step_size = 0.01 / (1.0 + progress)
        
        # Update points, projecting to tangent space
        new_pts = current_pts.copy()
        for idx in range(N):
            p = current_pts[idx]
            g = grad[idx]
            g_tangent = g - np.dot(g, p) * p
            new_pts[idx] = p - step_size * g_tangent
            norm = np.linalg.norm(new_pts[idx])
            if norm > 1e-10:
                new_pts[idx] = new_pts[idx] / norm
        
        # Evaluate true covering radius
        new_radius = _compute_circumradii(new_pts)
        
        if new_radius < best_radius - 1e-10:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts.copy()
        else:
            # No improvement, try a smaller step or perturbation
            current_pts = best_pts.copy()
            if it % 20 == 0 and time.time() < deadline - 20.0:
                # Add small random perturbation to escape local minima
                n_perturb = max(3, N // 20)
                idxs = np.random.choice(N, n_perturb, replace=False)
                for idx in idxs:
                    p = current_pts[idx]
                    rand_vec = np.random.randn(3)
                    tangent = rand_vec - np.dot(rand_vec, p) * p
                    tangent_norm = np.linalg.norm(tangent)
                    if tangent_norm > 1e-10:
                        current_pts[idx] = p + 0.02 * tangent / tangent_norm
                        current_pts[idx] = current_pts[idx] / np.linalg.norm(current_pts[idx])
    
    return best_pts

def _repulsive_force(pts, strength=0.1, iterations=50):
    """
    Apply repulsive forces between points to avoid clustering.
    Points that are too close push each other apart.
    """
    current_pts = pts.copy()
    
    for _ in range(iterations):
        forces = np.zeros_like(current_pts)
        for i in range(N):
            for j in range(i + 1, N):
                d = current_pts[i] - current_pts[j]
                dist = np.linalg.norm(d)
                if dist < 0.1:  # Only repel if close
                    force = d / (dist ** 2 + 1e-6) * strength
                    forces[i] += force
                    forces[j] -= force
        
        # Apply forces with projection to tangent space
        for i in range(N):
            p = current_pts[i]
            f = forces[i]
            f_tangent = f - np.dot(f, p) * p
            current_pts[i] = p + 0.01 * f_tangent
            current_pts[i] = current_pts[i] / np.linalg.norm(current_pts[i])
    
    return current_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Try multiple initializations and pick the best
    best_overall = None
    best_overall_radius = np.inf
    
    # Initialization 1: Fibonacci spiral
    pts1 = _fibonacci_spiral(N, seed=42)
    r1 = _compute_circumradii(pts1)
    if r1 < best_overall_radius:
        best_overall_radius = r1
        best_overall = pts1.copy()
    
    # Initialization 2: Random
    pts2 = _random_sphere(N, seed=7)
    r2 = _compute_circumradii(pts2)
    if r2 < best_overall_radius:
        best_overall_radius = r2
        best_overall = pts2.copy()
    
    # Initialization 3: Fibonacci with different seed
    pts3 = _fibonacci_spiral(N, seed=123)
    r3 = _compute_circumradii(pts3)
    if r3 < best_overall_radius:
        best_overall_radius = r3
        best_overall = pts3.copy()
    
    # Optimize the best initialization using soft-max approach
    try:
        optimized = _soft_max_optimize(best_overall, deadline)
        if np.all(np.isfinite(optimized)):
            opt_radius = _compute_circumradii(optimized)
            if opt_radius < best_overall_radius:
                best_overall = optimized
                best_overall_radius = opt_radius
    except Exception:
        pass
    
    # If time remains, apply repulsive force to further improve
    if time.time() < deadline - 15.0:
        try:
            repulsed = _repulsive_force(best_overall, strength=0.05, iterations=30)
            rep_radius = _compute_circumradii(repulsed)
            if rep_radius < best_overall_radius:
                best_overall = repulsed
                best_overall_radius = rep_radius
        except Exception:
            pass
    
    return _normalize(best_overall)
