"""Rewritten program for sphere covering with 50 caps."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import time


def _covering_radius_and_grad(points):
    """Compute the covering radius (in radians) and gradient for a set of points on the unit sphere.
    
    Returns (max_theta, grad) where grad has shape (N, 3) and points in the direction
    to move each point to reduce the covering radius.
    """
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: sampling-based estimate with no gradient
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 100000
        gauss = rng.randn(n_samples, 3)
        gauss /= np.linalg.norm(gauss, axis=1, keepdims=True)
        d, idx = tree.query(gauss)
        max_theta = 2 * np.arcsin(np.clip(d.max() / 2.0, 0, 1))
        return max_theta, np.zeros_like(points)
    
    simplices = hull.simplices
    p0 = points[simplices[:, 0]]
    p1 = points[simplices[:, 1]]
    p2 = points[simplices[:, 2]]
    
    e1 = p1 - p0
    e2 = p2 - p0
    
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    
    valid = n_norm.squeeze() > 1e-12
    if not np.all(valid):
        n_norm[~valid] = 1.0
    
    C = n / n_norm
    
    centroid = (p0 + p1 + p2) / 3.0
    centroid_norm = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid_norm[centroid_norm < 1e-12] = 1.0
    centroid = centroid / centroid_norm
    
    dot_cc = np.sum(C * centroid, axis=1)
    C = np.where(dot_cc[:, None] < 0, -C, C)
    
    dots = np.sum(C * p0, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    thetas = np.arccos(dots)
    
    if not np.all(valid):
        thetas[~valid] = 0.0
    
    max_theta = np.max(thetas)
    
    # Compute gradient: for the facet with max theta, push its vertices toward C
    grad = np.zeros_like(points)
    if max_theta > 1e-10:
        max_idx = np.argmax(thetas)
        if valid[max_idx]:
            facet_verts = simplices[max_idx]
            C_max = C[max_idx]
            # Move each vertex toward the circumcenter to reduce the angular distance
            for vi in facet_verts:
                # Gradient direction: move point toward C on the sphere
                # d/dp arccos(C·p) = -C / sqrt(1-(C·p)^2) projected tangent
                dot_p = np.sum(C_max * points[vi])
                dot_p = np.clip(dot_p, -1.0, 1.0)
                denom = np.sqrt(max(1e-12, 1.0 - dot_p * dot_p))
                # Tangent component of -C/denom
                tangent = -C_max / denom - (np.sum(C_max * points[vi]) / denom) * points[vi]
                grad[vi] += tangent
    
    return max_theta, grad


def _covering_radius(points):
    """Compute the covering radius (in radians) for a set of points on the unit sphere."""
    theta, _ = _covering_radius_and_grad(points)
    return theta


def _fast_covering_radius(points):
    """Fast approximation using nearest-neighbor sampling for quick evaluation."""
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    tree = cKDTree(points)
    rng = np.random.RandomState(42)
    n_samples = 50000
    gauss = rng.randn(n_samples, 3)
    gauss /= np.linalg.norm(gauss, axis=1, keepdims=True)
    d, _ = tree.query(gauss)
    return 2 * np.arcsin(np.clip(d.max() / 2.0, 0, 1))


def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on the unit sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _random_sphere(n, seed=42):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    gauss = rng.randn(n, 3)
    norms = np.linalg.norm(gauss, axis=1, keepdims=True)
    return gauss / norms


def _gradient_descent(pts, max_iter=2000, lr=0.01, seed=42):
    """Gradient descent using the covering radius gradient."""
    rng = np.random.RandomState(seed)
    current_pts = pts.copy()
    current_radius = _covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    for iteration in range(max_iter):
        if time.time() > 115:
            break
        
        theta, grad = _covering_radius_and_grad(current_pts)
        
        if theta < best_radius:
            best_radius = theta
            best_pts = current_pts.copy()
        
        # Adaptive learning rate
        effective_lr = lr * (1.0 + 0.5 * np.sin(iteration * 0.01))
        
        # Apply gradient step
        new_pts = current_pts - effective_lr * grad
        
        # Project back to sphere
        new_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / new_norms
        
        # Check if it improved
        new_radius = _fast_covering_radius(new_pts)
        if new_radius < current_radius:
            current_pts = new_pts
            current_radius = _covering_radius(current_pts)
        else:
            # Try smaller step
            new_pts = current_pts - effective_lr * 0.5 * grad
            new_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / new_norms
            new_radius = _fast_covering_radius(new_pts)
            if new_radius < current_radius:
                current_pts = new_pts
                current_radius = _covering_radius(current_pts)
            else:
                # Random perturbation to escape
                idx = rng.randint(N)
                delta = rng.randn(3) * 0.02
                current_pts[idx] = current_pts[idx] + delta
                current_pts[idx] /= np.linalg.norm(current_pts[idx])
                current_radius = _covering_radius(current_pts)
    
    return best_pts, best_radius


def _local_search(pts, max_iter=5000, step_size=0.05, seed=42, use_fast_eval=True):
    """Local search with simulated annealing to minimize covering radius."""
    rng = np.random.RandomState(seed)
    current_pts = pts.copy()
    current_radius = _covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    T = step_size * 0.5
    T_min = 1e-6
    cooling = 0.9995
    
    for iteration in range(max_iter):
        if T < T_min or time.time() > 115:
            break
        
        idx = rng.randint(N)
        original = current_pts[idx].copy()
        
        delta = rng.randn(3) * step_size
        new_pt = original + delta
        new_pt /= np.linalg.norm(new_pt)
        
        current_pts[idx] = new_pt
        
        if use_fast_eval and iteration % 3 == 0:
            new_radius = _fast_covering_radius(current_pts)
        else:
            new_radius = _covering_radius(current_pts)
        
        if new_radius < current_radius:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
        else:
            prob = np.exp(-(new_radius - current_radius) / T)
            if rng.rand() < prob:
                current_radius = new_radius
            else:
                current_pts[idx] = original
        
        T *= cooling
    
    return best_pts, best_radius


def _refine(pts, max_iter=3000, step_size=0.01, seed=12345):
    """Fine-grained local search for final polish."""
    rng = np.random.RandomState(seed)
    current_pts = pts.copy()
    current_radius = _covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    T = step_size * 0.3
    T_min = 1e-7
    cooling = 0.999
    
    for iteration in range(max_iter):
        if T < T_min or time.time() > 115:
            break
        
        idx = rng.randint(N)
        original = current_pts[idx].copy()
        
        delta = rng.randn(3) * step_size
        new_pt = original + delta
        new_pt /= np.linalg.norm(new_pt)
        
        current_pts[idx] = new_pt
        new_radius = _covering_radius(current_pts)
        
        if new_radius < current_radius:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
        else:
            prob = np.exp(-(new_radius - current_radius) / T)
            if rng.rand() < prob:
                current_radius = new_radius
            else:
                current_pts[idx] = original
        
        T *= cooling
    
    return best_pts, best_radius


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    
    best_pts = None
    best_radius = np.inf
    
    # Multi-start with different initial configurations
    initializations = [
        ("fib0", _fibonacci_sphere(N, 0.0)),
        ("fib025", _fibonacci_sphere(N, 0.25)),
        ("fib05", _fibonacci_sphere(N, 0.5)),
        ("fib075", _fibonacci_sphere(N, 0.75)),
        ("rand42", _random_sphere(N, seed=42)),
        ("rand123", _random_sphere(N, seed=123)),
        ("rand999", _random_sphere(N, seed=999)),
        ("rand2024", _random_sphere(N, seed=2024)),
    ]
    
    # First pass: coarse local search on each initialization
    for name, init_pts in initializations:
        if time.time() - start_time > 70:
            break
        
        pts, radius = _local_search(init_pts, max_iter=2000, step_size=0.08, seed=hash(name) % 100000, use_fast_eval=True)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Second pass: gradient descent on the best solution
    if best_pts is not None and time.time() - start_time < 90:
        pts, radius = _gradient_descent(best_pts, max_iter=1000, lr=0.005, seed=42)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts
    
    # Third pass: finer local search
    if best_pts is not None and time.time() - start_time < 105:
        pts, radius = _local_search(best_pts, max_iter=3000, step_size=0.03, seed=77777, use_fast_eval=True)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts
    
    # Fourth pass: gradient descent again
    if best_pts is not None and time.time() - start_time < 110:
        pts, radius = _gradient_descent(best_pts, max_iter=500, lr=0.002, seed=88888)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts
    
    # Final fine refinement
    if best_pts is not None and time.time() - start_time < 117:
        pts, radius = _refine(best_pts, max_iter=2000, step_size=0.005, seed=12345)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
