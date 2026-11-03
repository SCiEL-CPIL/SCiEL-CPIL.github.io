"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import ConvexHull


def _normalize(pts):
    """Normalize points to unit sphere, handling near-zero norms."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius_deg(pts):
    """
    Compute the covering radius in degrees for a set of points on the unit sphere.
    Uses the convex hull circumradius method.
    """
    pts = np.asarray(pts, dtype=np.float64)
    pts = _normalize(pts)
    
    # Compute convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use a grid-based approach
        return _covering_radius_grid(pts)
    
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter of triangle abc in 3D
        ab = b - a
        ac = c - a
        cross = np.cross(ab, ac)
        cross_norm_sq = np.dot(cross, cross)
        
        if cross_norm_sq < 1e-14:
            continue
        
        # Solve for circumcenter
        A = np.array([[np.dot(ab, ab), np.dot(ab, ac)],
                      [np.dot(ac, ab), np.dot(ac, ac)]])
        rhs = np.array([np.dot(ab, ab) / 2, np.dot(ac, ac) / 2])
        
        det = A[0,0] * A[1,1] - A[0,1] * A[1,0]
        if abs(det) < 1e-14:
            continue
        
        alpha = (rhs[0] * A[1,1] - rhs[1] * A[0,1]) / det
        beta = (A[0,0] * rhs[1] - A[1,0] * rhs[0]) / det
        
        t = alpha * ab + beta * ac
        circumcenter_3d = a + t
        circumcenter_norm = np.linalg.norm(circumcenter_3d)
        
        if circumcenter_norm < 1e-12:
            continue
        
        circumcenter_sphere = circumcenter_3d / circumcenter_norm
        
        cos_angle = np.dot(circumcenter_sphere, a)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.acos(cos_angle)
        
        if angle > max_circumradius:
            max_circumradius = angle
    
    return math.degrees(max_circumradius)


def _covering_radius_grid(pts, n_theta=150, n_phi=300):
    """Fallback covering radius computation using a grid."""
    pts = np.asarray(pts, dtype=np.float64)
    pts = _normalize(pts)
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    
    grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    max_min_angle = 0.0
    batch_size = 5000
    for start in range(0, len(grid_pts), batch_size):
        end = min(start + batch_size, len(grid_pts))
        batch = grid_pts[start:end]
        dots = batch @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)
        angles = np.arccos(min_dots)
        max_min_angle = max(max_min_angle, np.max(angles))
    
    return math.degrees(max_min_angle)


def _fibonacci_spiral(n, offset=0.0):
    """Generate a Fibonacci spiral configuration with optional offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + offset)
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)


def _lloyd_step(pts, beta=20.0):
    """
    Perform one Lloyd iteration: move each point to the centroid of its
    Voronoi cell (approximated using the convex hull).
    
    For sphere covering, we use a smooth-max approach to find the worst
    region and move points toward it.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Compute the convex hull to find facets
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts
    
    # For each facet, compute the circumcenter and use it as a target
    # for the vertices of that facet
    new_pts = pts.copy()
    
    # Compute pairwise dot products for force calculation
    dots = pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    
    # For each point, compute a weighted average of nearby circumcenters
    # weighted by how "bad" the facet is
    weights = np.zeros(n)
    targets = np.zeros((n, 3))
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        ab = b - a
        ac = c - a
        cross = np.cross(ab, ac)
        cross_norm_sq = np.dot(cross, cross)
        
        if cross_norm_sq < 1e-14:
            continue
        
        A_mat = np.array([[np.dot(ab, ab), np.dot(ab, ac)],
                          [np.dot(ac, ab), np.dot(ac, ac)]])
        rhs = np.array([np.dot(ab, ab) / 2, np.dot(ac, ac) / 2])
        
        det = A_mat[0,0] * A_mat[1,1] - A_mat[0,1] * A_mat[1,0]
        if abs(det) < 1e-14:
            continue
        
        alpha = (rhs[0] * A_mat[1,1] - rhs[1] * A_mat[0,1]) / det
        beta_coeff = (A_mat[0,0] * rhs[1] - A_mat[1,0] * rhs[0]) / det
        
        t = alpha_coeff * ab + beta_coeff * ac
        circumcenter_3d = a + t
        circumcenter_norm = np.linalg.norm(circumcenter_3d)
        
        if circumcenter_norm < 1e-12:
            continue
        
        circumcenter_sphere = circumcenter_3d / circumcenter_norm
        
        cos_angle = np.dot(circumcenter_sphere, a)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.acos(cos_angle)
        
        # Weight by exponential of the angle (smooth max)
        w = np.exp(beta * (angle - 11.0))  # Scale relative to ~11 degrees
        
        for idx in simplex:
            weights[idx] += w
            targets[idx] += w * circumcenter_sphere
    
    # Move points toward their weighted target
    for i in range(n):
        if weights[i] > 1e-10:
            target = targets[i] / weights[i]
            # Move a fraction of the way
            new_pts[i] = pts[i] + 0.1 * (target - pts[i])
    
    return _normalize(new_pts)


def _gradient_descent(pts, time_budget=30.0, lr=0.01, beta=15.0):
    """
    Gradient descent on a smooth approximation of the covering radius.
    Uses the smooth-max of facet circumradii.
    """
    pts = _normalize(pts)
    start_time = time.time()
    best_pts = pts.copy()
    best_radius = _covering_radius_deg(pts)
    
    # Precompute for efficiency
    for iteration in range(500):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Compute convex hull
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Compute gradient of smooth-max covering radius
        grad = np.zeros_like(pts)
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            ab = b - a
            ac = c - a
            cross = np.cross(ab, ac)
            cross_norm_sq = np.dot(cross, cross)
            
            if cross_norm_sq < 1e-14:
                continue
            
            A_mat = np.array([[np.dot(ab, ab), np.dot(ab, ac)],
                              [np.dot(ac, ab), np.dot(ac, ac)]])
            rhs = np.array([np.dot(ab, ab) / 2, np.dot(ac, ac) / 2])
            
            det = A_mat[0,0] * A_mat[1,1] - A_mat[0,1] * A_mat[1,0]
            if abs(det) < 1e-14:
                continue
            
            alpha = (rhs[0] * A_mat[1,1] - rhs[1] * A_mat[0,1]) / det
            beta_coeff = (A_mat[0,0] * rhs[1] - A_mat[1,0] * rhs[0]) / det
            
            t = alpha * ab + beta_coeff * ac
            circumcenter_3d = a + t
            circumcenter_norm = np.linalg.norm(circumcenter_3d)
            
            if circumcenter_norm < 1e-12:
                continue
            
            circumcenter_sphere = circumcenter_3d / circumcenter_norm
            
            cos_angle = np.dot(circumcenter_sphere, a)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle)
            
            # Smooth max weight
            w = np.exp(beta * (angle - 11.0))
            
            # Gradient: move vertices away from circumcenter if it's bad
            # Actually, move vertices TOWARD circumcenter to reduce the radius
            for idx in simplex:
                # The gradient of the angle w.r.t. pts[idx] is approximately
                # in the direction that moves pts[idx] away from the circumcenter
                # But we want to MINIMIZE the radius, so move toward circumcenter
                grad[idx] += w * (pts[idx] - circumcenter_sphere)
        
        # Normalize gradient and take step
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        new_pts = pts - lr * grad / grad_norm
        new_pts = _normalize(new_pts)
        
        new_radius = _covering_radius_deg(new_pts)
        
        if new_radius < best_radius:
            best_pts = new_pts.copy()
            best_radius = new_radius
            pts = new_pts
        else:
            lr *= 0.95
            if lr < 1e-5:
                break
    
    return best_pts


def _simulated_annealing(pts, time_budget=40.0):
    """
    Simulated annealing: perturb points and accept worse solutions with
    decreasing probability.
    """
    pts = _normalize(pts)
    start_time = time.time()
    best_pts = pts.copy()
    best_radius = _covering_radius_deg(pts)
    
    temperature = 0.5
    cooling_rate = 0.995
    
    for iteration in range(3000):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Perturb a random subset of points
        n_perturb = max(1, N // 10)
        indices = np.random.choice(N, n_perturb, replace=False)
        
        new_pts = pts.copy()
        for idx in indices:
            # Random perturbation on the sphere
            random_dir = np.random.randn(3)
            random_dir = random_dir / np.linalg.norm(random_dir)
            new_pts[idx] = pts[idx] + temperature * random_dir
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_radius = _covering_radius_deg(new_pts)
        
        # Accept or reject
        if new_radius < best_radius or np.random.random() < np.exp(-(new_radius - best_radius) / temperature):
            pts = new_pts
            if new_radius < best_radius:
                best_pts = new_pts.copy()
                best_radius = new_radius
        
        temperature *= cooling_rate
    
    return best_pts


def _spectral_force_step(pts, time_budget=20.0):
    """
    Force-directed optimization using spectral methods.
    Minimize variance of pairwise angular distances to promote uniformity.
    """
    pts = _normalize(pts)
    start_time = time.time()
    best_pts = pts.copy()
    best_radius = _covering_radius_deg(pts)
    
    lr = 0.02
    
    for iteration in range(2000):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Compute pairwise angular distances
        dots = pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        
        # Target: all pairwise distances should be close to the mean
        # For 130 points, mean distance ~ 60 degrees
        mean_angle = np.mean(angles)
        
        # Force: each point is pushed toward positions that equalize distances
        forces = np.zeros_like(pts)
        
        for i in range(N):
            # Compute gradient of variance w.r.t. pts[i]
            # Approximate: move pts[i] to reduce variance of its distances
            dists_i = angles[i, :]
            var_i = np.var(dists_i)
            
            # Gradient approximation: move toward centroid of neighbors
            # weighted by distance deviation
            weights = np.exp(-np.abs(dists_i - mean_angle) / 0.5)
            weights[i] = 0
            
            if np.sum(weights) > 1e-10:
                target = np.sum(weights[:, np.newaxis] * pts, axis=0) / np.sum(weights)
                forces[i] = target - pts[i]
        
        # Normalize and apply
        force_norm = np.linalg.norm(forces)
        if force_norm < 1e-10:
            break
        
        new_pts = pts + lr * forces / force_norm
        new_pts = _normalize(new_pts)
        
        new_radius = _covering_radius_deg(new_pts)
        
        if new_radius < best_radius:
            best_pts = new_pts.copy()
            best_radius = new_radius
            pts = new_pts
        else:
            lr *= 0.98
            if lr < 1e-5:
                break
    
    return best_pts


def construct():
    """
    Construct a set of 130 points on the unit sphere that minimizes the covering radius.
    Returns an array of shape (130, 3) of unit vectors.
    """
    start_time = time.time()
    total_time_budget = 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_radius = np.inf
    
    # Multi-restart strategy
    n_restarts = 5
    restart_time = total_time_budget / n_restarts
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > total_time_budget - 10:
            break
        
        # Generate initial configuration
        if restart == 0:
            pts = _fibonacci_spiral(N)
        elif restart == 1:
            pts = _random_sphere(N, seed=restart * 100 + 1)
        elif restart == 2:
            pts = _fibonacci_spiral(N, offset=restart * 0.3)
        elif restart == 3:
            pts = _random_sphere(N, seed=restart * 100 + 42)
        else:
            pts = _fibonacci_spiral(N, offset=0.5)
        
        # Phase 1: Lloyd-like refinement
        time_left = total_time_budget - (time.time() - start_time)
        if time_left > 15:
            pts = _lloyd_step(pts, beta=15.0)
            # Multiple Lloyd steps
            for _ in range(3):
                time_left = total_time_budget - (time.time() - start_time)
                if time_left < 10:
                    break
                pts = _lloyd_step(pts, beta=15.0)
        
        # Phase 2: Gradient descent
        time_left = total_time_budget - (time.time() - start_time)
        if time_left > 10:
            pts = _gradient_descent(pts, time_budget=min(time_left - 5, 25))
        
        # Phase 3: Simulated annealing
        time_left = total_time_budget - (time.time() - start_time)
        if time_left > 10:
            pts = _simulated_annealing(pts, time_budget=min(time_left - 5, 30))
        
        # Evaluate
        radius = _covering_radius_deg(pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Final fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
