import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _spherical_kmeans_init(n_clusters=50, n_iter=30, n_restarts=5, seed=42):
    """Spherical K-Means for better initial configuration."""
    rng = np.random.RandomState(seed)
    best_points = None
    best_cost = np.inf
    
    for restart in range(n_restarts):
        # Initialize centers randomly
        centers = rng.randn(n_clusters, 3)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        
        for _ in range(n_iter):
            # Assign each point to nearest center (on sphere)
            # Use negative dot product for distance
            dists = -np.dot(centers, centers.T)  # n x n
            # Actually we need to assign points, but we don't have points - we're generating them
            # For initialization, just use the centers as the configuration
            break
        
        # Use centers as initial configuration
        points = centers.copy()
        cost = _covering_radius(points)
        if cost < best_cost:
            best_cost = cost
            best_points = points.copy()
    
    return best_points

def _covering_radius(points):
    """
    Calculate the covering radius in radians for points on the unit sphere.
    Fast vectorized implementation.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Fallback for degenerate cases

    simplices = hull.simplices
    tri_pts = points[simplices]
    
    p0 = tri_pts[:, 0, :]
    p1 = tri_pts[:, 1, :]
    p2 = tri_pts[:, 2, :]
    
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    mask = norms > 1e-12
    if not np.any(mask):
        return 0.0
        
    # Distance from origin to the plane of the facet
    # d = |normal . p0| / |normal|
    # This is cos(alpha) where alpha is the angular circumradius
    d = np.zeros(len(normals))
    d[mask] = np.abs(np.sum(normals[mask] * p0[mask], axis=1)) / norms[mask]
    
    # Clip to valid range for arccos
    d = np.clip(d, 0.0, 1.0)
    
    alphas = np.arccos(d)
    return np.max(alphas)

def _covering_radius_smooth(points, beta=50.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi

    simplices = hull.simplices
    tri_pts = points[simplices]
    
    p0 = tri_pts[:, 0, :]
    p1 = tri_pts[:, 1, :]
    p2 = tri_pts[:, 2, :]
    
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    mask = norms > 1e-12
    if not np.any(mask):
        return 0.0
        
    d = np.zeros(len(normals))
    d[mask] = np.abs(np.sum(normals[mask] * p0[mask], axis=1)) / norms[mask]
    d = np.clip(d, 0.0, 1.0)
    
    alphas = np.arccos(d)
    
    # Log-sum-exp for smooth max
    max_alpha = np.max(alphas)
    shifted = alphas - max_alpha
    log_sum_exp = max_alpha + np.log(np.sum(np.exp(shifted)))
    return log_sum_exp / beta

def _smooth_obj_grad(params, beta=50.0):
    """
    Compute smooth objective and its gradient with respect to flattened params.
    Uses finite differences for the gradient (practical approach).
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    if not np.all(np.isfinite(pts)):
        return 1e6, np.zeros_like(params)
    
    f = _covering_radius_smooth(pts, beta=beta)
    
    # Finite difference gradient
    eps = 1e-6
    grad = np.zeros_like(params)
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + eps
        pts_p = _normalize(params.reshape(N, 3))
        f_p = _covering_radius_smooth(pts_p, beta=beta)
        params[i] = orig - eps
        pts_m = _normalize(params.reshape(N, 3))
        f_m = _covering_radius_smooth(pts_m, beta=beta)
        params[i] = orig
        grad[i] = (f_p - f_m) / (2 * eps)
    
    return f, grad

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _repulsive_relaxation(points, steps=500, alpha=0.1, time_limit=None, start_time=None):
    """
    Relax points using repulsive forces to spread them out.
    """
    if start_time is None:
        start_time = time.time()
    points = points.copy()
    n = points.shape[0]
    
    for step in range(steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Compute all pairwise differences
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        mask = dist_sq > 1e-12
        inv_dist3 = np.zeros_like(dist_sq)
        inv_dist3[mask] = 1.0 / (dist_sq[mask] * np.sqrt(dist_sq[mask]))
        
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        
        points += alpha * forces
        points = _normalize(points)
        
    return points

def _covering_force(points):
    """
    Compute forces that push points toward regions of high covering radius.
    """
    n = points.shape[0]
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: uniform repulsion
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        mask = dist_sq > 1e-12
        inv_dist3 = np.zeros_like(dist_sq)
        inv_dist3[mask] = 1.0 / (dist_sq[mask] * np.sqrt(dist_sq[mask]))
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        return forces
    
    simplices = hull.simplices
    tri_pts = points[simplices]
    
    p0 = tri_pts[:, 0, :]
    p1 = tri_pts[:, 1, :]
    p2 = tri_pts[:, 2, :]
    
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    mask = norms > 1e-12
    if not np.any(mask):
        return np.zeros_like(points)
        
    d = np.zeros(len(normals))
    d[mask] = np.abs(np.sum(normals[mask] * p0[mask], axis=1)) / norms[mask]
    d = np.clip(d, 0.0, 1.0)
    
    alphas = np.arccos(d)
    
    forces = np.zeros_like(points)
    
    for i, simplex in enumerate(simplices):
        if not mask[i]:
            continue
        weight = alphas[i]
        for vertex_idx in simplex:
            normal_dir = normals[i] / norms[i]
            forces[vertex_idx] += weight * normal_dir
    
    return forces

def _circumcenter_polish(points, max_iter=200, tol=1e-10, time_limit=None, start_time=None):
    """
    Polish by moving each point toward the circumcenter of its largest adjacent facet.
    This directly reduces the covering radius.
    """
    if start_time is None:
        start_time = time.time()
    points = points.copy()
    n = points.shape[0]
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        simplices = hull.simplices
        tri_pts = points[simplices]
        
        p0 = tri_pts[:, 0, :]
        p1 = tri_pts[:, 1, :]
        p2 = tri_pts[:, 2, :]
        
        v1 = p1 - p0
        v2 = p2 - p0
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1)
        
        mask = norms > 1e-12
        if not np.any(mask):
            break
        
        d = np.zeros(len(normals))
        d[mask] = np.abs(np.sum(normals[mask] * p0[mask], axis=1)) / norms[mask]
        d = np.clip(d, 0.0, 1.0)
        alphas = np.arccos(d)
        
        # For each point, find its largest adjacent facet and move toward its circumcenter
        updates = np.zeros_like(points)
        counts = np.zeros(n)
        
        for i, simplex in enumerate(simplices):
            if not mask[i]:
                continue
            weight = np.exp(50 * (alphas[i] - np.max(alphas[mask])))  # Softmax-like weighting
            for vertex_idx in simplex:
                # Circumcenter of the triangle in 3D
                # The circumcenter lies on the line through the centroid in the normal direction
                centroid = (p0[i] + p1[i] + p2[i]) / 3.0
                normal_dir = normals[i] / norms[i]
                # Distance from origin to circumcenter
                # For a triangle on the unit sphere, the circumcenter is at distance 1/cos(alpha) from origin
                # But we want the point on the sphere, so just use the normalized centroid projected
                circumcenter = centroid / np.linalg.norm(centroid)
                updates[vertex_idx] += weight * (circumcenter - points[vertex_idx])
                counts[vertex_idx] += weight
        
        # Apply updates with damping
        for i in range(n):
            if counts[i] > 1e-10:
                updates[i] /= counts[i]
        
        # Damped update
        step_size = 0.1
        points = points + step_size * updates
        points = _normalize(points)
        
        # Check convergence
        if np.max(np.abs(updates)) < tol:
            break
    
    return points

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    def time_remaining():
        return deadline - time.time()

    # 1. Initial Configuration: Try both Fibonacci and K-Means
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_points = _normalize(fib_points)
    
    km_points = _spherical_kmeans_init(n_clusters=N, n_iter=20, n_restarts=3, seed=42)
    
    best_points = fib_points.copy()
    best_radius = _covering_radius(best_points)
    
    if _covering_radius(km_points) < best_radius:
        best_points = km_points
        best_radius = _covering_radius(best_points)
    
    # Relax the initial configuration
    relaxed = _repulsive_relaxation(best_points, steps=800, alpha=0.05, time_limit=4, start_time=start_time)
    rad = _covering_radius(relaxed)
    if rad < best_radius:
        best_points = relaxed
        best_radius = rad

    # 2. Define Objective for Optimizer
    def objective(params):
        if time.time() > deadline:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _covering_radius(pts)

    def objective_smooth(params):
        if time.time() > deadline:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _covering_radius_smooth(pts, beta=50.0)

    def optimize_from(points):
        nonlocal best_points, best_radius
        params0 = points.flatten()
        
        # Phase 1: BFGS with smooth surrogate
        try:
            result_bfgs = minimize(objective_smooth, params0, method='BFGS',
                                   options={'maxiter': 300, 'gtol': 1e-8, 'disp': False})
            if result_bfgs.fun < best_radius:
                best_points = _normalize(result_bfgs.x.reshape(N, 3))
                best_radius = _covering_radius(best_points)
        except Exception:
            pass
        
        # Phase 2: Circumcenter polish
        if time_remaining() > 10:
            polished = _circumcenter_polish(best_points, max_iter=100, time_limit=5, start_time=start_time)
            rad = _covering_radius(polished)
            if rad < best_radius:
                best_points = polished
                best_radius = rad
        
        # Phase 3: Nelder-Mead polish with exact objective
        if time_remaining() > 10:
            try:
                result_nm = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if result_nm.fun < best_radius:
                    best_points = _normalize(result_nm.x.reshape(N, 3))
                    best_radius = result_nm.fun
            except Exception:
                pass

    # Run optimization from the best relaxed point
    optimize_from(best_points)
    
    # 4. Random Restarts
    while time_remaining() > 15:
        # Perturb current best
        noise_scale = 0.25
        noise = np.random.normal(0, noise_scale, size=best_points.shape)
        perturbed = best_points + noise
        perturbed = _normalize(perturbed)
        
        # Quick relaxation
        relaxed = _repulsive_relaxation(perturbed, steps=150, alpha=0.05, 
                                        time_limit=2, start_time=start_time)
        rad = _covering_radius(relaxed)
        
        if rad < best_radius:
            best_points = relaxed
            best_radius = rad
            optimize_from(best_points)
        elif time_remaining() > 25:
            # Even if not better, try a small polish from perturbed
            small_polish = _circumcenter_polish(perturbed, max_iter=50, time_limit=2, start_time=start_time)
            rad2 = _covering_radius(small_polish)
            if rad2 < best_radius:
                best_points = small_polish
                best_radius = rad2
                optimize_from(best_points)

    # 5. Final Polish
    if time_remaining() > 8:
        # Circumcenter polish first
        polished = _circumcenter_polish(best_points, max_iter=200, time_limit=4, start_time=start_time)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_points = polished
            best_radius = rad
        
        # Then Nelder-Mead with tight tolerance
        if time_remaining() > 4:
            try:
                result_final = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if result_final.fun < best_radius:
                    best_points = _normalize(result_final.x.reshape(N, 3))
                    best_radius = result_final.fun
            except Exception:
                pass

    return best_points
