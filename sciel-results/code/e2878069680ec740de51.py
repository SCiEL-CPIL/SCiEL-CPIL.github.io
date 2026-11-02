import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return points / norms

def _spherical_circumcenters(hull, points):
    """
    Compute circumcenters and radii for all facets of the convex hull.
    Returns:
        centers: (M, 3) array of circumcenters (unit vectors)
        radii: (M,) array of angular radii (radians)
    """
    facets = hull.simplices
    centers = np.zeros((len(facets), 3))
    radii = np.zeros(len(facets))
    
    for i, face in enumerate(facets):
        v0 = points[face[0]]
        v1 = points[face[1]]
        v2 = points[face[2]]
        
        n = np.cross(v1 - v0, v2 - v1)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            centers[i] = v0
            radii[i] = 0
            continue
        
        center = n / norm
        
        eq = hull.equations[i]
        hull_normal = eq[:3]
        
        if np.dot(center, hull_normal) < 0:
            center = -center
            
        centers[i] = center
        radii[i] = np.arccos(np.clip(np.dot(center, v0), -1.0, 1.0))
        
    return centers, radii

def _covering_radius(points):
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    centers, radii = _spherical_circumcenters(hull, points)
    return np.max(radii)

def _soft_max_radius(points, beta=50.0):
    """
    Smooth surrogate for max facet radius using log-sum-exp.
    Returns (soft_max_value, gradients w.r.t. points)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    centers, radii = _spherical_circumcenters(hull, points)
    
    # Log-sum-exp for soft max
    max_r = np.max(radii)
    shifted = radii - max_r
    weights = np.exp(beta * shifted)
    weights /= weights.sum()
    soft_val = max_r - np.log(weights.sum()) / beta
    
    # Gradient
    grads = np.zeros_like(points)
    facets = hull.simplices
    
    for i in range(len(facets)):
        w = weights[i]
        if w < 1e-15:
            continue
        face = facets[i]
        center = centers[i]
        v0 = points[face[0]]
        
        for j in face:
            vj = points[j]
            dot_cj = np.dot(center, vj)
            sin_rj = np.sqrt(max(1.0 - dot_cj * dot_cj, 1e-15))
            grads[j] += w * (-center / sin_rj)
    
    return soft_val, grads

def _repulsion_relax(points, n_iter=100, timeout=5.0):
    """
    Repulsion-based relaxation to spread points evenly.
    Uses a potential that penalizes close pairs.
    """
    start_time = time.time()
    points = points.copy()
    for it in range(n_iter):
        if time.time() - start_time > timeout:
            break
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(i+1, N):
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < 1e-8:
                    d = np.random.randn(3) * 1e-8
                    dist = np.linalg.norm(d)
                # Repulsive force
                f = d / (dist**3)
                forces[i] += f
                forces[j] -= f
        # Move points
        step = 0.01 * (1.0 - it/n_iter)
        points += step * forces
        points = _normalize(points)
    return points

def _lloyd_iteration(points, n_iter=50, timeout=5.0):
    """
    Lloyd's algorithm: move each point to the centroid of its Voronoi cell (on sphere).
    Approximated by moving to the average of nearest neighbors.
    """
    start_time = time.time()
    points = points.copy()
    for it in range(n_iter):
        if time.time() - start_time > timeout:
            break
        # Compute pairwise distances
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i, j] = np.arccos(np.clip(np.dot(points[i], points[j]), -1.0, 1.0))
        # For each point, find its Voronoi region centroid (approximate by averaging close neighbors)
        new_points = np.zeros_like(points)
        for i in range(N):
            # Find k nearest neighbors
            order = np.argsort(dists[i])
            k = 5
            neighbors = points[order[1:k+1]]  # exclude self
            # Weight by inverse distance
            w = 1.0 / np.maximum(dists[i, order[1:k+1]], 1e-8)
            w /= w.sum()
            new_points[i] = np.sum(neighbors * w[:, np.newaxis], axis=0)
            new_points[i] = new_points[i] / np.linalg.norm(new_points[i])
        # Blend
        alpha = 0.5 * (1.0 - it/n_iter)
        points = _normalize((1-alpha) * points + alpha * new_points)
    return points

def _optimize_lbfgs(points, timeout=20.0, beta_start=20.0, beta_end=100.0, n_stages=3):
    """
    Optimize using L-BFGS-B on the soft-max surrogate with annealing beta.
    """
    start_time = time.time()
    points = points.copy().flatten()
    best_val = np.pi
    
    # Anneal beta
    betas = np.linspace(beta_start, beta_end, n_stages)
    for beta in betas:
        if time.time() - start_time > timeout:
            break
        stage_timeout = min(timeout - (time.time() - start_time), 10.0)
        
        def objective(x, beta=beta):
            if time.time() - start_time > timeout:
                raise Exception("timeout")
            pts = x.reshape(N, 3)
            pts = _normalize(pts)
            val, _ = _soft_max_radius(pts, beta=beta)
            return val
        
        def gradient(x, beta=beta):
            if time.time() - start_time > timeout:
                raise Exception("timeout")
            pts = x.reshape(N, 3)
            pts = _normalize(pts)
            _, grads = _soft_max_radius(pts, beta=beta)
            # Project gradient onto tangent space of sphere
            proj_grads = np.zeros_like(pts)
            for i in range(N):
                pi = pts[i]
                gi = grads[i]
                proj_grads[i] = gi - np.dot(gi, pi) * pi
            return proj_grads.flatten()
        
        try:
            result = minimize(
                objective, points, jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 300}
            )
            opt_pts = result.x.reshape(N, 3)
            opt_pts = _normalize(opt_pts)
            val = _covering_radius(opt_pts)
            if val < best_val:
                best_val = val
                points = opt_pts.flatten()
        except Exception:
            pass
    
    opt_pts = points.reshape(N, 3)
    opt_pts = _normalize(opt_pts)
    return opt_pts, _covering_radius(opt_pts)

def _local_polish(points, timeout=5.0):
    """
    Final polish: targeted moves of worst facet vertices toward circumcenters.
    """
    start_time = time.time()
    points = points.copy()
    
    for _ in range(50):
        if time.time() - start_time > timeout:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        centers, radii = _spherical_circumcenters(hull, points)
        max_r = np.max(radii)
        
        if max_r < 1e-8:
            break
        
        # Find worst facet
        worst_idx = np.argmax(radii)
        face = hull.simplices[worst_idx]
        center = centers[worst_idx]
        
        # Move vertices of worst facet toward center
        step = 0.01 * (1.0 - 0.9 * _ / 50.0)
        for j in face:
            v = points[j]
            dir_vec = center - v
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                points[j] += step * dir_vec / norm_dir
        
        # Renormalize
        points = _normalize(points)
    
    return points, _covering_radius(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_points = None
    best_radius = np.pi
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    starts = [init_fib]
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    for _ in range(15):
        if time.time() > deadline - 10:
            break
        p = rng.normal(size=(N, 3))
        p = _normalize(p)
        starts.append(p)
        
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        remaining = deadline - time.time()
        if remaining < 20:
            break
            
        # Phase 0: Pre-processing with repulsion and Lloyd
        if time.time() < deadline - 15:
            start = _repulsion_relax(start, n_iter=50, timeout=min(remaining - 15, 3.0))
            if time.time() < deadline - 10:
                start = _lloyd_iteration(start, n_iter=30, timeout=min(remaining - 10, 3.0))
        
        # Phase 1: L-BFGS on soft-max surrogate
        lbfgs_timeout = min(remaining - 10, 30.0)
        opt_pts, radius = _optimize_lbfgs(start, timeout=lbfgs_timeout)
        
        # Phase 2: Local polish
        if time.time() < deadline - 5:
            polish_timeout = min(deadline - time.time() - 3, 5.0)
            opt_pts, radius = _local_polish(opt_pts, timeout=polish_timeout)
        
        if radius < best_radius:
            best_radius = radius
            best_points = opt_pts.copy()
            
    if best_points is None:
        return init_fib
        
    return best_points
