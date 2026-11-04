import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _facet_circumradii(points, simplices):
    """
    Compute circumradii and circumcenters for all hull facets.
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        valid: (M,) boolean array of valid facets
    """
    M = len(simplices)
    if M == 0:
        return np.array([]), np.empty((0,3)), np.array([], dtype=bool)
        
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        radii[valid] = dist
    
    return radii, centers, valid

def _covering_radius_rad(points):
    """
    Compute the covering radius of a set of points on the unit sphere (in radians).
    """
    try:
        hull = ConvexHull(points)
        radii, centers, valid = _facet_circumradii(points, hull.simplices)
        if not np.any(valid):
            return np.pi
        return float(np.max(radii[valid]))
    except Exception:
        return np.pi

def _smooth_max_objective_and_grad(x_flat, temperature=0.01):
    """
    Compute smooth max of facet circumradii and its gradient.
    x_flat is a flattened array of shape (3*N,).
    """
    points = x_flat.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1.5, np.zeros(3 * N)
    
    radii, centers, valid = _facet_circumradii(points, hull.simplices)
    
    if not np.any(valid):
        return 1.5, np.zeros(3 * N)
    
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return 0.0, np.zeros(3 * N)
    
    max_r = np.max(r_valid)
    if max_r < 1e-15:
        return 0.0, np.zeros(3 * N)
    
    # Smooth max (log-sum-exp)
    shifted = (r_valid - max_r) / temperature
    exps = np.exp(shifted)
    smooth_val = max_r + temperature * np.log(np.sum(exps))
    
    weights = exps / np.sum(exps)
    
    grad_flat = np.zeros(3 * N)
    
    for k in range(len(f_valid)):
        w_k = weights[k]
        if w_k < 1e-8:
            continue
        
        tri = f_valid[k]
        c = c_valid[k]
        r_k = r_valid[k]
        
        if r_k < 1e-12:
            continue
        
        cos_r = np.dot(c, points[tri[0]])
        sin_r = np.sqrt(max(1e-15, 1.0 - cos_r * cos_r))
        
        # Gradient of arccos(c·p) w.r.t. p is -c/sin_r
        # Project to tangent space
        for idx in tri:
            p = points[idx]
            grad_dir = -c / sin_r
            proj = np.dot(grad_dir, p)
            tangent_grad = grad_dir - proj * p
            grad_flat[3*idx:3*idx+3] += w_k * tangent_grad
    
    return float(smooth_val), grad_flat

def _electrostatic_refine(pts, n_iters=100, lr=0.005):
    """Electrostatic repulsion to spread points evenly."""
    pts = pts.copy()
    for _ in range(n_iters):
        D = pts @ pts.T
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        forces /= norms
        pts = pts + lr * forces
        pts = _normalize(pts)
    return pts

def _nelder_mead_polish(points, max_facets=20, max_iter=300, time_limit=10.0):
    """
    Polish using Nelder-Mead on the vertices of the largest facets.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = _covering_radius_rad(points)
    
    try:
        hull = ConvexHull(points)
        radii, centers, valid = _facet_circumradii(points, hull.simplices)
        if not np.any(valid):
            return best_points
    except Exception:
        return best_points
    
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return best_points
    
    top_k = min(max_facets, len(r_valid))
    top_indices = np.argsort(r_valid)[-top_k:]
    
    vertex_indices = set()
    for idx in top_indices:
        for v in f_valid[idx]:
            vertex_indices.add(v)
    
    vertex_indices = list(vertex_indices)
    
    if len(vertex_indices) == 0:
        return best_points
    
    def objective(x_flat):
        if time.time() - start > time_limit:
            return np.pi
        x = x_flat.reshape(-1, 3)
        new_points = points.copy()
        for i, idx in enumerate(vertex_indices):
            new_points[idx] = x[i]
        new_points = _normalize(new_points)
        if not np.all(np.isfinite(new_points)):
            return np.pi
        return _covering_radius_rad(new_points)
    
    x0 = points[vertex_indices].flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < best_radius:
            x_opt = result.x.reshape(-1, 3)
            best_points = points.copy()
            for i, idx in enumerate(vertex_indices):
                best_points[idx] = x_opt[i]
            best_points = _normalize(best_points)
            best_radius = result.fun
    except Exception:
        pass
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Phase 1: Initialization
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)

    # Phase 2: Electrostatic Refinement
    points = _electrostatic_refine(points, n_iters=50, lr=0.005)
    
    best_points = points.copy()
    best_rad = _covering_radius_rad(best_points)

    # Phase 3: Smooth Surrogate Optimization with L-BFGS-B
    if time.time() < deadline - 30:
        x0 = points.flatten()
        
        # Use a moderate temperature for the smooth max
        temp = 0.01
        
        try:
            result = minimize(
                _smooth_max_objective_and_grad,
                x0,
                args=(temp,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_rad:
                new_pts = _normalize(result.x.reshape(N, 3))
                new_rad = _covering_radius_rad(new_pts)
                if new_rad < best_rad and np.all(np.isfinite(new_pts)):
                    best_rad = new_rad
                    best_points = new_pts.copy()
        except Exception:
            pass

    # Phase 4: Nelder-Mead Polish on Worst Facets
    if time.time() < deadline - 10:
        best_points = _nelder_mead_polish(best_points, max_facets=30, max_iter=200, time_limit=10.0)
        best_rad = _covering_radius_rad(best_points)

    # Phase 5: Final High-Precision Polish
    if time.time() < deadline - 2:
        # Another round of L-BFGS with tighter tolerance
        x0 = best_points.flatten()
        temp = 0.005
        try:
            result = minimize(
                _smooth_max_objective_and_grad,
                x0,
                args=(temp,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_rad:
                new_pts = _normalize(result.x.reshape(N, 3))
                new_rad = _covering_radius_rad(new_pts)
                if new_rad < best_rad and np.all(np.isfinite(new_pts)):
                    best_rad = new_rad
                    best_points = new_pts.copy()
        except Exception:
            pass

    # Final Nelder-Mead polish
    if time.time() < deadline - 1:
        best_points = _nelder_mead_polish(best_points, max_facets=10, max_iter=100, time_limit=5.0)
        best_rad = _covering_radius_rad(best_points)

    return best_points
