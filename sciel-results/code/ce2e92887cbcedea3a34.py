import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares, minimize
import time

N = 130

def _covering_radius(points):
    """Compute the covering radius in radians."""
    try:
        hull = ConvexHull(points)
        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            n = np.cross(v1 - v2, v1 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
            rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            if rad > max_rad:
                max_rad = rad
    except Exception:
        return np.pi
    return max_rad

def _facet_circumradii(points):
    """Compute circumradii for all hull facets."""
    hull = ConvexHull(points)
    radii = []
    simplices = []
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        n = np.cross(v1 - v2, v1 - v3)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        c = n / norm_n
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        radii.append(rad)
        simplices.append(simplex)
    return np.array(radii), np.array(simplices)

def _smooth_max(radii, beta=50.0):
    """Log-sum-exp smooth approximation of max(radii)."""
    m = np.max(radii)
    if m < -1e10:
        return 0.0
    shifted = radii - m
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -50.0, 0.0)
    s = np.sum(np.exp(beta * shifted))
    return m + np.log(s) / beta

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate objective: log-sum-exp of facet circumradii.
    x: flat array of 2N tangent-half-angle parameters.
    """
    u = x[:N]
    v = x[N:2*N]
    denom = 1.0 + u*u + v*v
    pts = np.empty((N, 3))
    pts[:, 0] = 2.0 * u / denom
    pts[:, 1] = 2.0 * v / denom
    pts[:, 2] = (1.0 - u*u - v*v) / denom
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 10.0  # Large penalty
    
    radii = []
    for simplex in hull.simplices:
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        n = np.cross(v1 - v2, v1 - v3)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        c = n / norm_n
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        radii.append(rad)
    
    if not radii:
        return 10.0
    
    return _smooth_max(np.array(radii), beta)

def _smooth_objective_grad(x, beta=50.0):
    """
    Numerical gradient of smooth objective using central differences.
    Returns (objective_value, gradient).
    """
    f0 = _smooth_objective(x, beta)
    grad = np.zeros_like(x)
    eps = 1e-6
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        fp = _smooth_objective(xp, beta)
        fm = _smooth_objective(xm, beta)
        grad[i] = (fp - fm) / (2.0 * eps)
    return f0, grad

def _lsq_residuals(x, theta):
    """
    Residuals for least-squares: each facet's circumradius should equal theta.
    x: flat array of 2N coordinates (tan-half-angle params for sphere)
    theta: target covering radius
    """
    u = x[:N]
    v = x[N:2*N]
    denom = 1.0 + u*u + v*v
    pts = np.empty((N, 3))
    pts[:, 0] = 2.0 * u / denom
    pts[:, 1] = 2.0 * v / denom
    pts[:, 2] = (1.0 - u*u - v*v) / denom
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.ones(100) * 10.0
    
    radii = []
    for simplex in hull.simplices:
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        n = np.cross(v1 - v2, v1 - v3)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            radii.append(theta)
            continue
        c = n / norm_n
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        radii.append(rad)
    
    radii = np.array(radii)
    return radii - theta

def construct():
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    
    start_time = time.time()
    max_time = 145.0  # Safety margin before 150s kill
    
    def to_tanhalf(pts):
        u = np.empty(N)
        v = np.empty(N)
        for i in range(N):
            x, y, z = pts[i]
            denom = 1.0 - z
            if denom < 1e-10:
                denom = 1e-10
            u[i] = x / denom
            v[i] = y / denom
        return np.concatenate([u, v])
    
    def from_tanhalf(x):
        u = x[:N]
        v = x[N:2*N]
        denom = 1.0 + u*u + v*v
        pts = np.empty((N, 3))
        pts[:, 0] = 2.0 * u / denom
        pts[:, 1] = 2.0 * v / denom
        pts[:, 2] = (1.0 - u*u - v*v) / denom
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts / norms
    
    # Phase 1: Greedy refinement
    step_size = 0.05
    min_step_size = 0.001
    fail_count = 0
    
    while time.time() - start_time < max_time - 40:
        prev_points = points.copy()
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        facet_radii = []
        facet_indices = []
        for idx, simplex in enumerate(hull.simplices):
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            n = np.cross(v1 - v2, v1 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
            rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            facet_radii.append(rad)
            facet_indices.append(simplex)
            
        if not facet_radii:
            break
            
        sorted_indices = np.argsort(facet_radii)[::-1]
        
        K = 5
        moved_indices = set()
        for k in range(min(K, len(sorted_indices))):
            facet_idx = sorted_indices[k]
            simplex = facet_indices[facet_idx]
            
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            n = np.cross(v1 - v2, v1 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
                
            for j in range(3):
                v_idx = simplex[j]
                if v_idx in moved_indices:
                    continue
                v = points[v_idx]
                new_v = v + step_size * (c - v)
                norm_v = np.linalg.norm(new_v)
                if norm_v > 1e-10:
                    points[v_idx] = new_v / norm_v
                    moved_indices.add(v_idx)
                    
        current_rad = _covering_radius(points)
        
        if current_rad < best_rad:
            best_rad = current_rad
            best_points = points.copy()
            fail_count = 0
        else:
            points = prev_points
            fail_count += 1
            if fail_count > 5:
                step_size *= 0.8
                fail_count = 0
                if step_size < min_step_size:
                    break

    # Phase 2: Smooth surrogate optimization with L-BFGS-B
    if time.time() - start_time < max_time - 20:
        points = best_points.copy()
        x0 = to_tanhalf(points)
        
        # Use moderate beta for stability
        beta_val = 30.0
        
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta_val,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_rad:
                new_pts = from_tanhalf(result.x)
                new_rad = _covering_radius(new_pts)
                if new_rad < best_rad and np.all(np.isfinite(new_pts)):
                    best_rad = new_rad
                    best_points = new_pts.copy()
        except Exception:
            pass
    
    # Phase 2b: Levenberg-Marquardt nonlinear least-squares
    if time.time() - start_time < max_time - 15:
        points = best_points.copy()
        
        for theta_attempt in [best_rad, best_rad * 0.98, best_rad * 1.02]:
            if time.time() - start_time > max_time - 10:
                break
            
            x0 = to_tanhalf(points)
            
            try:
                result = least_squares(
                    _lsq_residuals,
                    x0,
                    args=(theta_attempt,),
                    method='lm',
                    max_nfev=5000,
                    xtol=1e-10,
                    ftol=1e-10
                )
                
                u = result.x[:N]
                v = result.x[N:2*N]
                denom = 1.0 + u*u + v*v
                new_pts = np.empty((N, 3))
                new_pts[:, 0] = 2.0 * u / denom
                new_pts[:, 1] = 2.0 * v / denom
                new_pts[:, 2] = (1.0 - u*u - v*v) / denom
                new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
                
                if np.all(np.isfinite(new_pts)) and np.all(np.linalg.norm(new_pts, axis=1) > 0.99):
                    new_rad = _covering_radius(new_pts)
                    if new_rad < best_rad:
                        best_rad = new_rad
                        best_points = new_pts.copy()
                        points = new_pts.copy()
                        
            except Exception:
                pass
    
    # Phase 3: High-precision greedy polish with adaptive step
    if time.time() - start_time < max_time - 5:
        points = best_points.copy()
        step_size = 0.002
        for _ in range(200):
            if time.time() - start_time > max_time - 2:
                break
            prev_points = points.copy()
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            max_facet_rad = 0.0
            worst_circumcenter = None
            worst_indices = None
            for idx, simplex in enumerate(hull.simplices):
                v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                n = np.cross(v1 - v2, v1 - v3)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                c = n / norm_n
                centroid = (v1 + v2 + v3) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
                rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
                if rad > max_facet_rad:
                    max_facet_rad = rad
                    worst_circumcenter = c
                    worst_indices = simplex
            if worst_indices is None:
                break
            for j in range(3):
                v_idx = worst_indices[j]
                v = points[v_idx]
                new_v = v + step_size * (worst_circumcenter - v)
                points[v_idx] = new_v / np.linalg.norm(new_v)
            current_rad = _covering_radius(points)
            if current_rad < best_rad:
                best_rad = current_rad
                best_points = points.copy()
            else:
                points = prev_points
                step_size *= 0.7
                if step_size < 1e-5:
                    break

    # Phase 4: Final ultra-fine polish
    if time.time() - start_time < max_time - 2:
        points = best_points.copy()
        step_size = 0.0005
        for _ in range(100):
            if time.time() - start_time > max_time - 1:
                break
            prev_points = points.copy()
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            max_facet_rad = 0.0
            worst_circumcenter = None
            worst_indices = None
            for idx, simplex in enumerate(hull.simplices):
                v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                n = np.cross(v1 - v2, v1 - v3)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                c = n / norm_n
                centroid = (v1 + v2 + v3) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
                rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
                if rad > max_facet_rad:
                    max_facet_rad = rad
                    worst_circumcenter = c
                    worst_indices = simplex
            if worst_indices is None:
                break
            for j in range(3):
                v_idx = worst_indices[j]
                v = points[v_idx]
                new_v = v + step_size * (worst_circumcenter - v)
                points[v_idx] = new_v / np.linalg.norm(new_v)
            current_rad = _covering_radius(points)
            if current_rad < best_rad:
                best_rad = current_rad
                best_points = points.copy()
            else:
                points = prev_points

    return best_points
