"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s

def _normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return v / n

def _compute_covering_radius(pts):
    """Compute the covering radius in degrees using convex hull facets."""
    try:
        hull = ConvexHull(pts)
        max_rad = 0.0
        for simplex in hull.simplices:
            tri = pts[simplex]
            # Compute circumradius of triangle on unit sphere
            # For a spherical triangle, the circumradius can be computed
            a, b, c = tri
            # Side lengths (angular distances)
            ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            # Circumradius formula for spherical triangle
            s = (ab + bc + ca) / 2.0
            # Use the formula: sin(R) = sin(ab)*sin(bc)*sin(ca) / (4*sin(s-a)*sin(s-b)*sin(s-c))
            # But this can be unstable. Alternative: compute circumcenter directly.
            # Circumcenter of triangle on sphere: solve for point equidistant from a, b, c
            # |x-a|^2 = |x-b|^2 => 2*x.(b-a) = |b|^2 - |a|^2 = 0 (since unit sphere)
            # So x is in plane perpendicular to (b-a) through origin, similarly for (c-a)
            # x = alpha*(b-a) x (c-a) direction? No.
            # The circumcenter lies in the plane containing a, b, c.
            # Normal to plane: n = (b-a) x (c-a)
            # Circumcenter = k * n where k is chosen so that |k*n - a|^2 = 1
            # |k*n|^2 - 2k*n.a + |a|^2 = 1 => k^2*n.n - 2k*n.a = 0
            # k = 2*n.a / (n.n)
            n = np.cross(b - a, c - a)
            nn = np.dot(n, n)
            if nn < 1e-15:
                continue
            k = 2.0 * np.dot(n, a) / nn
            x = k * n
            x_norm = np.linalg.norm(x)
            if x_norm < 1e-12:
                continue
            x = x / x_norm
            # Distance from x to a
            d = np.arccos(np.clip(np.dot(x, a), -1, 1))
            max_rad = max(max_rad, d)
        return np.degrees(max_rad)
    except Exception:
        return 90.0  # fallback: worst case

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _smooth_covering_surrogate(pts, beta=50.0):
    """
    Smooth surrogate: for each point on sphere, distance to nearest config point.
    We approximate by checking distances to all config points and using soft-max.
    But this is expensive. Instead, use the convex hull approach with smoothing.
    
    Alternative: minimize the max of circumradii using a soft-max.
    """
    try:
        hull = ConvexHull(pts)
        radii = []
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri
            n = np.cross(b - a, c - a)
            nn = np.dot(n, n)
            if nn < 1e-15:
                continue
            k = 2.0 * np.dot(n, a) / nn
            x = k * n
            x_norm = np.linalg.norm(x)
            if x_norm < 1e-12:
                continue
            x = x / x_norm
            d = np.arccos(np.clip(np.dot(x, a), -1, 1))
            radii.append(d)
        if not radii:
            return 90.0
        radii = np.array(radii)
        # Soft-max approximation
        m = np.max(radii)
        if m < 1e-12:
            return 0.0
        shifted = radii - m
        log_sum = m + np.log(np.sum(np.exp(beta * shifted))) / beta
        return np.degrees(log_sum)
    except Exception:
        return 90.0

def _objective(pts_flat):
    """Objective function for optimizer."""
    pts = pts_flat.reshape(-1, 3)
    pts = _normalize(pts)
    return _smooth_covering_surrogate(pts, beta=30.0)

def _gradient(pts_flat):
    """Numerical gradient."""
    pts = pts_flat.reshape(-1, 3)
    pts = _normalize(pts)
    n = len(pts)
    grad = np.zeros_like(pts_flat)
    h = 1e-5
    for i in range(n):
        for j in range(3):
            pts_plus = pts.copy()
            pts_plus[i, j] += h
            pts_plus = _normalize(pts_plus)
            pts_minus = pts.copy()
            pts_minus[i, j] -= h
            pts_minus = _normalize(pts_minus)
            f_plus = _smooth_covering_surrogate(pts_plus, beta=30.0)
            f_minus = _smooth_covering_surrogate(pts_minus, beta=30.0)
            grad[i * 3 + j] = (f_plus - f_minus) / (2 * h)
    return grad

def _optimize_from_initial(pts0, max_iter=500, tol=1e-8):
    """Optimize starting from pts0."""
    pts = _normalize(pts0.copy())
    best_pts = pts.copy()
    best_val = _compute_covering_radius(pts)
    
    # Use L-BFGS-B with numerical gradient
    x0 = pts.flatten()
    
    result = minimize(
        _objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxfun': 5000}
    )
    
    if result.success:
        pts_opt = _normalize(result.x.reshape(-1, 3))
        val = _compute_covering_radius(pts_opt)
        if val < best_val:
            best_pts = pts_opt
            best_val = val
    
    return best_pts, best_val

def _anneal_optimize(pts0, n_restarts=5, max_iter=200):
    """Annealing: perturb and re-optimize."""
    best_pts, best_val = _optimize_from_initial(pts0, max_iter=max_iter)
    
    for r in range(n_restarts):
        if time.time() > _DEADLINE:
            break
        # Perturb
        rng = np.random.RandomState(42 + r)
        noise = rng.randn(N, 3) * 0.1
        pts_perturbed = _normalize(pts0 + noise)
        pts_new, val_new = _optimize_from_initial(pts_perturbed, max_iter=max_iter)
        if val_new < best_val:
            best_pts = pts_new
            best_val = val_new
        pts0 = best_pts
    
    return best_pts, best_val

def construct():
    global _start_time
    t0 = time.time()
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Fibonacci spiral
    initial_configs.append(_fibonacci_sphere(N))
    
    # Random configurations
    for seed in [0, 1, 2, 3, 4]:
        initial_configs.append(_random_sphere(N, seed))
    
    best_pts = None
    best_val = 180.0
    
    for i, init in enumerate(initial_configs):
        if time.time() - t0 > _DEADLINE:
            break
        
        if i == 0:
            # First config: more thorough optimization
            pts, val = _anneal_optimize(init, n_restarts=3, max_iter=300)
        else:
            # Subsequent: lighter optimization
            pts, val = _optimize_from_initial(init, max_iter=200)
        
        if val < best_val:
            best_val = val
            best_pts = pts
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Final polish: high-precision optimization
    if time.time() - t0 < _DEADLINE - 10:
        pts_final, val_final = _optimize_from_initial(best_pts, max_iter=500, tol=1e-10)
        if val_final < best_val:
            best_pts = pts_final
            best_val = val_final
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
