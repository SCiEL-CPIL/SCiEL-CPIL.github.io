import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by triples of points.
    """
    n = len(points)
    if n < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score is the minimum triangle area divided by the convex hull area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _decode_fourier(params):
    """
    Decode Fourier parameters into point coordinates.
    params: [R0, A1, B1, A2, B2, ..., AK, BK]
    The curve is r(theta) = R0 + sum(Ak*cos(k*theta) + Bk*sin(k*theta))
    Points are placed at uniform angles theta_i = 2*pi*i/N.
    """
    R0 = params[0]
    # Ensure R0 is sufficiently large to maintain convexity
    # We constrain the amplitudes in the optimizer bounds, but clamp here for safety
    R0 = max(R0, 1.0)
    
    n_harmonics = (len(params) - 1) // 2
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    r = np.full(N, R0)
    for k in range(1, n_harmonics + 1):
        Ak = params[1 + 2*(k-1)]
        Bk = params[2 + 2*(k-1)]
        r += Ak * np.cos(k * angles) + Bk * np.sin(k * angles)
        
    # Ensure r is positive
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    return np.column_stack([x, y])

def _objective_fourier(params):
    """
    Objective function for Fourier parameter optimization.
    We want to maximize min_area / hull_area.
    So we minimize hull_area / min_area.
    """
    pts = _decode_fourier(params)
    
    # Check for degeneracy
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
        
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
        
    return hull_area / min_area

def _normalize_unit_area(points):
    """
    Scale points so that the convex hull area is 1.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    # Fallback: scale by bounding box
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    
    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Define search space for Fourier coefficients
    # R0: base radius, [0.5, 2.0]
    # Amplitudes: small to ensure convexity. 
    # For strict convexity, |r''| < r is a sufficient condition (roughly).
    # r(theta) = R0 + A1 cos theta + ...
    # r''(theta) = -A1 cos theta - 4 A2 cos 2theta ...
    # Max |r''| ~ sum(k^2 * A_k). 
    # If R0=1, and we want r'' < r, we should keep sum(k^2 * A_k) < 1.
    # Let's be conservative: A1, B1 in [-0.3, 0.3], A2, B2 in [-0.1, 0.1], etc.
    
    n_harmonics = 4
    bounds = [(0.5, 2.0)]  # R0
    for k in range(1, n_harmonics + 1):
        # Decay amplitude bounds with frequency
        amp_bound = 0.5 / (k ** 1.5)
        bounds.append((-amp_bound, amp_bound))  # Ak
        bounds.append((-amp_bound, amp_bound))  # Bk
        
    x0 = np.array([1.0] + [0.0] * (2 * n_harmonics))
    
    # Phase 1: Differential Evolution for global search
    # DE is robust but can be slow. We limit iterations.
    try:
        res_de = differential_evolution(
            _objective_fourier, 
            bounds, 
            seed=42, 
            maxiter=50, 
            tol=1e-8,
            popsize=20, 
            mutation=(0.3, 1.0), 
            recombination=0.7,
            polish=False,
            update_interval=1
        )
        if res_de is not None and np.isfinite(res_de.fun) and res_de.fun < 1e9:
            pts_de = _decode_fourier(res_de.x)
            _consider(pts_de)
            # Refine DE result with local optimizer
            if time.time() < deadline - 30:
                res_nm = minimize(
                    _objective_fourier, 
                    res_de.x, 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res_nm is not None and np.isfinite(res_nm.fun) and res_nm.fun < 1e9:
                    pts_nm = _decode_fourier(res_nm.x)
                    _consider(pts_nm)
    except Exception:
        pass
    
    # Phase 2: Local search from regular polygon (all harmonics 0)
    # This is a good starting point.
    if time.time() < deadline - 20:
        x0_reg = np.array([1.0] + [0.0] * (2 * n_harmonics))
        try:
            res_reg = minimize(
                _objective_fourier, 
                x0_reg, 
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-11, 'ftol': 1e-13}
            )
            if res_reg is not None and np.isfinite(res_reg.fun) and res_reg.fun < 1e9:
                pts_reg = _decode_fourier(res_reg.x)
                _consider(pts_reg)
        except Exception:
            pass

    # Phase 3: Try a few random Fourier configs and refine
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        x0_rand = np.array([1.0])
        for k in range(1, n_harmonics + 1):
            amp_bound = 0.5 / (k ** 1.5)
            x0_rand = np.append(x0_rand, rng.uniform(-amp_bound, amp_bound, 2))
        
        try:
            res_rand = minimize(
                _objective_fourier, 
                x0_rand, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_rand is not None and np.isfinite(res_rand.fun) and res_rand.fun < 1e9:
                pts_rand = _decode_fourier(res_rand.x)
                _consider(pts_rand)
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    final_points = _normalize_unit_area(best_pts)
    
    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    return final_points

# EVOLVE-BLOCK-END
