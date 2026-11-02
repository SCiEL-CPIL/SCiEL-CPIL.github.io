"""
Combined approach: Fourier-based smooth convex boundary + direct radial refinement.
1. Use a low-frequency Fourier representation of the radial function to get a smooth
   convex boundary (reduces effective dimensionality and enforces smoothness).
2. Place 14 points at optimized angular positions along this boundary.
3. Refine with multi-phase optimization: Powell on Fourier coefficients, then
   Nelder-Mead on full 2D coordinates, then coordinate descent.
4. Also try direct radial optimization as an alternative starting point.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
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
    Calculate the score: min_triangle_area / hull_area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _fourier_curve(coeffs, n_pts=512):
    """
    Generate a smooth closed curve from Fourier coefficients.
    coeffs: array of length 2*K+1 (DC + K harmonics, real and imaginary).
    Returns (n_pts, 2) array of points on the curve.
    """
    K = (len(coeffs) - 1) // 2
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    x = np.zeros(n_pts)
    y = np.zeros(n_pts)
    x += coeffs[0]
    y += coeffs[1] if len(coeffs) > 1 else 0.0
    
    for k in range(1, K + 1):
        a = coeffs[2 * k]
        b = coeffs[2 * k + 1]
        x += a * np.cos(k * t)
        y += b * np.sin(k * t)
    
    return np.column_stack([x, y])

def _fourier_to_points(coeffs, n_pts=N):
    """
    Convert Fourier coefficients to N points by sampling the curve at n_pts angles.
    """
    curve = _fourier_curve(coeffs, n_pts=n_pts)
    return curve

def _decode_fourier_params(params):
    """
    Decode optimization parameters into N points.
    params: [a0, b0, a1, b1, ..., aK, bK, theta_0, theta_1, ..., theta_{N-1}]
    Where a0, b0 are DC, a_k, b_k are harmonic coefficients, theta_i are angular positions.
    """
    K = 2  # Use 2 harmonics (5 coefficients total: a0, b0, a1, b1, a2, b2)
    n_coeffs = 2 * K + 2  # a0, b0, a1, b1, a2, b2
    coeffs = params[:n_coeffs]
    thetas = params[n_coeffs:n_coeffs + N]
    
    # Ensure convexity: DC term should be large enough
    # r(θ) = a0 + b0 + a1*cos(θ) + b1*sin(θ) + a2*cos(2θ) + b2*sin(2θ)
    # For convexity, we need r(θ) > 0 and the curve to be star-shaped
    a0 = coeffs[0]
    b0 = coeffs[1]
    # Make sure the base radius is positive
    base_r = max(a0, 0.5)
    
    # Reconstruct with guaranteed positive base
    coeffs_adj = coeffs.copy()
    coeffs_adj[0] = base_r
    coeffs_adj[1] = b0
    
    # Sample at the given angles
    t = thetas
    x = np.zeros(N)
    y = np.zeros(N)
    x += coeffs_adj[0]
    y += coeffs_adj[1]
    
    for k in range(1, K + 1):
        a = coeffs_adj[2 * k]
        b = coeffs_adj[2 * k + 1]
        x += a * np.cos(k * t)
        y += b * np.sin(k * t)
    
    return np.column_stack([x, y])

def _objective_fourier(params, deadline=None):
    """Objective for Fourier-based optimization."""
    if deadline is not None and time.time() > deadline:
        return 1e10
    pts = _decode_fourier_params(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    s = _score(pts)
    if s <= 0:
        return 1e10
    return -s

def _radii_to_points_fixed_angles(r, angles):
    """Convert radii to points with fixed angles."""
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.column_stack([x, y])

def _score_from_radii(r, angles):
    pts = _radii_to_points_fixed_angles(r, angles)
    return _score(pts)

def _neg_score_from_radii(r, angles):
    r = np.maximum(r, 0.1)
    return -_score_from_radii(r, angles)

def _normalize_unit_area(points):
    """Normalize points to unit convex hull area."""
    try:
        hull = ConvexHull(points)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            return points * scale
    except Exception:
        pass
    # Fallback: normalize by bounding box
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
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    best_score_val = 0.0
    best_pts = None
    
    def _consider(pts):
        nonlocal best_score_val, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score_val:
            best_score_val = s
            best_pts = pts.copy()
    
    # ---- Strategy 1: Fourier-based optimization ----
    if time.time() < deadline - 50:
        # Initial Fourier coefficients: circle + small harmonics
        # a0=1, b0=0, a1=0.1, b1=0.05, a2=0.05, b2=0.03
        # Angles: regular polygon
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
        initial_params = np.array([1.0, 0.0, 0.1, 0.05, 0.05, 0.03])
        initial_params = np.concatenate([initial_params, base_angles])
        
        # Phase 1a: Powell on Fourier params
        if time.time() < deadline - 40:
            try:
                res1 = minimize(_objective_fourier, initial_params, method='Powell',
                                options={'maxiter': 1500, 'ftol': 1e-12, 'xtol': 1e-12})
                pts1 = _decode_fourier_params(res1.x)
                _consider(pts1)
            except Exception:
                pass
        
        # Phase 1b: Nelder-Mead refinement on Fourier params
        if time.time() < deadline - 25:
            try:
                res2 = minimize(_objective_fourier, res1.x, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                pts2 = _decode_fourier_params(res2.x)
                _consider(pts2)
            except Exception:
                pass
    
    # ---- Strategy 2: Direct radial optimization with fixed angles ----
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    if time.time() < deadline - 30:
        # Initial guess: Regular 14-gon (all radii = 1)
        r_init = np.ones(N)
        
        # Phase 2a: Powell optimization on radii
        if time.time() < deadline - 25:
            try:
                res3 = minimize(lambda r: _neg_score_from_radii(r, angles), r_init, method='Powell',
                                options={'maxiter': 1500, 'ftol': 1e-12, 'xtol': 1e-12})
                r_candidate = np.maximum(res3.x, 0.1)
                pts3 = _radii_to_points_fixed_angles(r_candidate, angles)
                _consider(pts3)
            except Exception:
                pass
        
        # Phase 2b: Nelder-Mead refinement on radii
        if time.time() < deadline - 15:
            try:
                res4 = minimize(lambda r: _neg_score_from_radii(r, angles), res3.x, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                r_candidate = np.maximum(res4.x, 0.1)
                pts4 = _radii_to_points_fixed_angles(r_candidate, angles)
                _consider(pts4)
            except Exception:
                pass
    
    # ---- Strategy 3: Full 2D coordinate optimization from best so far ----
    if best_pts is not None and time.time() < deadline - 15:
        # Start from the best points found so far
        pts_flat = best_pts.flatten().copy()
        
        def _obj_2d(flat):
            if time.time() > deadline - 2:
                return 1e10
            pts = flat.reshape(N, 2)
            if np.any(~np.isfinite(pts)):
                return 1e10
            s = _score(pts)
            if s <= 0:
                return 1e10
            return -s
        
        # Phase 3a: Nelder-Mead on full 2D coordinates
        if time.time() < deadline - 10:
            try:
                res5 = minimize(_obj_2d, pts_flat, method='Nelder-Mead',
                                options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                pts5 = res5.x.reshape(N, 2)
                _consider(pts5)
            except Exception:
                pass
        
        # Phase 3b: Powell on full 2D coordinates
        if time.time() < deadline - 5:
            try:
                res6 = minimize(_obj_2d, res5.x, method='Powell',
                                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12})
                pts6 = res6.x.reshape(N, 2)
                _consider(pts6)
            except Exception:
                pass
    
    # ---- Final coordinate descent polish ----
    if best_pts is not None and time.time() < deadline - 3:
        pts_cd = best_pts.copy()
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                for dim in range(2):
                    def obj_1d(val):
                        pts_tmp = pts_cd.copy()
                        pts_tmp[i, dim] = val
                        s = _score(pts_tmp)
                        return -s if s > 0 else 1e10
                    
                    try:
                        res_cd = minimize(obj_1d, pts_cd[i, dim], method='Brent',
                                          bounds=(-5.0, 5.0), options={'xtol': 1e-12})
                        if res_cd.fun < obj_1d(pts_cd[i, dim]):
                            pts_cd[i, dim] = res_cd.x
                            improved = True
                    except Exception:
                        pass
            
            _consider(pts_cd)
            if not improved:
                break
    
    # Normalize to unit area
    if best_pts is not None:
        best_pts = _normalize_unit_area(best_pts)
    
    # Fallback: regular 14-gon
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_pts = _normalize_unit_area(best_pts)
    
    return best_pts
