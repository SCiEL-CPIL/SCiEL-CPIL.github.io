"""Heilbronn convex regions n=14 — two concentric heptagons with analytical mixed-radius + 1D rotation."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar, minimize

N = 14

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area using pairwise cross products."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n - 2):
        a = points[i]
        for j in range(i + 1, n - 1):
            b = points[j]
            dk = points[j+1:n] - a
            db = b - a
            crosses = db[0] * dk[:, 1] - db[1] * dk[:, 0]
            areas = 0.5 * np.abs(crosses)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        n = len(points)
        if n < 3:
            return 0.0
        cx, cy = points[:, 0].mean(), points[:, 1].mean()
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        order = np.argsort(angles)
        hpts = points[order]
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += hpts[i][0] * hpts[j][1] - hpts[j][0] * hpts[i][1]
        return abs(area) * 0.5

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _two_heptagons(r, theta):
    n_out = 7
    n_in = 7
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _score_two_heptagons(r, theta):
    pts = _two_heptagons(r, theta)
    return _score(pts)

def _analytical_r_for_theta(theta):
    """
    For a given relative rotation theta between two concentric heptagons,
    find the inner radius r that maximizes the minimum area over all triangles.
    
    The key insight: with 7-fold symmetry, the critical triangles are:
    - Pure outer: 3 consecutive outer vertices
    - Pure inner: 3 consecutive inner vertices  
    - Mixed: 2 outer + 1 inner, or 1 outer + 2 inner
    
    For the mixed case, the minimum area occurs at specific angular separations.
    We can analytically determine r that balances pure and mixed triangle areas.
    
    For two concentric regular heptagons with relative rotation theta:
    - Pure outer triangle area (3 consecutive): A_out = (3/4) * sin(2*pi/7) * 1^2 / 2... 
      Actually, area of triangle formed by 3 consecutive vertices of unit heptagon:
      The central angle between adjacent vertices is 2*pi/7.
      Three consecutive vertices span 2*(2*pi/7) = 4*pi/7.
      Area = 0.5 * 1 * 1 * sin(2*pi/7) + 0.5 * 1 * 1 * sin(2*pi/7) + 0.5 * 1 * 1 * sin(4*pi/7)
      Wait, let me compute directly.
      
    Let me just use numerical optimization for r given theta, but with a good initial guess.
    """
    # The optimal r should balance the pure outer triangle area with mixed triangle areas.
    # Pure outer (3 consecutive on unit circle): 
    # vertices at angles 0, 2pi/7, 4pi/7
    # Area = 0.5 * |cross(v1-v0, v2-v0)|
    a0 = np.array([1.0, 0.0])
    a1 = np.array([np.cos(2*np.pi/7), np.sin(2*np.pi/7)])
    a2 = np.array([np.cos(4*np.pi/7), np.sin(4*np.pi/7)])
    A_pure_out = 0.5 * abs((a1[0]-a0[0])*(a2[1]-a0[1]) - (a1[1]-a0[1])*(a2[0]-a0[0]))
    
    # For mixed triangles, the minimum area depends on theta and r.
    # A good initial guess: r such that pure inner triangle area equals A_pure_out
    # Pure inner (3 consecutive on circle of radius r): A_pure_in = r^2 * A_pure_out
    # So r = 1.0 would make them equal, but that's not right because mixed triangles matter.
    
    # Better heuristic: the mixed triangle with 2 outer (adjacent) and 1 inner has area
    # that depends on the angular position of the inner point relative to the outer pair.
    # The worst case is when the inner point is closest to the line through the two outer points.
    
    # Let's do a quick 1D optimization for r
    def neg_score_r(r):
        if r <= 0.01 or r > 0.99:
            return 1.0
        return -_score_two_heptagons(r, theta)
    
    # Coarse scan to find good starting point
    r_scan = np.linspace(0.2, 0.9, 20)
    best_r = 0.5
    best_s = -1.0
    for rr in r_scan:
        s = _score_two_heptagons(rr, theta)
        if s > best_s:
            best_s = s
            best_r = rr
    
    # Fine optimization
    try:
        res = minimize_scalar(
            neg_score_r,
            bounds=(max(0.05, best_r - 0.1), min(0.98, best_r + 0.1)),
            method='bounded',
            options={'xatol': 1e-10}
        )
        if res.success:
            return res.x
    except Exception:
        pass
    
    return best_r

def _optimize_theta_full(time_limit=30.0):
    """
    Full 1D optimization over theta, where for each theta we analytically/numerically
    find the optimal r. This is a nested 1D problem: outer loop over theta, inner over r.
    """
    start_time = time.time()
    
    def neg_theta(theta):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 2:
            return 1.0
        r_opt = _analytical_r_for_theta(theta)
        return -_score_two_heptagons(r_opt, theta)
    
    # Coarse scan over theta in [0, pi/7] due to 7-fold symmetry
    n_scan = 30
    thetas = np.linspace(0, np.pi / 7, n_scan, endpoint=False)
    best_score = 0.0
    best_theta = 0.0
    best_r = 0.5
    
    for th in thetas:
        if time.time() > start_time + time_limit - 5:
            break
        r_opt = _analytical_r_for_theta(th)
        s = _score_two_heptagons(r_opt, th)
        if s > best_score:
            best_score = s
            best_theta = th
            best_r = r_opt
    
    # Fine local optimization
    if time.time() < start_time + time_limit - 5:
        try:
            res = minimize_scalar(
                neg_theta,
                bounds=(max(0, best_theta - np.pi / 28), min(np.pi / 7, best_theta + np.pi / 28)),
                method='bounded',
                options={'xatol': 1e-10}
            )
            if res.success and time.time() < start_time + time_limit - 2:
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_theta = res.x
                    best_r = _analytical_r_for_theta(best_theta)
        except Exception:
            pass
    
    return best_score, best_r, best_theta

def _local_optimize_free(points, time_limit=15.0):
    """Local optimization in full 2D space."""
    start_time = time.time()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if time.time() < start_time + time_limit:
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
    
    # One restart with small perturbation
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        perturb_scale = 0.03
        x_perturbed = best_pts.flatten() + perturb_scale * np.random.randn(len(best_pts.flatten()))
        try:
            res2 = minimize(
                obj, x_perturbed, method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if time.time() < start_time + time_limit:
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_pts = res2.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Nested 1D optimization (theta outer, r inner)
    s1, r_opt, theta_opt = _optimize_theta_full(time_limit=60.0)
    if s1 > best_score:
        best_score = s1
        best_points = _two_heptagons(r_opt, theta_opt).copy()
    
    # Strategy 2: Try a few additional theta values with their optimal r
    if time.time() < deadline - 40:
        additional_thetas = [
            np.pi / 14, 2 * np.pi / 14, 3 * np.pi / 14, 4 * np.pi / 14,
            np.pi / 28, 3 * np.pi / 28, 5 * np.pi / 28, 7 * np.pi / 28
        ]
        for th in additional_thetas:
            if time.time() > deadline - 35:
                break
            r_opt = _analytical_r_for_theta(th)
            s = _score_two_heptagons(r_opt, th)
            if s > best_score:
                best_score = s
                best_points = _two_heptagons(r_opt, th).copy()
    
    # Strategy 3: Regular 14-gon as baseline
    if time.time() < deadline - 30:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Strategy 4: Final local polish
    if best_points is not None and time.time() < deadline - 10:
        time_for_polish = min(20.0, deadline - 10 - time.time())
        s, pts_opt = _local_optimize_free(best_points, time_limit=time_for_polish)
        if s > best_score:
            best_score = s
            best_points = pts_opt.copy()
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    final_score = _score(best_points)
    if final_score <= 0:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
