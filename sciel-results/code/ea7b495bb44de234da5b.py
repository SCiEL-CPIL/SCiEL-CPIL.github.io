"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

N = 14
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _min_triangle_area_ratio(pts):
    """Compute minimum triangle area divided by convex hull area."""
    n = len(pts)
    # Compute all triangle areas vectorized
    # Area of triangle (i,j,k) = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    
    # Convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area < 1e-12:
        return 0.0
    
    return min_area / hull_area


def _score_negative(pts_flat):
    """Negative score for minimization. Points are flattened (n*2,)."""
    pts = pts_flat.reshape(N, 2)
    return -_min_triangle_area_ratio(pts)


def _regular_polygon():
    """Regular 14-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _fourier_convex_curve(theta, coeffs):
    """
    Evaluate a strictly convex radial function r(theta) using Fourier series.
    coeffs: [r0, a1, b1, a2, b2, ..., aM, bM]
    r(theta) = r0 + sum_{k=1}^{M} (a_k * cos(k*theta) + b_k * sin(k*theta))
    We ensure r > 0 by clamping.
    """
    M = (len(coeffs) - 1) // 2
    r = np.full_like(theta, coeffs[0], dtype=float)
    for k in range(1, M + 1):
        r += coeffs[2 * k - 1] * np.cos(k * theta)
        r += coeffs[2 * k] * np.sin(k * theta)
    # Ensure positivity
    r = np.maximum(r, 0.1)
    return r


def _optimize_with_fourier():
    """
    Optimize angular positions of 14 points on a Fourier-defined convex curve.
    Parameters: angles (14,), Fourier coeffs (r0, a1, b1, a2, b2, a3, b3)
    Total params: 14 + 7 = 21
    """
    n_angles = N
    n_coeffs = 7  # r0, a1, b1, a2, b2, a3, b3
    total = n_angles + n_coeffs
    
    def objective(params):
        angles = params[:n_angles]
        coeffs = params[n_angles:]
        
        # Ensure angles are sorted and in [0, 2*pi)
        angles = np.sort(angles) % (2 * np.pi)
        
        # Compute points on the curve
        r = _fourier_convex_curve(angles, coeffs)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        
        return _score_negative(pts.flatten())
    
    # Bounds: angles in [0, 2*pi), coeffs in [-0.5, 2.0] for r0, [-0.3, 0.3] for harmonics
    bounds = [(0, 2 * np.pi)] * n_angles
    bounds.append((0.5, 2.0))       # r0
    bounds.extend([(-0.3, 0.3)] * 6) # a1,b1,a2,b2,a3,b3
    
    # Initial guess: regular polygon angles, r0=1, small harmonics
    x0 = np.zeros(total)
    x0[:n_angles] = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    x0[n_angles] = 1.0  # r0
    
    result = differential_evolution(
        objective, bounds,
        seed=42,
        maxiter=300,
        popsize=15,
        tol=1e-8,
        init='sobol',
        polish=True
    )
    
    best_params = result.x
    angles = np.sort(best_params[:n_angles]) % (2 * np.pi)
    coeffs = best_params[n_angles:]
    r = _fourier_convex_curve(angles, coeffs)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    return pts, -result.fun


def _local_polish(pts, max_iter=200):
    """Local refinement using gradient-free optimization on point coordinates."""
    from scipy.optimize import minimize
    
    def neg_score(pts_flat):
        return _score_negative(pts_flat)
    
    best_pts = pts.copy()
    best_score = -neg_score(pts.flatten())
    
    for _ in range(3):
        res = minimize(
            neg_score,
            best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if res.fun < neg_score(best_pts.flatten()):
            best_pts = res.x.reshape(N, 2)
            best_score = -res.fun
        else:
            break
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    
    # Start with regular polygon
    pts = _regular_polygon()
    best_pts = pts.copy()
    best_score = _min_triangle_area_ratio(pts)
    
    # Phase 1: Global search with Fourier parameterization
    remaining = TIME_LIMIT - (time.time() - start_time) - 20.0  # leave 20s for polish
    if remaining > 10:
        try:
            fourier_pts, fourier_score = _optimize_with_fourier()
            if fourier_score > best_score:
                best_pts = fourier_pts
                best_score = fourier_score
        except Exception:
            pass
    
    # Phase 2: Local polish
    remaining = TIME_LIMIT - (time.time() - start_time) - 5.0
    if remaining > 5:
        try:
            polished_pts, polished_score = _local_polish(best_pts, max_iter=int(remaining * 5))
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
    
    # Final check
    if time.time() - start_time > TIME_LIMIT:
        return best_pts
    
    return best_pts
