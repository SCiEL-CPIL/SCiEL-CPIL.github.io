"""Optimized placement of 13 points to maximize minimum triangle area / convex hull area."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time

# Deadline guard: 120s limit, use 110s to be safe
_DEADLINE = 110.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _min_triangle_area(pts):
    """Compute the minimum triangle area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj - pi, pk - pi)|
                cross = (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) - \
                        (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _score_points(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-15:
            return 0.0
    except Exception:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _fourier_to_points(coeffs, n_harmonics=3):
    """
    Convert Fourier coefficients to 13 points on/near a smooth convex curve.
    
    coeffs layout: [r0, theta_offset, a1, b1, a2, b2, a3, b3]
    r(θ) = r0 + sum_{k=1}^{n_harmonics} (a_k cos(kθ) + b_k sin(kθ))
    
    Points are placed at equally spaced angles, with radius modulated by Fourier series.
    """
    n = N
    r0 = coeffs[0]
    theta_offset = coeffs[1]
    
    # Base angles
    base_theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + theta_offset
    
    # Fourier modulation
    r = np.full(n, r0)
    for k in range(1, n_harmonics + 1):
        a = coeffs[2 + 2 * (k - 1)]
        b = coeffs[2 + 2 * (k - 1) + 1]
        r += a * np.cos(k * base_theta) + b * np.sin(k * base_theta)
    
    # Ensure positive radius
    r = np.maximum(r, 0.1)
    
    pts = np.column_stack([r * np.cos(base_theta), r * np.sin(base_theta)])
    return pts


def _neg_score_fourier(coeffs):
    """Negative score for minimization."""
    if _remaining() < 5.0:
        return 1e6
    pts = _fourier_to_points(coeffs)
    score = _score_points(pts)
    return -score


def _neg_score_direct(pts_flat):
    """Negative score for direct point optimization."""
    if _remaining() < 5.0:
        return 1e6
    pts = pts_flat.reshape(-1, 2)
    score = _score_points(pts)
    return -score


def _try_regular_polygon():
    """Regular 13-gon as baseline."""
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    return pts


def _try_concentric_rings():
    """Two rings: 7 outer, 6 inner."""
    theta_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    theta_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    r_out, r_in = 1.0, 0.5
    pts_out = np.column_stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])
    pts_in = np.column_stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])
    return np.vstack([pts_out, pts_in])


def construct():
    global _start_time
    _start_time = time.time()
    
    best_score = -1.0
    best_pts = None
    
    # Initial candidates
    candidates = [
        _try_regular_polygon(),
        _try_concentric_rings(),
    ]
    
    # Add a few random convex-ish configurations
    rng = np.random.RandomState(N)
    for _ in range(3):
        theta = np.sort(rng.uniform(0, 2 * np.pi, N))
        r = rng.uniform(0.5, 1.0, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(pts)
    
    # Evaluate initial candidates
    for pts in candidates:
        if _remaining() < 10.0:
            break
        score = _score_points(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Fourier parameterization optimization
    if _remaining() > 30.0:
        n_harmonics = 2
        n_params = 2 + 2 * n_harmonics  # r0, theta_offset, a1, b1, a2, b2
        bounds = [(0.3, 1.5), (0, 2 * np.pi)] + [(-0.3, 0.3)] * (2 * n_harmonics)
        
        # Multi-start differential evolution on Fourier coefficients
        n_starts = 3
        for s in range(n_starts):
            if _remaining() < 20.0:
                break
            x0 = rng.uniform(
                [b[0] for b in bounds],
                [b[1] for b in bounds]
            )
            try:
                result = differential_evolution(
                    _neg_score_fourier,
                    bounds=bounds,
                    x0=x0,
                    maxiter=30,
                    popsize=10,
                    tol=1e-6,
                    seed=N + s,
                    polish=False,
                )
                if result.fun < 1e5:
                    pts = _fourier_to_points(result.x)
                    score = _score_points(pts)
                    if score > best_score:
                        best_score = score
                        best_pts = pts.copy()
            except Exception:
                pass
    
    # Direct point optimization from best candidate
    if best_pts is not None and _remaining() > 40.0:
        pts_flat = best_pts.flatten()
        bounds = [(-2, 2)] * (2 * N)
        
        n_starts = 2
        for s in range(n_starts):
            if _remaining() < 30.0:
                break
            perturbation = rng.normal(0, 0.1, size=2 * N)
            x0 = pts_flat + perturbation
            try:
                result = differential_evolution(
                    _neg_score_direct,
                    bounds=bounds,
                    x0=x0,
                    maxiter=20,
                    popsize=8,
                    tol=1e-6,
                    seed=N * 100 + s,
                    polish=False,
                )
                if result.fun < 1e5:
                    pts = result.x.reshape(-1, 2)
                    score = _score_points(pts)
                    if score > best_score:
                        best_score = score
                        best_pts = pts.copy()
            except Exception:
                pass
    
    # Final polish with Nelder-Mead on the best configuration
    if best_pts is not None and _remaining() > 15.0:
        from scipy.optimize import minimize
        try:
            result = minimize(
                _neg_score_direct,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12},
            )
            if result.fun < 1e5:
                pts = result.x.reshape(-1, 2)
                score = _score_points(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # Ensure we always return a valid point set
    if best_pts is None or len(best_pts) != N:
        best_pts = _try_regular_polygon()
    
    return best_pts
# EVOLVE-BLOCK-END
