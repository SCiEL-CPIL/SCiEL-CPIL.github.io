import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
            if area_hull < 1e-9:
                return 0.0
            
            x = pts[:, 0]
            y = pts[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            min_area = np.min(areas)
            return min_area / area_hull
        except Exception:
            return 0.0

    def _soft_objective(points_flat, beta=100.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = pts - pts.mean(axis=0)
        
        try:
            hull = ConvexHull(pts)
            area_hull = hull.volume
            if area_hull < 1e-9:
                return 1e6
            pts = pts * np.sqrt(1.0 / area_hull)
        except Exception:
            return 1e6

        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def _exact_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), angle_offset=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _fourier_to_points(coeffs):
        """Convert Fourier coefficients to 13 point positions."""
        n_harm = len(coeffs) // 2
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        for m in range(1, n_harm + 1):
            a_m = coeffs[2 * (m - 1)]
            b_m = coeffs[2 * (m - 1) + 1]
            radii += a_m * np.cos(m * angles) + b_m * np.sin(m * angles)
        radii = np.maximum(radii, 0.1)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])
    
    def _fourier_objective(coeffs, beta=100.0):
        """Objective for Fourier parameterization."""
        pts = _fourier_to_points(coeffs)
        return _soft_objective(pts.flatten(), beta)

    # Generate initializations
    initializations = []
    initializations.append(_regular_polygon(13, 1.0))
    
    rng = np.random.RandomState(42)
    for _ in range(20):
        pts = _regular_polygon(13, 1.0)
        pts += rng.randn(*pts.shape) * 0.1
        initializations.append(pts)
        
    fourier_bounds = [(-0.15, 0.15)] * 8
    rng_fourier = np.random.RandomState(123)
    for _ in range(20):
        coeffs = rng_fourier.uniform(-0.1, 0.1, size=8)
        pts = _fourier_to_points(coeffs)
        initializations.append(pts)

    best_score = -float('inf')
    best_points = None

    # Phase 1: Global Search in Fourier Space
    if time.time() < start_time + time_limit - 60:
        try:
            de_result = opt.differential_evolution(
                _fourier_objective,
                bounds=fourier_bounds,
                seed=42,
                maxiter=100,
                popsize=25,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < 1e5:
                best_coeffs = de_result.x
                de_points = _fourier_to_points(best_coeffs)
                de_score = compute_score(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
                initializations.append(de_points)
        except Exception:
            pass

    # Phase 2: Local Refinement
    for idx, init in enumerate(initializations):
        if time.time() > start_time + time_limit - 10:
            break
            
        current = init.copy()
        current_score = compute_score(current)
        
        # Step 2a: Refine in Fourier Space using L-BFGS-B
        if time.time() < start_time + time_limit - 5:
            try:
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                radii = np.sqrt(current[:, 0]**2 + current[:, 1]**2)
                A = np.zeros((N, 8))
                for m in range(1, 5):
                    A[:, 2*(m-1)] = np.cos(m * angles)
                    A[:, 2*(m-1)+1] = np.sin(m * angles)
                coeffs_fit, _, _, _ = np.linalg.lstsq(A, radii - 1.0, rcond=None)
                
                res = opt.minimize(
                    lambda c: _fourier_objective(c, beta=200.0),
                    coeffs_fit,
                    method='L-BFGS-B',
                    bounds=fourier_bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < 1e5:
                    f_pts = _fourier_to_points(res.x)
                    f_score = compute_score(f_pts)
                    if f_score > current_score:
                        current = f_pts
                        current_score = f_score
            except Exception:
                pass

        # Step 2b: Refine in Cartesian Space using Nelder-Mead
        if time.time() < start_time + time_limit - 2:
            try:
                res = opt.minimize(
                    _exact_objective,
                    current.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                candidate = res.x.reshape(-1, 2)
                score = compute_score(candidate)
                
                if score > current_score:
                    current = candidate
                    current_score = score
            except Exception:
                pass

        if current_score > best_score:
            best_score = current_score
            best_points = current.copy()

    if best_points is None:
        best_points = initializations[0]
        best_score = compute_score(best_points)

    return best_points
