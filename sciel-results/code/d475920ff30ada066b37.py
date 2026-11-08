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

    def _get_hull_area(pts):
        """Get convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            area_hull = _get_hull_area(pts)
        except Exception:
            return 0.0
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

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _soft_objective(points_flat, beta=100.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min: -log(sum(exp(-beta * areas))) / beta
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
        """Convert Fourier coefficients to 13 point positions.
        coeffs = [a1, b1, a2, b2, a3, b3, a4, b4] (8 params)
        r_k = 1 + sum_{m=1}^{4} (a_m * cos(m*θ_k) + b_m * sin(m*θ_k))
        """
        n = len(coeffs) // 2  # number of harmonics
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        for m in range(1, n + 1):
            a_m = coeffs[2 * (m - 1)]
            b_m = coeffs[2 * (m - 1) + 1]
            radii += a_m * np.cos(m * angles) + b_m * np.sin(m * angles)
        # Ensure positive radii (convexity)
        radii = np.maximum(radii, 0.1)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])
    
    def _fourier_objective(coeffs, beta=100.0):
        """Objective for Fourier parameterization (minimize negative soft-min area)."""
        pts = _fourier_to_points(coeffs)
        return _soft_objective(pts.flatten(), beta)

    # Generate initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 on boundary + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. Various split configurations
    for k_outer in range(5, 13):
        k_inner = N - k_outer
        outer = _regular_polygon(k_outer, 1.0)
        for r_inner in [0.3, 0.5, 0.7]:
            if k_inner == 1:
                inner = np.array([[0.0, 0.0]])
            else:
                inner = _regular_polygon(k_inner, r_inner)
            config = np.vstack([outer, inner])
            initializations.append(config)
            
    # 4. Random perturbations
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 3):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.05
        initializations.append(perturbed)
        
    # Fourier-based initializations
    rng_fourier = np.random.RandomState(123)
    for _ in range(10):
        coeffs = rng_fourier.uniform(-0.1, 0.1, size=8)
        pts = _fourier_to_points(coeffs)
        initializations.append(pts)

    best_score = -float('inf')
    best_points = None

    # Phase 1: Global Search on Fourier Space
    # This helps find a good convex topology
    if time.time() < start_time + time_limit - 60:
        try:
            fourier_bounds = [(-0.15, 0.15)] * 8
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
                de_points = _fourier_to_points(de_result.x)
                de_score = compute_score(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
                initializations.append(de_points)
        except Exception:
            pass

    # Phase 2: Local Refinement from Initializations
    for idx, init in enumerate(initializations):
        if time.time() > start_time + time_limit - 25:
            break
            
        current = init.copy()
        current_score = compute_score(current)
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        betas = [50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Final polish with Nelder-Mead on exact objective
        if time.time() > start_time + time_limit - 10:
            break
        try:
            res = opt.minimize(
                _exact_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass

    # Final check
    if best_points is None:
        best_points = initializations[0]
    
    return best_points
