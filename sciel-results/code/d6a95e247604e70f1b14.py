import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 148.0  # Safety margin for 150s limit

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
        r_k = 1 + sum_{m=1}^{4} (a_m * cos(m*theta_k) + b_m * sin(m*theta_k))
        """
        n_harmonics = len(coeffs) // 2
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        for m in range(1, n_harmonics + 1):
            a_m = coeffs[2 * (m - 1)]
            b_m = coeffs[2 * (m - 1) + 1]
            radii += a_m * np.cos(m * angles) + b_m * np.sin(m * angles)
        # Ensure positive radii (convexity)
        radii = np.maximum(radii, 0.1)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])

    def _fourier_soft_objective(coeffs, beta=200.0):
        """
        Objective in Fourier space: minimize negative soft-min triangle area.
        Points are on a convex curve r(theta) = 1 + sum harmonics.
        """
        pts = _fourier_to_points(coeffs)
        # Normalize to unit hull area
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
        
        # Soft min via log-sum-exp
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def _fourier_exact_objective(coeffs):
        """Exact objective in Fourier space."""
        pts = _fourier_to_points(coeffs)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Fourier-based search: 8 parameters (4 harmonics)
    # This guarantees convexity and reduces the search space dramatically
    
    fourier_bounds = [(-0.2, 0.2)] * 8
    best_score = -float('inf')
    best_points = _regular_polygon(13, 1.0).copy()
    best_coeffs = np.zeros(8)
    
    # Phase 1: Differential evolution in Fourier space (global search)
    if time.time() < start_time + time_limit - 40:
        try:
            de_result = opt.differential_evolution(
                _fourier_soft_objective,
                bounds=fourier_bounds,
                seed=42,
                maxiter=80,
                popsize=20,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='latinhypercube',
                updateinterval=0,
                callback=None
            )
            if de_result.fun < 1e5:
                best_coeffs = de_result.x.copy()
                de_points = _fourier_to_points(best_coeffs)
                de_score = compute_score(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B in Fourier space with annealed beta
    rng = np.random.RandomState(42)
    n_starts = 12
    
    for s in range(n_starts):
        if time.time() > start_time + time_limit - 20:
            break
        
        # Generate starting point: mix of DE result and random
        if s == 0:
            coeffs_init = best_coeffs.copy()
        elif s == 1:
            coeffs_init = np.zeros(8)  # Regular polygon
        else:
            coeffs_init = rng.uniform(-0.15, 0.15, size=8)
        
        current_coeffs = coeffs_init.copy()
        
        # Anneal beta from low to high for smoother optimization
        betas = [30.0, 80.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                res = opt.minimize(
                    lambda c, b=beta: _fourier_soft_objective(c, b),
                    current_coeffs,
                    method='L-BFGS-B',
                    bounds=fourier_bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < 1e5:
                    current_coeffs = res.x.copy()
            except Exception:
                pass
        
        # Evaluate and update best
        candidate_points = _fourier_to_points(current_coeffs)
        candidate_score = compute_score(candidate_points)
        
        if candidate_score > best_score:
            best_score = candidate_score
            best_points = candidate_points.copy()
            best_coeffs = current_coeffs.copy()
    
    # Phase 3: High-precision polish with exact objective in Fourier space
    if time.time() < start_time + time_limit - 10:
        try:
            res = opt.minimize(
                _fourier_exact_objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < 1e5:
                nm_points = _fourier_to_points(res.x)
                nm_score = compute_score(nm_points)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
                    best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final coordinate descent in Cartesian space for fine-tuning
    if time.time() < start_time + time_limit - 5:
        pts = best_points.copy()
        base_steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        
        for iteration in range(30):
            if time.time() > start_time + time_limit - 2:
                break
            improved = False
            for i in range(N):
                best_local = pts[i].copy()
                best_local_score = compute_score(pts)
                
                for step in base_steps:
                    directions = [
                        (step, 0), (-step, 0), (0, step), (0, -step),
                        (step*0.707, step*0.707), (-step*0.707, step*0.707),
                        (step*0.707, -step*0.707), (-step*0.707, -step*0.707)
                    ]
                    for dx, dy in directions:
                        new_pt = pts.copy()
                        new_pt[i] = pts[i] + np.array([dx, dy])
                        score = compute_score(new_pt)
                        if score > best_local_score + 1e-12:
                            best_local_score = score
                            best_local = new_pt[i].copy()
                            improved = True
                
                if improved:
                    pts[i] = best_local
            
            if not improved:
                break
        
        final_score = compute_score(pts)
        if final_score > best_score:
            best_score = final_score
            best_points = pts.copy()
    
    return best_points
