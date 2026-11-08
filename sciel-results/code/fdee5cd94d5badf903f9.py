import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

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

    def _points_from_fourier(params):
        """
        Generate 13 points from Fourier series parameters.
        params[0] = base radius
        params[1:4] = amplitude, phase for k=1
        params[4:7] = amplitude, phase for k=2
        params[7:10] = amplitude, phase for k=3
        params[10] = center offset x
        params[11] = center offset y
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = params[0]
        
        # Add harmonics
        # k=1
        r += params[1] * np.cos(1 * angles + params[2])
        # k=2
        r += params[3] * np.cos(2 * angles + params[4])
        # k=3
        r += params[5] * np.cos(3 * angles + params[6])
        
        # Ensure positive radius to maintain convexity/star-shapedness relative to origin
        # If radius becomes negative, the point flips, which might break convexity or create self-intersection
        # We clip to a small positive value to prevent degeneracy
        r = np.maximum(r, 0.1)
        
        x = r * np.cos(angles) + params[10]
        y = r * np.sin(angles) + params[11]
        
        return np.column_stack([x, y])

    def _soft_objective(params, beta=50.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = _points_from_fourier(params)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        # Since hull area is 1, score is just min_area
        return -soft_min_area

    def _hard_objective(params):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = _points_from_fourier(params)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Define search space for Differential Evolution
    # params: [radius, amp1, phase1, amp2, phase2, amp3, phase3, cx, cy]
    # Radius ~ 1.0
    # Amplitudes small to preserve convexity, but allow some deformation
    # Phases 0..2pi
    # Center offsets small
    
    bounds = [
        (0.5, 1.5),   # radius
        (0.0, 0.2),   # amp1
        (0.0, 2*np.pi), # phase1
        (0.0, 0.2),   # amp2
        (0.0, 2*np.pi), # phase2
        (0.0, 0.2),   # amp3
        (0.0, 2*np.pi), # phase3
        (-0.2, 0.2),  # cx
        (-0.2, 0.2)   # cy
    ]
    
    best_score = -float('inf')
    best_params = None
    best_points = None

    # Use Differential Evolution to find a good starting point in the low-dim space
    # This helps escape local minima that might trap gradient-based methods from a single start
    if time.time() < start_time + time_limit - 30:
        try:
            de_res = opt.differential_evolution(
                lambda p, b=50.0: _soft_objective(p, b),
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-4,
                polish=False, # We will do our own polish
                update_interval=1,
                popsize=15
            )
            best_params = de_res.x
            best_points = _points_from_fourier(best_params)
            best_score = compute_score(best_points)
        except Exception:
            pass

    # If DE failed or didn't run, use a regular polygon as initial guess
    if best_params is None:
        best_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        best_points = _points_from_fourier(best_params)
        best_score = compute_score(best_points)

    # Refine using L-BFGS-B on the low-dimensional parameter space
    # This is much faster and more stable than optimizing 26 variables directly
    current_params = best_params.copy()
    
    # Annealing beta for smooth refinement
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > start_time + time_limit - 10:
            break
        try:
            res = opt.minimize(
                lambda p, b=beta: _soft_objective(p, b),
                current_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            current_params = res.x
            current_points = _points_from_fourier(current_params)
            score = compute_score(current_points)
            
            if score > best_score:
                best_score = score
                best_points = current_points.copy()
                best_params = current_params.copy()
        except Exception:
            pass

    # Final high-precision polish on the hard objective
    if time.time() < start_time + time_limit - 5:
        try:
            res = opt.minimize(
                _hard_objective,
                current_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            final_params = res.x
            final_points = _points_from_fourier(final_params)
            final_score = compute_score(final_points)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
                best_params = final_params.copy()
        except Exception:
            pass

    # Fallback if something went wrong
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
