import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit

def _compute_score(points):
    """Compute the minimum vertex score for a given polygon (vectorized)."""
    n = len(points)
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    dists_to_centroid = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    pts = pts / mean_dist
    
    # Check strict convexity
    edges = np.roll(pts, -1, axis=0) - pts
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0
    
    # Check no repeated points
    min_dist = np.min(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2))
    if min_dist < 1e-10:
        return 0.0
    
    # Compute vertex scores vectorized
    # Distance matrix
    diff = pts[:, None, :] - pts[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    
    min_score = 1.0
    for i in range(n):
        d_sorted = np.sort(dist_matrix[i])
        if d_sorted[0] < 0.05:
            min_score = 0.0
            break
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            min_score = 0.0
            break
        if k == 0:
            four = d_sorted[0:4]
        elif k == 17:
            four = d_sorted[15:19]
        else:
            four = d_sorted[k-1:k+3]
        ratios = np.maximum(four / D, D / four)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        min_score = min(min_score, score)
    return min_score

def _angles_to_points(angles, n=20):
    """Convert angles (summing to 2*pi) to polygon vertices on unit circle."""
    # Ensure angles are positive and sum to 2*pi
    angles = np.maximum(angles, 1e-6)
    angles = angles / np.sum(angles) * 2 * np.pi
    
    # Cumulative angles
    cum_angles = np.concatenate([[0], np.cumsum(angles)])
    x = np.cos(cum_angles[:-1])
    y = np.sin(cum_angles[:-1])
    return np.stack([x, y], axis=1)

def _fourier_to_angles(params, n=20):
    """Convert Fourier mode coefficients to angles.
    params: [a1, b1, a2, b2, a3, b3, a4, b4]
    """
    base_angle = 2 * np.pi / n
    angles = np.full(n, base_angle)
    for k in range(1, 5):
        a = params[2*(k-1)]
        b = params[2*(k-1)+1]
        theta = 2 * np.pi * k * np.arange(n) / n
        angles += a * np.cos(theta) + b * np.sin(theta)
    
    # Ensure angles are positive
    angles = np.maximum(angles, 0.01)
    
    # Normalize to sum to 2*pi
    angles = angles / np.sum(angles) * 2 * np.pi
    return angles

def _score_from_angles(angles):
    pts = _angles_to_points(angles)
    return _compute_score(pts)

def _neg_score_from_angles(angles):
    return -_score_from_angles(angles)

def _neg_score_from_fourier(params):
    angles = _fourier_to_angles(params)
    return -_score_from_angles(angles)

def _coordinate_descent_refine(angles, max_iter=200, tol=1e-10):
    """Coordinate descent refinement on angles."""
    n = len(angles)
    current = angles.copy()
    current_score = _score_from_angles(current)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            # Try adjusting angle[i] while keeping others fixed
            # Normalize: if we change angle[i], we need to keep sum = 2*pi
            # So adjust angle[i] and scale others proportionally
            best_val = current[i]
            best_s = current_score
            
            for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                new_angle = current[i] + delta
                if new_angle < 0.01:
                    continue
                new_angles = current.copy()
                new_angles[i] = new_angle
                # Renormalize
                new_angles = new_angles / np.sum(new_angles) * 2 * np.pi
                s = _score_from_angles(new_angles)
                if s > best_s + tol:
                    best_s = s
                    best_val = new_angle
            
            if best_val != current[i]:
                current[i] = best_val
                current = current / np.sum(current) * 2 * np.pi
                current_score = best_s
                improved = True
        
        if not improved:
            break
    
    return current, current_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Initial guess: regular polygon
    angles0 = np.full(n, 2 * np.pi / n)
    best_points = _angles_to_points(angles0)
    best_score = _compute_score(best_points)
    
    np.random.seed(42)
    
    # Phase 1: Differential evolution on Fourier parameters
    if time.time() < deadline - 150:
        try:
            bounds = [(-0.3, 0.3)] * 8
            result = differential_evolution(
                _neg_score_from_fourier, bounds,
                seed=42, maxiter=100, tol=1e-8,
                polish=True, init='sobol',
                workers=1, popsize=25
            )
            if result.fun < -best_score:
                params_opt = result.x
                angles_opt = _fourier_to_angles(params_opt)
                pts = _angles_to_points(angles_opt)
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
                    angles0 = angles_opt
        except Exception:
            pass
    
    # Phase 2: Coordinate descent refinement
    if time.time() < deadline - 60:
        try:
            angles_refined, score_refined = _coordinate_descent_refine(angles0, max_iter=300)
            if score_refined > best_score:
                best_score = score_refined
                best_points = _angles_to_points(angles_refined)
                angles0 = angles_refined
        except Exception:
            pass
    
    # Phase 3: Local Nelder-Mead on Fourier params
    if time.time() < deadline - 30:
        try:
            if 'result' in locals() and result.fun < -best_score:
                init_params = result.x
            else:
                init_params = np.zeros(8)
            
            result_local = minimize(_neg_score_from_fourier, init_params, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10,
                                      'adaptive': True})
            angles_local = _fourier_to_angles(result_local.x)
            pts = _angles_to_points(angles_local)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
                angles0 = angles_local
        except Exception:
            pass
    
    # Phase 4: Final coordinate descent
    if time.time() < deadline - 10:
        try:
            angles_final, score_final = _coordinate_descent_refine(angles0, max_iter=500)
            if score_final > best_score:
                best_score = score_final
                best_points = _angles_to_points(angles_final)
        except Exception:
            pass
    
    return best_points
