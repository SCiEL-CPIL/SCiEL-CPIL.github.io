import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _vertex_score(polygon, i):
    """Calculate the score for vertex i of the polygon."""
    v = polygon[i]
    diffs = polygon - v
    diffs = np.delete(diffs, i, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    dists = np.sort(dists)
    
    if dists[0] < 0.05:
        return 0.0
        
    gaps = np.diff(dists)
    min_gap = np.min(gaps)
    k = np.argmin(gaps)
    
    D = (dists[k] + dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
        
    if k == 0:
        ds = dists[0:4]
    elif k == 17:
        ds = dists[15:19]
    else:
        ds = dists[k-1:k+3]
        
    ratios = np.maximum(ds / D, D / ds)
    raw = np.mean(ratios)
    
    if raw == 0:
        return 1.0
    return min(1.0, 1.0 / raw)

def _total_score(polygon):
    """Calculate the minimum vertex score over all vertices."""
    min_score = 1.0
    for i in range(polygon.shape[0]):
        s = _vertex_score(polygon, i)
        if s < min_score:
            min_score = s
            if min_score == 0.0:
                return 0.0
    return min_score

def _is_convex(polygon):
    """Check if polygon is strictly convex (counter-clockwise)."""
    n = polygon.shape[0]
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross <= 1e-9:
            return False
    return True

def _fourier_to_R(coeffs, n, angles):
    """Convert Fourier coefficients to radial values.
    coeffs[0] is the DC component (mean radius),
    coeffs[1] and coeffs[2] are cos and sin of frequency 1,
    coeffs[3] and coeffs[4] are cos and sin of frequency 2, etc.
    """
    R = np.zeros(n)
    R += coeffs[0]
    for m in range(1, len(coeffs) // 2 + 1):
        idx_c = 2 * m - 1
        idx_s = 2 * m
        if idx_c >= len(coeffs) or idx_s >= len(coeffs):
            break
        R += coeffs[idx_c] * np.cos(m * angles)
        R += coeffs[idx_s] * np.sin(m * angles)
    return R

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Regular polygon baseline
    R_reg = np.ones(n)
    x_reg = R_reg * np.cos(angles)
    y_reg = R_reg * np.sin(angles)
    polygon_reg = np.stack([x_reg, y_reg], axis=1)
    
    centroid = np.mean(polygon_reg, axis=0)
    polygon_reg = polygon_reg - centroid
    mean_dist = np.mean(np.linalg.norm(polygon_reg, axis=1))
    if mean_dist > 0.01:
        polygon_reg = polygon_reg / mean_dist
    else:
        return polygon_reg
        
    score_reg = _total_score(polygon_reg)
    
    start_time = time.time()
    time_limit = 630.0
    current_best = score_reg
    best_polygon = polygon_reg.copy()
    
    # Use Fourier representation: DC + up to 3 harmonics = 7 parameters
    # This constrains to smooth shapes, maintaining convexity more easily
    n_harmonics = 3
    n_params = 1 + 2 * n_harmonics  # DC + cos/sin for each harmonic
    
    def get_polygon_from_coeffs(coeffs):
        R = _fourier_to_R(coeffs, n, angles)
        if np.any(R <= 0.3):
            return None
        x = R * np.cos(angles)
        y = R * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        return polygon

    def objective(coeffs):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            return 1e9
            
        polygon = get_polygon_from_coeffs(coeffs)
        if polygon is None:
            return 1e9
            
        if not _is_convex(polygon):
            return 1e9 + (time.time() - start_time)
            
        s = _total_score(polygon)
        return -s
        
    # Bounds: DC component around 1, harmonics small
    bounds = [(0.8, 1.2)]  # DC
    for m in range(1, n_harmonics + 1):
        # Decay harmonic amplitudes
        amp = 0.15 / m
        bounds.append((-amp, amp))
        bounds.append((-amp, amp))
    
    # Multiple restarts with different seeds
    n_restarts = 5
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 60:
            break
            
        seed = 42 + restart * 100
        
        # For restarts > 0, perturb the initial bounds slightly
        if restart == 0:
            bounds_cur = bounds
        else:
            bounds_cur = []
            for b in bounds:
                center = (b[0] + b[1]) / 2
                half = (b[1] - b[0]) / 2
                # Add small random offset to center
                offset = np.random.uniform(-0.02, 0.02)
                bounds_cur.append((center - half + offset, center + half + offset))
        
        try:
            res = differential_evolution(
                objective, 
                bounds=bounds_cur,
                seed=seed,
                maxiter=150,
                tol=1e-5,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False
            )
            
            if res.fun < -current_best:
                coeffs_opt = res.x
                polygon_opt = get_polygon_from_coeffs(coeffs_opt)
                
                if polygon_opt is not None and _is_convex(polygon_opt):
                    centroid_opt = np.mean(polygon_opt, axis=0)
                    polygon_opt = polygon_opt - centroid_opt
                    mean_dist_opt = np.mean(np.linalg.norm(polygon_opt, axis=1))
                    if mean_dist_opt > 0.01:
                        polygon_opt = polygon_opt / mean_dist_opt
                        score_opt = _total_score(polygon_opt)
                        if score_opt > current_best:
                            best_polygon = polygon_opt
                            current_best = score_opt
        except Exception:
            pass
    
    # Local refinement using Nelder-Mead on Fourier coefficients
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            # Extract approximate Fourier coefficients from best_polygon
            x_best = best_polygon[:, 0]
            y_best = best_polygon[:, 1]
            R_current = np.sqrt(x_best**2 + y_best**2)
            
            # Fit Fourier coefficients
            coeffs_init = np.zeros(n_params)
            coeffs_init[0] = np.mean(R_current)
            for m in range(1, n_harmonics + 1):
                idx_c = 2 * m - 1
                idx_s = 2 * m
                coeffs_init[idx_c] = np.mean(R_current * np.cos(m * angles))
                coeffs_init[idx_s] = np.mean(R_current * np.sin(m * angles))
            
            def objective_local(coeffs):
                elapsed_local = time.time() - start_time
                if elapsed_local > time_limit - 10:
                    return 1e9
                    
                polygon = get_polygon_from_coeffs(coeffs)
                if polygon is None:
                    return 1e9
                    
                if not _is_convex(polygon):
                    return 1e9
                    
                s = _total_score(polygon)
                return -s
                
            res_local = minimize(objective_local, coeffs_init, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res_local.fun < -current_best:
                coeffs_opt = res_local.x
                polygon_opt = get_polygon_from_coeffs(coeffs_opt)
                
                if polygon_opt is not None and _is_convex(polygon_opt):
                    centroid_opt = np.mean(polygon_opt, axis=0)
                    polygon_opt = polygon_opt - centroid_opt
                    mean_dist_opt = np.mean(np.linalg.norm(polygon_opt, axis=1))
                    if mean_dist_opt > 0.01:
                        polygon_opt = polygon_opt / mean_dist_opt
                        score_opt = _total_score(polygon_opt)
                        if score_opt > current_best:
                            best_polygon = polygon_opt
                            current_best = score_opt
        except Exception:
            pass
            
    # Final safety check
    if not _is_convex(best_polygon):
        return polygon_reg
        
    return best_polygon
