import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

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

def _normalize(polygon):
    """Normalize polygon: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(polygon, axis=0)
    polygon_norm = polygon - centroid
    dists_to_centroid = np.linalg.norm(polygon_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return None
    return polygon_norm / mean_dist

def _fourier_to_radii(coeffs, n, angles):
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

def _radii_to_polygon(R, n, angles):
    """Convert radii to polygon coordinates."""
    if np.any(R <= 0.1):
        return None
    x = R * np.cos(angles)
    y = R * np.sin(angles)
    return np.stack([x, y], axis=1)

def _polygon_to_fourier_coeffs(polygon, n, angles, n_harmonics):
    """Extract Fourier coefficients from a polygon."""
    x = polygon[:, 0]
    y = polygon[:, 1]
    R = np.sqrt(x**2 + y**2)
    
    n_params = 1 + 2 * n_harmonics
    coeffs = np.zeros(n_params)
    coeffs[0] = np.mean(R)
    for m in range(1, n_harmonics + 1):
        idx_c = 2 * m - 1
        idx_s = 2 * m
        coeffs[idx_c] = np.mean(R * np.cos(m * angles))
        coeffs[idx_s] = np.mean(R * np.sin(m * angles))
    return coeffs

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Regular polygon baseline
    R_reg = np.ones(n)
    polygon_reg = _radii_to_polygon(R_reg, n, angles)
    polygon_reg = _normalize(polygon_reg)
    if polygon_reg is None:
        # Fallback: return regular polygon
        x = R_reg * np.cos(angles)
        y = R_reg * np.sin(angles)
        return np.stack([x, y], axis=1)
        
    score_reg = _total_score(polygon_reg)
    best_polygon = polygon_reg.copy()
    current_best = score_reg
    
    start_time = time.time()
    time_limit = 600.0  # Safety margin before 630s kill
    
    # Fourier representation: DC + up to 4 harmonics = 9 parameters
    n_harmonics = 4
    n_params = 1 + 2 * n_harmonics
    
    def get_polygon_from_coeffs(coeffs):
        R = _fourier_to_radii(coeffs, n, angles)
        if np.any(R <= 0.3):
            return None
        return _radii_to_polygon(R, n, angles)
    
    def objective(coeffs):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            return 1e9
            
        polygon = get_polygon_from_coeffs(coeffs)
        if polygon is None:
            return 1e9
            
        if not _is_convex(polygon):
            return 1e9
            
        s = _total_score(polygon)
        return -s
        
    # Phase 1: Differential Evolution for global search
    # Bounds: DC component around 1, harmonics small with decay
    bounds = [(0.8, 1.2)]  # DC
    for m in range(1, n_harmonics + 1):
        amp = 0.12 / m
        bounds.append((-amp, amp))
        bounds.append((-amp, amp))
    
    # Multiple restarts with different seeds and perturbed bounds
    n_restarts = 8
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 120:
            break
            
        seed = 42 + restart * 137
        
        # Perturb bounds for diversity
        if restart == 0:
            bounds_cur = bounds
        else:
            bounds_cur = []
            for b in bounds:
                center = (b[0] + b[1]) / 2
                half = (b[1] - b[0]) / 2
                offset = np.random.uniform(-0.03, 0.03)
                bounds_cur.append((center - half + offset, center + half + offset))
        
        try:
            res = differential_evolution(
                objective, 
                bounds=bounds_cur,
                seed=seed,
                maxiter=200,
                tol=1e-6,
                popsize=12,
                mutation=(0.6, 1.2),
                recombination=0.7,
                polish=False
            )
            
            if res.fun < -current_best:
                coeffs_opt = res.x
                polygon_opt = get_polygon_from_coeffs(coeffs_opt)
                
                if polygon_opt is not None and _is_convex(polygon_opt):
                    polygon_norm = _normalize(polygon_opt)
                    if polygon_norm is not None:
                        score_opt = _total_score(polygon_norm)
                        if score_opt > current_best:
                            best_polygon = polygon_norm
                            current_best = score_opt
        except Exception:
            pass
    
    # Phase 2: Local refinement using Nelder-Mead on Fourier coefficients
    elapsed = time.time() - start_time
    if elapsed < time_limit - 60:
        try:
            coeffs_init = _polygon_to_fourier_coeffs(best_polygon, n, angles, n_harmonics)
            
            def objective_local(coeffs):
                elapsed_local = time.time() - start_time
                if elapsed_local > time_limit - 15:
                    return 1e9
                    
                polygon = get_polygon_from_coeffs(coeffs)
                if polygon is None:
                    return 1e9
                    
                if not _is_convex(polygon):
                    return 1e9
                    
                s = _total_score(polygon)
                return -s
                
            res_local = minimize(objective_local, coeffs_init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res_local.fun < -current_best:
                coeffs_opt = res_local.x
                polygon_opt = get_polygon_from_coeffs(coeffs_opt)
                
                if polygon_opt is not None and _is_convex(polygon_opt):
                    polygon_norm = _normalize(polygon_opt)
                    if polygon_norm is not None:
                        score_opt = _total_score(polygon_norm)
                        if score_opt > current_best:
                            best_polygon = polygon_norm
                            current_best = score_opt
        except Exception:
            pass
    
    # Phase 3: Additional local refinement with smaller step sizes
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            coeffs_init = _polygon_to_fourier_coeffs(best_polygon, n, angles, n_harmonics)
            
            def objective_fine(coeffs):
                elapsed_fine = time.time() - start_time
                if elapsed_fine > time_limit - 5:
                    return 1e9
                    
                polygon = get_polygon_from_coeffs(coeffs)
                if polygon is None:
                    return 1e9
                    
                if not _is_convex(polygon):
                    return 1e9
                    
                s = _total_score(polygon)
                return -s
                
            res_fine = minimize(objective_fine, coeffs_init, method='Nelder-Mead',
                                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12,
                                        'initial_simplex': None})
            
            if res_fine.fun < -current_best:
                coeffs_opt = res_fine.x
                polygon_opt = get_polygon_from_coeffs(coeffs_opt)
                
                if polygon_opt is not None and _is_convex(polygon_opt):
                    polygon_norm = _normalize(polygon_opt)
                    if polygon_norm is not None:
                        score_opt = _total_score(polygon_norm)
                        if score_opt > current_best:
                            best_polygon = polygon_norm
                            current_best = score_opt
        except Exception:
            pass
    
    # Final validation and return
    if _is_convex(best_polygon):
        return best_polygon
    else:
        # Fallback to regular polygon
        x = R_reg * np.cos(angles)
        y = R_reg * np.sin(angles)
        return np.stack([x, y], axis=1)
