import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def compute_scores(polygon):
    n = 20
    scores = np.zeros(n)
    for i in range(n):
        v = polygon[i]
        dists = np.linalg.norm(polygon - v, axis=1)
        dists[i] = np.inf
        dists = np.sort(dists)
        if dists[0] < 0.05:
            scores[i] = 0.0
            continue
        diffs = np.diff(dists)
        k = int(np.argmin(diffs))
        D = (dists[k] + dists[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            d_sel = dists[0:4]
        elif k == 17:
            d_sel = dists[15:19]
        else:
            d_sel = dists[k-1:k+3]
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
    return scores

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    mean_dist = np.mean(np.linalg.norm(pts, axis=1))
    if mean_dist <= 0.01:
        return None
    return pts / mean_dist

def _is_valid(pts):
    n = len(pts)
    if n != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                return False
    # Check convexity
    v_curr = pts
    v_next = np.roll(pts, -1, axis=0)
    v_next2 = np.roll(pts, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    if np.any(cross <= 1e-6):
        return False
    return True

def _min_vertex_score(pts):
    scores = compute_scores(pts)
    return np.min(scores)

def _support_func_to_polygon(coeffs, n_harmonics=4, n_samples=20):
    """
    Convert Fourier coefficients of support function to polygon vertices.
    h(theta) = sum_{k=0}^{n_h} (a_k cos(k*theta) + b_k sin(k*theta))
    
    The polygon is the intersection of half-planes x*cos(theta_i) + y*sin(theta_i) <= h(theta_i)
    for theta_i = 2*pi*i/n_samples.
    
    We compute the vertices as the intersection of consecutive half-planes.
    """
    theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    # Compute support function values
    h = np.zeros(n_samples)
    for k in range(n_harmonics + 1):
        a_k = coeffs[2*k]
        b_k = coeffs[2*k + 1]
        h += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
    
    # Ensure h is positive (for convexity)
    # If any h is negative, the polygon might not contain the origin
    # We'll add a small offset to ensure positivity
    if np.min(h) < 0.1:
        h += 0.1 - np.min(h)
    
    # For each pair of consecutive half-planes, find the intersection
    # Half-plane i: x*cos(theta_i) + y*sin(theta_i) <= h_i
    # Intersection of half-planes i and i+1:
    #   x*cos(theta_i) + y*sin(theta_i) = h_i
    #   x*cos(theta_{i+1}) + y*sin(theta_{i+1}) = h_{i+1}
    
    vertices = np.zeros((n_samples, 2))
    for i in range(n_samples):
        i_next = (i + 1) % n_samples
        
        c1 = np.cos(theta[i])
        s1 = np.sin(theta[i])
        c2 = np.cos(theta[i_next])
        s2 = np.sin(theta[i_next])
        
        det = c1 * s2 - s1 * c2
        if abs(det) < 1e-10:
            # Degenerate, use midpoint
            vertices[i] = np.array([(h[i] + h[i_next]) / 2, 0])
            continue
        
        x = (h[i] * s2 - h[i_next] * s1) / det
        y = (c1 * h[i_next] - c2 * h[i]) / det
        vertices[i] = np.array([x, y])
    
    return vertices

def _fourier_radial_to_polygon(coeffs, n_harmonics=4, n_samples=20):
    """
    Convert Fourier coefficients of radial function to polygon vertices.
    r(theta) = 1 + sum_{k=1}^{n_h} (a_k cos(k*theta) + b_k sin(k*theta))
    
    This is a simpler parameterization that guarantees convexity for small perturbations.
    """
    theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    r = np.ones(n_samples)
    for k in range(1, n_harmonics + 1):
        a_k = coeffs[2*(k-1)]
        b_k = coeffs[2*(k-1) + 1]
        r += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
    
    # Ensure radius is positive
    r = np.maximum(r, 0.3)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _objective_support(params, n_harmonics=4):
    """Objective for support function optimization."""
    n_samples = 20
    polygon = _support_func_to_polygon(params, n_harmonics, n_samples)
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    # Check convexity
    v_curr = polygon_norm
    v_next = np.roll(polygon_norm, -1, axis=0)
    v_next2 = np.roll(polygon_norm, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    if penalty > 1e-4:
        return 100.0 + penalty
    
    scores = compute_scores(polygon_norm)
    min_score = np.min(scores)
    return -min_score

def _objective_radial(params, n_harmonics=4):
    """Objective for radial function optimization."""
    n_samples = 20
    polygon = _fourier_radial_to_polygon(params, n_harmonics, n_samples)
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    v_curr = polygon_norm
    v_next = np.roll(polygon_norm, -1, axis=0)
    v_next2 = np.roll(polygon_norm, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    if penalty > 1e-4:
        return 100.0 + penalty
    
    scores = compute_scores(polygon_norm)
    min_score = np.min(scores)
    return -min_score

def _soft_min_objective(params, temperature=0.05, n_harmonics=4):
    """Smooth surrogate using log-sum-exp."""
    n_samples = 20
    polygon = _fourier_radial_to_polygon(params, n_harmonics, n_samples)
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    v_curr = polygon_norm
    v_next = np.roll(polygon_norm, -1, axis=0)
    v_next2 = np.roll(polygon_norm, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    if penalty > 1e-4:
        return 100.0 + penalty
    
    scores = compute_scores(polygon_norm)
    scores_clipped = np.maximum(scores, 1e-10)
    neg_scores = -scores_clipped
    max_neg = np.max(neg_scores)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg)))
    soft_min_val = (1.0 / temperature) * log_sum
    
    return soft_min_val

def _free_angle_radius_to_polygon(params):
    """Convert angle/radius parameters to polygon."""
    n = 20
    angles = params[:n]
    radii = params[n:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _objective_free(params):
    """Objective for free angle/radius optimization."""
    polygon = _free_angle_radius_to_polygon(params)
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    v_curr = polygon_norm
    v_next = np.roll(polygon_norm, -1, axis=0)
    v_next2 = np.roll(polygon_norm, -2, axis=0)
    e1 = v_next - v_curr
    e2 = v_next2 - v_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    neg_cross = np.minimum(cross, 0)
    penalty = np.sum(neg_cross**2)
    
    if penalty > 1e-4:
        return 100.0 + penalty
    
    scores = compute_scores(polygon_norm)
    min_score = np.min(scores)
    return -min_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    n = 20
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_polygon = regular.copy()
    best_score = _min_vertex_score(_normalize(regular))
    
    # Phase 1: Differential Evolution with radial Fourier parameterization
    try:
        if time.time() < deadline - 180:
            n_harmonics = 4
            n_params = 2 * n_harmonics
            bounds = [(-0.3, 0.3)] * n_params
            
            de_result = differential_evolution(
                _objective_radial, bounds, args=(n_harmonics,),
                seed=42, maxiter=150, tol=1e-5, popsize=25,
                mutation=(0.5, 1.0), recombination=0.7,
                polish=False, init='latinhypercube', updating='immediate'
            )
            
            if de_result.fun < 100:
                polygon = _fourier_radial_to_polygon(de_result.x, n_harmonics, n)
                polygon_norm = _normalize(polygon)
                if polygon_norm is not None and _is_valid(polygon_norm):
                    s = _min_vertex_score(polygon_norm)
                    if s > best_score:
                        best_score = s
                        best_polygon = polygon_norm
    except Exception:
        pass
    
    # Phase 2: Differential Evolution with support function parameterization
    try:
        if time.time() < deadline - 150:
            n_harmonics = 4
            n_params = 2 * (n_harmonics + 1)
            # a0 should be around 1 for a reasonable polygon
            bounds = [(0.5, 1.5)] + [(-0.2, 0.2)] * (n_params - 1)
            
            de_result2 = differential_evolution(
                _objective_support, bounds, args=(n_harmonics,),
                seed=123, maxiter=100, tol=1e-5, popsize=20,
                mutation=(0.5, 1.0), recombination=0.7,
                polish=False, init='latinhypercube', updating='immediate'
            )
            
            if de_result2.fun < 100:
                polygon = _support_func_to_polygon(de_result2.x, n_harmonics, n)
                polygon_norm = _normalize(polygon)
                if polygon_norm is not None and _is_valid(polygon_norm):
                    s = _min_vertex_score(polygon_norm)
                    if s > best_score:
                        best_score = s
                        best_polygon = polygon_norm
    except Exception:
        pass
    
    # Phase 3: Local polish with Nelder-Mead on free angle/radius
    try:
        if time.time() < deadline - 60:
            pts = best_polygon
            centroid = np.mean(pts, axis=0)
            pts_centered = pts - centroid
            radii_init = np.linalg.norm(pts_centered, axis=1)
            angles_init = np.arctan2(pts_centered[:, 1], pts_centered[:, 0])
            x0_free = np.concatenate([angles_init, radii_init])
            
            polish_result = minimize(
                _objective_free, x0_free,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            
            if polish_result.fun < 100:
                polygon = _free_angle_radius_to_polygon(polish_result.x)
                polygon_norm = _normalize(polygon)
                if polygon_norm is not None and _is_valid(polygon_norm):
                    s = _min_vertex_score(polygon_norm)
                    if s > best_score:
                        best_score = s
                        best_polygon = polygon_norm
    except Exception:
        pass
    
    # Phase 4: Soft-min optimization for smoother landscape
    try:
        if time.time() < deadline - 30 and best_score > 0.5:
            n_harmonics = 4
            n_params = 2 * n_harmonics
            
            # Convert current best to radial Fourier coefficients
            pts = best_polygon
            centroid = np.mean(pts, axis=0)
            pts_centered = pts - centroid
            radii_current = np.linalg.norm(pts_centered, axis=1)
            angles_current = np.arctan2(pts_centered[:, 1], pts_centered[:, 0])
            
            # Fit Fourier coefficients to current radii
            theta = angles_current
            r = radii_current - 1.0  # Subtract base radius
            
            # Simple least squares fit
            A = np.zeros((n, n_params))
            for k in range(1, n_harmonics + 1):
                A[:, 2*(k-1)] = np.cos(k * theta)
                A[:, 2*(k-1) + 1] = np.sin(k * theta)
            
            coeffs_init, _, _, _ = np.linalg.lstsq(A, r, rcond=None)
            
            for temp in [0.1, 0.05, 0.02]:
                if time.time() >= deadline - 20:
                    break
                
                soft_obj = lambda p: _soft_min_objective(p, temperature=temp, n_harmonics=n_harmonics)
                
                soft_result = minimize(
                    soft_obj, coeffs_init,
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                
                if soft_result.fun < 100:
                    polygon = _fourier_radial_to_polygon(soft_result.x, n_harmonics, n)
                    polygon_norm = _normalize(polygon)
                    if polygon_norm is not None and _is_valid(polygon_norm):
                        s = _min_vertex_score(polygon_norm)
                        if s > best_score:
                            best_score = s
                            best_polygon = polygon_norm
                            # Update coeffs_init for next iteration
                            pts_new = polygon_norm
                            centroid_new = np.mean(pts_new, axis=0)
                            pts_new_centered = pts_new - centroid_new
                            radii_new = np.linalg.norm(pts_new_centered, axis=1)
                            angles_new = np.arctan2(pts_new_centered[:, 1], pts_new_centered[:, 0])
                            theta_new = angles_new
                            r_new = radii_new - 1.0
                            A_new = np.zeros((n, n_params))
                            for k in range(1, n_harmonics + 1):
                                A_new[:, 2*(k-1)] = np.cos(k * theta_new)
                                A_new[:, 2*(k-1) + 1] = np.sin(k * theta_new)
                            coeffs_init, _, _, _ = np.linalg.lstsq(A_new, r_new, rcond=None)
    except Exception:
        pass
    
    # Phase 5: Final high-precision polish
    try:
        if time.time() < deadline - 10 and best_score > 0.5:
            pts = best_polygon
            centroid = np.mean(pts, axis=0)
            pts_centered = pts - centroid
            radii_init = np.linalg.norm(pts_centered, axis=1)
            angles_init = np.arctan2(pts_centered[:, 1], pts_centered[:, 0])
            x0_final = np.concatenate([angles_init, radii_init])
            
            final_result = minimize(
                _objective_free, x0_final,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if final_result.fun < 100:
                polygon = _free_angle_radius_to_polygon(final_result.x)
                polygon_norm = _normalize(polygon)
                if polygon_norm is not None and _is_valid(polygon_norm):
                    s = _min_vertex_score(polygon_norm)
                    if s > best_score:
                        best_score = s
                        best_polygon = polygon_norm
    except Exception:
        pass
    
    # Fallback to regular polygon if nothing worked or invalid
    if best_score <= 0 or not _is_valid(best_polygon):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_polygon
