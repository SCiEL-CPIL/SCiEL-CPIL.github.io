# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return centered
    return centered / mean_dist

def _is_valid_convex(polygon):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turning +-2*pi."""
    n = len(polygon)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    # Compute cross products of consecutive edges
    cross_prods = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    # All must be nonzero and same sign
    if np.any(np.abs(cross_prods) < 1e-12):
        return False
    if np.any(cross_prods > 0) and np.any(cross_prods < 0):
        return False
    # Total turning should be +-2*pi
    total_turn = 0.0
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        angle1 = np.arctan2(e1[1], e1[0])
        angle2 = np.arctan2(e2[1], e2[0])
        d_angle = angle2 - angle1
        while d_angle > np.pi:
            d_angle -= 2 * np.pi
        while d_angle < -np.pi:
            d_angle += 2 * np.pi
        total_turn += d_angle
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return False
    return True

def _polygon_score_fast(polygon):
    """Compute the minimum vertex score over all vertices."""
    n = len(polygon)
    # Compute all pairwise distances at once
    diffs = polygon[:, np.newaxis, :] - polygon[np.newaxis, :, :]
    dists_sq = np.einsum('ijk,ijk->ij', diffs, diffs)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    scores = np.empty(n)
    for i in range(n):
        dists_i = np.sort(dists[i])
        
        if dists_i[0] < 0.05:
            scores[i] = 0.0
            continue
        
        gaps = dists_i[1:] - dists_i[:-1]
        k = np.argmin(gaps)
        D = (dists_i[k] + dists_i[k + 1]) / 2
        
        if D < 0.1:
            scores[i] = 0.0
            continue
        
        if k == 0:
            four_dists = dists_i[0:4]
        elif k == 17:
            four_dists = dists_i[15:19]
        else:
            four_dists = dists_i[k - 1:k + 3]
        
        ratios_a = four_dists / D
        ratios_b = D / four_dists
        raw = np.mean(np.maximum(ratios_a, ratios_b))
        scores[i] = min(1.0, 1.0 / raw)
    
    return np.min(scores)

def _score_from_angles(angles):
    """Compute score given angles for fixed radii (all 1)."""
    n = len(angles)
    angles_sorted = np.sort(angles)
    polygon = np.stack([np.cos(angles_sorted), np.sin(angles_sorted)], axis=1)
    polygon = _normalize(polygon)
    
    if not _is_valid_convex(polygon):
        return -1.0
    
    return _polygon_score_fast(polygon)

def _score_from_radii(radii):
    """Compute score given radii for fixed angles."""
    n = len(radii)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.maximum(radii, 0.1)
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    
    if not _is_valid_convex(polygon):
        return -1.0
    
    return _polygon_score_fast(polygon)

def _score_from_angles_radii(params):
    """Compute score given both angles and radii."""
    n = 10  # half of 20
    angles_half = params[:n]
    radii_half = params[n:]
    
    # Create full angles by symmetry (20 points)
    angles = np.zeros(20)
    angles[:10] = angles_half
    angles[10:] = 2*np.pi - angles_half[::-1]
    angles = np.sort(angles)
    
    radii = np.zeros(20)
    radii[:10] = radii_half
    radii[10:] = radii_half[::-1]
    
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    
    if not _is_valid_convex(polygon):
        return -1.0
    
    return _polygon_score_fast(polygon)

def _local_refinement(params, step_size, n_steps, deadline, score_func):
    """Perform local refinement with adaptive step sizes."""
    n = len(params)
    best_params = params.copy()
    best_score = score_func(best_params)
    if best_score < 0:
        return params, 0.0
    
    current_step = step_size
    for step in range(n_steps):
        if time.time() > deadline - 15.0:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 15.0:
                break
            
            # Try positive and negative perturbations
            for direction in [1, -1]:
                trial = best_params.copy()
                trial[i] += direction * current_step
                # Ensure angles are in valid range
                if i < 10:
                    trial[i] = np.clip(trial[i], 0.01, 2*np.pi - 0.01)
                else:
                    trial[i] = np.maximum(trial[i], 0.3)
                
                score = score_func(trial)
                if score > best_score:
                    best_params = trial
                    best_score = score
                    improved = True
        
        if not improved:
            current_step *= 0.5
            if current_step < 1e-5:
                break
        else:
            current_step *= 0.85
    
    return best_params, best_score

def _nm_optimize(params, deadline, score_func, bounds):
    """Nelder-Mead optimization."""
    n = len(params)
    
    def objective(x):
        if time.time() > deadline - 15.0:
            return 1e9
        score = score_func(x)
        if score < 0:
            return 1e9
        return -score
    
    try:
        res = minimize(objective, params, method='Nelder-Mead',
                      options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < 1e8:
            return res.x, -res.fun
    except Exception:
        pass
    
    return params, score_func(params)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses multi-phase optimization over both angles and radii.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Phase 1: Start with regular polygon
    best_radii = np.ones(n)
    best_score = _score_from_radii(best_radii)
    
    # Phase 2: Differential evolution on radii
    if time.time() < deadline - 120.0:
        def de_objective(radii):
            if time.time() > deadline - 120.0:
                return 1e9
            score = _score_from_radii(radii)
            if score < 0:
                return 1e9
            return -score
        
        bounds = [(0.3, 2.0)] * n
        try:
            de_result = differential_evolution(de_objective, bounds, maxiter=50, 
                                               popsize=15, tol=1e-5, seed=42,
                                               polish=False, workers=1)
            if de_result.fun < 1e8:
                de_radii = de_result.x
                de_score = _score_from_radii(de_radii)
                if de_score > best_score:
                    best_radii = de_radii
                    best_score = de_score
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead on radii
    if time.time() < deadline - 60.0:
        nm_radii, nm_score = _nm_optimize(best_radii, deadline, _score_from_radii, None)
        if nm_score > best_score:
            best_radii = nm_radii
            best_score = nm_score
    
    # Phase 4: Local refinement on radii
    if time.time() < deadline - 30.0:
        refined_radii, refined_score = _local_refinement(best_radii, 0.02, 50, deadline, _score_from_radii)
        if refined_score > best_score:
            best_radii = refined_radii
            best_score = refined_score
    
    # Phase 5: Try optimizing angles with fixed radii pattern
    if time.time() < deadline - 40.0:
        # Use the best radii pattern but optimize angles
        # Parameterize angles as deviations from regular
        n_dev = n
        def angle_params_to_polygon(params):
            dev = params[:n_dev]
            a = angles + dev
            a = np.sort(a)
            # Wrap around
            a = a % (2*np.pi)
            a = np.sort(a)
            # Ensure spacing
            polygon = np.stack([best_radii * np.cos(a), best_radii * np.sin(a)], axis=1)
            return _normalize(polygon)
        
        def angle_score_func(params):
            if time.time() > deadline - 40.0:
                return -1.0
            polygon = angle_params_to_polygon(params)
            if not _is_valid_convex(polygon):
                return -1.0
            return _polygon_score_fast(polygon)
        
        initial_dev = np.zeros(n_dev)
        nm_dev, nm_dev_score = _nm_optimize(initial_dev, deadline, angle_score_func, None)
        if nm_dev_score > best_score:
            # Reconstruct with optimized angles
            a = angles + nm_dev[:n_dev]
            a = np.sort(a % (2*np.pi))
            polygon = np.stack([best_radii * np.cos(a), best_radii * np.sin(a)], axis=1)
            best_polygon = _normalize(polygon)
            best_score = nm_dev_score
            return best_polygon
    
    # Reconstruct final polygon
    best_radii = np.maximum(best_radii, 0.1)
    p = np.stack([best_radii * np.cos(angles), best_radii * np.sin(angles)], axis=1)
    best_polygon = _normalize(p)
    
    # Final validity check
    if not _is_valid_convex(best_polygon):
        # Fallback to regular polygon
        p = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best_polygon = _normalize(p)
    
    return best_polygon

# EVOLVE-BLOCK-END
