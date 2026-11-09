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
    """Compute the minimum vertex score over all vertices (vectorized)."""
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

def _polygon_score_smooth(polygon, beta=50.0):
    """Compute a smooth surrogate of the minimum vertex score using soft-min."""
    n = len(polygon)
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
    
    # Soft-min: -log(sum(exp(-beta * scores))) / beta
    min_score = np.min(scores)
    if min_score <= 0:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-beta * scores)))
    soft_min = -log_sum / beta
    return soft_min

def _polygon_from_angles(angles):
    """Build polygon from angles (unit radii) and normalize."""
    angles = np.asarray(angles, dtype=float)
    p = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return _normalize(p)

def _score_from_angles(angles):
    """Compute score given angles."""
    polygon = _polygon_from_angles(angles)
    if not _is_valid_convex(polygon):
        return -1.0
    return _polygon_score_fast(polygon)

def _score_from_angles_smooth(angles, beta=50.0):
    """Compute smooth score given angles."""
    polygon = _polygon_from_angles(angles)
    if not _is_valid_convex(polygon):
        return -1.0
    return _polygon_score_smooth(polygon, beta)

def _score_from_radii(radii, angles):
    """Compute score given radii for fixed angles."""
    radii = np.maximum(radii, 0.1)
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    if not _is_valid_convex(polygon):
        return -1.0
    return _polygon_score_fast(polygon)

def _score_from_radii_smooth(radii, angles, beta=50.0):
    """Compute smooth score given radii for fixed angles."""
    radii = np.maximum(radii, 0.1)
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    if not _is_valid_convex(polygon):
        return -1.0
    return _polygon_score_smooth(polygon, beta)

def _bipartite_polygon(R1, R2, offset):
    """Build polygon with alternating radii on a 10-gon angular grid."""
    n10 = 10
    base_angles = np.linspace(0, 2 * np.pi, n10, endpoint=False) + offset
    # Alternate: even indices get R1, odd get R2
    radii = np.zeros(20)
    angles = np.zeros(20)
    for i in range(n10):
        radii[2*i] = R1
        angles[2*i] = base_angles[i]
        radii[2*i+1] = R2
        angles[2*i+1] = base_angles[i] + np.pi / n10  # half-step offset
    p = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    return _normalize(p)

def _score_bipartite(params, beta=None):
    """Score for bipartite parameterization. params = [R1, R2, offset]."""
    R1, R2, offset = params
    R1 = max(R1, 0.1)
    R2 = max(R2, 0.1)
    polygon = _bipartite_polygon(R1, R2, offset)
    if not _is_valid_convex(polygon):
        return -1.0
    if beta is not None:
        return _polygon_score_smooth(polygon, beta)
    return _polygon_score_fast(polygon)

def _local_refinement_angles(angles, step_size, n_steps, deadline):
    """Local refinement on angles with adaptive step sizes."""
    n = len(angles)
    best_angles = angles.copy()
    best_score = _score_from_angles(best_angles)
    if best_score < 0:
        return angles, 0.0
    
    current_step = step_size
    for step in range(n_steps):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 10.0:
                break
            
            for direction in [1, -1]:
                trial = best_angles.copy()
                trial[i] += direction * current_step
                score = _score_from_angles(trial)
                if score > best_score:
                    best_angles = trial
                    best_score = score
                    improved = True
        
        if not improved:
            current_step *= 0.5
            if current_step < 1e-5:
                break
        else:
            current_step *= 0.8
    
    return best_angles, best_score

def _de_optimize_angles(angles_init, deadline):
    """Use differential evolution to optimize angles globally."""
    n = len(angles_init)
    
    def objective(angles):
        if time.time() > deadline - 10.0:
            return 1e9
        score = _score_from_angles(angles)
        if score < 0:
            return 1e9
        return -score
    
    try:
        res = differential_evolution(objective, bounds=[(-np.pi, np.pi)] * n,
                                      maxiter=100, popsize=15, seed=42,
                                      tol=1e-4, atol=1e-4, polish=False,
                                      updating='immediate', init='sobol')
        if res.fun < 1e8:
            return res.x, -res.fun
    except Exception:
        pass
    
    return angles_init, _score_from_angles(angles_init)

def _nm_optimize_angles(angles_init, deadline):
    """Nelder-Mead refinement on angles."""
    n = len(angles_init)
    
    def objective(angles):
        if time.time() > deadline - 10.0:
            return 1e9
        score = _score_from_angles(angles)
        if score < 0:
            return 1e9
        return -score
    
    try:
        res = minimize(objective, angles_init, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < 1e8:
            return res.x, -res.fun
    except Exception:
        pass
    
    return angles_init, _score_from_angles(angles_init)

def _nm_optimize_radii(radii_init, angles, deadline):
    """Nelder-Mead refinement on radii for fixed angles."""
    n = len(radii_init)
    
    def objective(radii):
        if time.time() > deadline - 10.0:
            return 1e9
        score = _score_from_radii(radii, angles)
        if score < 0:
            return 1e9
        return -score
    
    try:
        res = minimize(objective, radii_init, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < 1e8:
            return res.x, -res.fun
    except Exception:
        pass
    
    return radii_init, _score_from_radii(radii_init, angles)

def _nm_optimize_bipartite(params_init, deadline, beta=50.0):
    """Nelder-Mead refinement on bipartite parameters with smooth surrogate."""
    
    def objective(params):
        if time.time() > deadline - 10.0:
            return 1e9
        score = _score_bipartite(params, beta=beta)
        if score < 0:
            return 1e9
        return -score
    
    try:
        res = minimize(objective, params_init, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < 1e8:
            return res.x, -res.fun
    except Exception:
        pass
    
    return params_init, _score_bipartite(params_init, beta=beta)

def _de_optimize_bipartite(deadline, beta=50.0):
    """Differential evolution on bipartite parameters."""
    
    def objective(params):
        if time.time() > deadline - 10.0:
            return 1e9
        score = _score_bipartite(params, beta=beta)
        if score < 0:
            return 1e9
        return -score
    
    bounds = [(0.3, 2.0), (0.3, 2.0), (0, 2 * np.pi / 10)]
    
    try:
        res = differential_evolution(objective, bounds=bounds,
                                      maxiter=200, popsize=20, seed=42,
                                      tol=1e-4, atol=1e-4, polish=False,
                                      updating='immediate', init='sobol')
        if res.fun < 1e8:
            return res.x, -res.fun
    except Exception:
        pass
    
    return None, -1.0

def _score_from_angles_radii_smooth(angles, radii, beta=50.0):
    """Compute smooth score given both angles and radii."""
    radii = np.maximum(radii, 0.1)
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    if not _is_valid_convex(polygon):
        return -1.0
    return _polygon_score_smooth(polygon, beta)

def _score_from_angles_radii(angles, radii):
    """Compute exact score given both angles and radii."""
    radii = np.maximum(radii, 0.1)
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    if not _is_valid_convex(polygon):
        return -1.0
    return _polygon_score_fast(polygon)

def _nm_optimize_joint(angles_init, radii_init, deadline, beta=50.0):
    """Nelder-Mead on joint (angles, radii) with smooth surrogate."""
    x0 = np.concatenate([angles_init, radii_init])
    n = len(angles_init)
    
    def objective(x):
        if time.time() > deadline - 10.0:
            return 1e9
        a = x[:n]
        r = x[n:]
        score = _score_from_angles_radii_smooth(a, r, beta)
        if score < 0:
            return 1e9
        return -score
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < 1e8:
            a_opt = res.x[:n]
            r_opt = res.x[n:]
            exact = _score_from_angles_radii(a_opt, r_opt)
            if exact > 0:
                return a_opt, r_opt, exact
    except Exception:
        pass
    
    return angles_init, radii_init, _score_from_angles_radii(angles_init, radii_init)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses bipartite parameterization + joint angle/radii optimization with smooth surrogate.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 20
    angles_regular = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    best_score = -1.0
    best_angles = angles_regular.copy()
    best_radii = np.ones(n)
    best_polygon = None
    
    def _update_best(polygon, score):
        nonlocal best_score, best_polygon
        if score > best_score and _is_valid_convex(polygon):
            best_score = score
            best_polygon = polygon.copy()
    
    # Phase 0: Bipartite radial search (reduced 3D parameter space)
    if time.time() < deadline - 400.0:
        bip_params, bip_score = _de_optimize_bipartite(deadline, beta=50.0)
        if bip_score > 0:
            R1, R2, offset = bip_params
            R1 = max(R1, 0.1)
            R2 = max(R2, 0.1)
            bip_poly = _bipartite_polygon(R1, R2, offset)
            _update_best(bip_poly, _polygon_score_fast(bip_poly))
        
        # NM refinement on bipartite params with smooth surrogate
        if time.time() < deadline - 300.0:
            if bip_params is not None:
                nm_bip_params, nm_bip_score = _nm_optimize_bipartite(bip_params, deadline, beta=50.0)
                if nm_bip_score > 0:
                    R1, R2, offset = nm_bip_params
                    R1 = max(R1, 0.1)
                    R2 = max(R2, 0.1)
                    bip_poly = _bipartite_polygon(R1, R2, offset)
                    _update_best(bip_poly, _polygon_score_fast(bip_poly))
            
            # Try a few initial points for bipartite NM
            rng = np.random.default_rng(42)
            for trial in range(3):
                if time.time() > deadline - 250.0:
                    break
                init_params = np.array([
                    0.8 + rng.uniform(-0.3, 0.3),
                    0.8 + rng.uniform(-0.3, 0.3),
                    rng.uniform(0, 2 * np.pi / 10)
                ])
                nm_p, nm_s = _nm_optimize_bipartite(init_params, deadline, beta=50.0)
                if nm_s > 0:
                    R1, R2, offset = nm_p
                    R1 = max(R1, 0.1)
                    R2 = max(R2, 0.1)
                    bip_poly = _bipartite_polygon(R1, R2, offset)
                    _update_best(bip_poly, _polygon_score_fast(bip_poly))
    
    # Phase 1: Evaluate diverse initial candidates
    candidates_angles = [angles_regular.copy()]
    rng = np.random.default_rng(123)
    for amp in [0.05, 0.1, 0.2]:
        for freq in [1, 2, 3]:
            pert = amp * np.cos(freq * angles_regular)
            candidates_angles.append(angles_regular + pert)
    for _ in range(10):
        coeffs = rng.normal(0, 0.12, 4)
        pert = np.zeros(n)
        for f in range(1, 5):
            pert += coeffs[f-1] * np.cos(f * angles_regular)
        candidates_angles.append(angles_regular + pert)
    
    for angles_cand in candidates_angles:
        if time.time() > deadline - 200.0:
            break
        score = _score_from_angles(angles_cand)
        if score > best_score:
            best_score = score
            best_angles = angles_cand.copy()
            best_radii = np.ones(n)
            p = np.stack([np.cos(best_angles), np.sin(best_angles)], axis=1)
            best_polygon = _normalize(p)
    
    # Phase 2: Joint angle+radii optimization with smooth surrogate
    if time.time() < deadline - 150.0:
        a_opt, r_opt, s_opt = _nm_optimize_joint(best_angles, best_radii, deadline, beta=50.0)
        if s_opt > best_score:
            best_score = s_opt
            best_angles = a_opt.copy()
            best_radii = r_opt.copy()
            p = np.stack([best_radii * np.cos(best_angles), best_radii * np.sin(best_angles)], axis=1)
            best_polygon = _normalize(p)
    
    # Phase 3: Local refinement on angles
    if time.time() < deadline - 80.0:
        local_angles, local_score = _local_refinement_angles(best_angles, 0.01, 25, deadline)
        if local_score > best_score:
            best_score = local_score
            best_angles = local_angles.copy()
            p = np.stack([best_radii * np.cos(best_angles), best_radii * np.sin(best_angles)], axis=1)
            best_polygon = _normalize(p)
    
    # Phase 4: Optimize radii for the best angles
    if time.time() < deadline - 60.0:
        radii_nm, radii_score = _nm_optimize_radii(best_radii, best_angles, deadline)
        if radii_score > best_score:
            best_score = radii_score
            best_radii = radii_nm.copy()
            p = np.stack([best_radii * np.cos(best_angles), best_radii * np.sin(best_angles)], axis=1)
            best_polygon = _normalize(p)
    
    # Phase 5: Joint refinement with perturb-and-refine strategy
    if time.time() < deadline - 40.0:
        rng2 = np.random.default_rng(7)
        for trial in range(5):
            if time.time() > deadline - 20.0:
                break
            # Small perturbation of best solution
            pert_angles = best_angles + rng2.normal(0, 0.02, n)
            pert_radii = best_radii + rng2.normal(0, 0.02, n)
            pert_radii = np.maximum(pert_radii, 0.1)
            
            a_opt, r_opt, s_opt = _nm_optimize_joint(pert_angles, pert_radii, deadline, beta=50.0)
            if s_opt > best_score:
                best_score = s_opt
                best_angles = a_opt.copy()
                best_radii = r_opt.copy()
                p = np.stack([best_radii * np.cos(best_angles), best_radii * np.sin(best_angles)], axis=1)
                best_polygon = _normalize(p)
    
    # Phase 6: Final local polish on radii
    if time.time() < deadline - 20.0:
        n_r = len(best_radii)
        best_r = best_radii.copy()
        best_s = best_score
        current_step = 0.005
        for step in range(20):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(n_r):
                if time.time() > deadline - 10.0:
                    break
                for direction in [1, -1]:
                    trial = best_r.copy()
                    trial[i] += direction * current_step
                    trial = np.maximum(trial, 0.1)
                    score = _score_from_radii(trial, best_angles)
                    if score > best_s:
                        best_r = trial
                        best_s = score
                        improved = True
            if not improved:
                current_step *= 0.5
                if current_step < 1e-4:
                    break
            else:
                current_step *= 0.8
        if best_s > best_score:
            best_score = best_s
            best_radii = best_r.copy()
            p = np.stack([best_radii * np.cos(best_angles), best_radii * np.sin(best_angles)], axis=1)
            best_polygon = _normalize(p)
    
    # Phase 7: High-precision final polish with exact objective
    if time.time() < deadline - 10.0:
        def obj_final(x):
            if time.time() > deadline - 5.0:
                return 1e9
            a = x[:n]
            r = x[n:]
            score = _score_from_angles_radii(a, r)
            if score < 0:
                return 1e9
            return -score
        
        x0 = np.concatenate([best_angles, best_radii])
        try:
            res = minimize(obj_final, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < 1e8:
                a_final = res.x[:n]
                r_final = res.x[n:]
                final_score = _score_from_angles_radii(a_final, r_final)
                if final_score > best_score:
                    best_score = final_score
                    best_angles = a_final.copy()
                    best_radii = r_final.copy()
                    p = np.stack([best_radii * np.cos(best_angles), best_radii * np.sin(best_angles)], axis=1)
                    best_polygon = _normalize(p)
        except Exception:
            pass
    
    # Ensure we have a valid polygon
    if best_polygon is None or not _is_valid_convex(best_polygon):
        p = np.stack([np.cos(angles_regular), np.sin(angles_regular)], axis=1)
        best_polygon = _normalize(p)
    
    return best_polygon

# EVOLVE-BLOCK-END
