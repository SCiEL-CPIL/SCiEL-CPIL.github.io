import numpy as np
import time
from scipy.optimize import minimize

def _compute_all_scores(polygon):
    """Compute per-vertex scores for all vertices of polygon."""
    n = len(polygon)
    scores = np.zeros(n)
    diff = polygon[:, np.newaxis, :] - polygon[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    for i in range(n):
        d = np.sort(dists[i])
        if d[0] < 0.05:
            scores[i] = 0.0
            continue
        gaps = np.diff(d)
        k = int(np.argmin(gaps))
        D = (d[k] + d[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            selected = d[0:4]
        elif k == 17:
            selected = d[15:19]
        else:
            selected = d[k-1:k+3]
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
    return scores

def _polygon_min_score(polygon):
    """Compute the minimum vertex score over all vertices."""
    return float(np.min(_compute_all_scores(polygon)))

def _normalize(polygon):
    """Normalize polygon: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(polygon, axis=0)
    poly = polygon - centroid
    dists = np.sqrt(np.sum(poly**2, axis=1))
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return poly
    return poly / mean_dist

def _is_strictly_convex(polygon):
    """Check if polygon is strictly convex with correct total turning."""
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    cross_prods = []
    for i in range(n):
        v1 = polygon[(i+1) % n] - polygon[i]
        v2 = polygon[(i+2) % n] - polygon[(i+1) % n]
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    if np.any(np.abs(cross_prods) < 1e-12):
        return False
    if np.any(cross_prods > 0) and np.any(cross_prods < 0):
        return False
    edge_angles = []
    for i in range(n):
        v = polygon[(i+1) % n] - polygon[i]
        edge_angles.append(np.arctan2(v[1], v[0]))
    turns = []
    for i in range(n):
        a1 = edge_angles[i]
        a2 = edge_angles[(i+1) % n]
        turn = a2 - a1
        while turn > np.pi:
            turn -= 2*np.pi
        while turn <= -np.pi:
            turn += 2*np.pi
        turns.append(turn)
    total_turn = sum(turns)
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return False
    return True

def _softmin(scores, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    m = np.min(scores)
    if m < 0:
        return m
    # Numerically stable log-sum-exp
    shifted = scores - m
    return m - np.log(np.sum(np.exp(-beta * shifted))) / beta

def _build_polygon_from_angles(theta_pert, base_angles, radii=None):
    """
    Build polygon from angular perturbations.
    If radii is None, assumes unit circle (regular polygon shape).
    """
    n = len(theta_pert)
    thetas = base_angles + theta_pert
    if radii is None:
        radii = np.ones(n)
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    return np.stack([x, y], axis=1)

def _softmin_objective_angles(theta_pert, base_angles, beta=50.0):
    """
    Objective for L-BFGS-B: Negative soft-min score using angular perturbations.
    """
    n = len(theta_pert)
    polygon = _build_polygon_from_angles(theta_pert, base_angles)
    polygon = _normalize(polygon)
    
    if not _is_strictly_convex(polygon):
        return 10.0
    
    scores = _compute_all_scores(polygon)
    if np.any(scores < 0):
        return 10.0
        
    return -_softmin(scores, beta)

def _hard_objective_angles(theta_pert, base_angles):
    """
    Objective for Nelder-Mead: Negative hard min score using angular perturbations.
    """
    n = len(theta_pert)
    polygon = _build_polygon_from_angles(theta_pert, base_angles)
    polygon = _normalize(polygon)
    
    if not _is_strictly_convex(polygon):
        return 10.0
    
    return -_polygon_min_score(polygon)

def _build_polygon_from_angles_radii(theta_pert, radii_pert, base_angles):
    """
    Build polygon from angular and radial perturbations.
    """
    n = len(theta_pert)
    thetas = base_angles + theta_pert
    radii = 1.0 + radii_pert
    radii = np.maximum(radii, 0.1) # Ensure positive radii
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    return np.stack([x, y], axis=1)

def _hard_objective_angles_radii(params, base_angles):
    """
    Objective for Nelder-Mead on combined angle and radius perturbations.
    params = [theta_0...theta_19, r_0...r_19]
    """
    n = 20
    theta_pert = params[:n]
    radii_pert = params[n:]
    
    polygon = _build_polygon_from_angles_radii(theta_pert, radii_pert, base_angles)
    polygon = _normalize(polygon)
    
    if not _is_strictly_convex(polygon):
        return 10.0
    
    return -_polygon_min_score(polygon)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon maximizing the minimum per-vertex equidistance score.
    Combines angular perturbation search (inspiration) with multi-phase refinement (current).
    """
    start_time = time.time()
    deadline = start_time + 580.0
    n = 20
    
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Regular polygon baseline
    best_polygon = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1)
    best_polygon = _normalize(best_polygon)
    best_score = _polygon_min_score(best_polygon)
    
    np.random.seed(42)
    
    def time_left():
        return deadline - time.time()
        
    # Phase 1: L-BFGS-B on Angular Perturbations with Soft-Min Surrogate
    # This explores the shape of the polygon on the unit circle
    initial_theta = np.zeros(n)
    
    try:
        if time_left() > 150:
            result = minimize(
                _softmin_objective_angles, initial_theta.copy(), 
                args=(base_angles, 30.0), 
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            # Refine with higher beta (sharper min approximation)
            if time_left() > 100:
                result2 = minimize(
                    _softmin_objective_angles, result.x.copy(), 
                    args=(base_angles, 100.0), 
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                poly = _normalize(_build_polygon_from_angles(result2.x, base_angles))
                if _is_strictly_convex(poly):
                    s = _polygon_min_score(poly)
                    if s > best_score:
                        best_score = s
                        best_polygon = poly
    except Exception:
        pass

    # Phase 2: Multi-restart L-BFGS-B on Angular Perturbations
    # Try different random starting points to escape local minima
    for restart in range(10):
        if time_left() < 120:
            break
        
        scale = 0.2 / (1 + restart * 0.3)
        theta_init = np.random.uniform(-scale, scale, n)
        
        try:
            result = minimize(
                _softmin_objective_angles, theta_init, 
                args=(base_angles, 50.0), 
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if time_left() < 80:
                break
                
            # Quick polish with higher beta
            result2 = minimize(
                _softmin_objective_angles, result.x.copy(), 
                args=(base_angles, 150.0), 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            poly = _normalize(_build_polygon_from_angles(result2.x, base_angles))
            if _is_strictly_convex(poly):
                s = _polygon_min_score(poly)
                if s > best_score:
                    best_score = s
                    best_polygon = poly
        except Exception:
            pass

    # Phase 3: Nelder-Mead on Angular Perturbations (Exact Metric)
    # Refine the best angular configuration found so far
    if time_left() > 80:
        # Extract current best angles from best_polygon
        centroid = np.mean(best_polygon, axis=0)
        bp_centered = best_polygon - centroid
        current_thetas = np.arctan2(bp_centered[:, 1], bp_centered[:, 0])
        
        # Calculate perturbations relative to base angles
        theta_pert = current_thetas - base_angles
        # Unwrap to ensure continuity
        for i in range(1, n):
            diff = theta_pert[i] - theta_pert[i-1]
            if diff > np.pi:
                theta_pert[i] -= 2*np.pi
            elif diff < -np.pi:
                theta_pert[i] += 2*np.pi
        
        try:
            result = minimize(
                _hard_objective_angles, theta_pert.copy(),
                args=(base_angles,),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            poly = _normalize(_build_polygon_from_angles(result.x, base_angles))
            if _is_strictly_convex(poly):
                s = _polygon_min_score(poly)
                if s > best_score:
                    best_score = s
                    best_polygon = poly
        except Exception:
            pass

    # Phase 4: Nelder-Mead on Combined Angular and Radial Perturbations
    # Allow slight radial variation to fine-tune distances
    if time_left() > 60:
        # Extract angles and radii from current best
        centroid = np.mean(best_polygon, axis=0)
        bp_centered = best_polygon - centroid
        current_thetas = np.arctan2(bp_centered[:, 1], bp_centered[:, 0])
        current_radii = np.sqrt(np.sum(bp_centered**2, axis=1))
        
        theta_pert = current_thetas - base_angles
        for i in range(1, n):
            diff = theta_pert[i] - theta_pert[i-1]
            if diff > np.pi:
                theta_pert[i] -= 2*np.pi
            elif diff < -np.pi:
                theta_pert[i] += 2*np.pi
                
        radii_pert = current_radii - 1.0
        
        initial_params = np.concatenate([theta_pert, radii_pert])
        
        try:
            result = minimize(
                _hard_objective_angles_radii, initial_params.copy(),
                args=(base_angles,),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            poly = _normalize(_build_polygon_from_angles_radii(result.x[:n], result.x[n:], base_angles))
            if _is_strictly_convex(poly):
                s = _polygon_min_score(poly)
                if s > best_score:
                    best_score = s
                    best_polygon = poly
        except Exception:
            pass

    # Phase 5: Coordinate Descent on Angles (Fine-tuning)
    if time_left() > 30:
        centroid = np.mean(best_polygon, axis=0)
        bp_centered = best_polygon - centroid
        current_thetas = np.arctan2(bp_centered[:, 1], bp_centered[:, 0])
        theta_pert = current_thetas - base_angles
        for i in range(1, n):
            diff = theta_pert[i] - theta_pert[i-1]
            if diff > np.pi:
                theta_pert[i] -= 2*np.pi
            elif diff < -np.pi:
                theta_pert[i] += 2*np.pi
                
        improved = True
        step_size = 0.01
        while improved and time_left() > 10:
            improved = False
            for i in range(n):
                if time_left() < 10:
                    break
                base_val = theta_pert[i]
                base_score = _hard_objective_angles(theta_pert, base_angles)
                
                for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                    trial = theta_pert.copy()
                    trial[i] = base_val + delta
                    trial_score = _hard_objective_angles(trial, base_angles)
                    if trial_score < base_score:
                        base_score = trial_score
                        theta_pert[i] = trial[i]
                        improved = True
                        
        if improved:
            poly = _normalize(_build_polygon_from_angles(theta_pert, base_angles))
            if _is_strictly_convex(poly):
                s = _polygon_min_score(poly)
                if s > best_score:
                    best_score = s
                    best_polygon = poly

    # Final verification
    final_score = _polygon_min_score(best_polygon)
    if final_score <= 0 or not _is_strictly_convex(best_polygon):
        best_polygon = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1)
        best_polygon = _normalize(best_polygon)
    
    return best_polygon
