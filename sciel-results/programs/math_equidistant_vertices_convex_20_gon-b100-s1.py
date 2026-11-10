import numpy as np
import time
from scipy.optimize import minimize

def compute_metric(points):
    """
    Computes the metric described in the task contract.
    points: (N, 2) array
    Returns: (metric, is_valid)
    """
    N = points.shape[0]
    if N != 20:
        return 0.0, False
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0, False
    
    normalized = centered / mean_dist
    
    # Check for repeated points
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, 0.0)
    min_pair_dist = np.min(dist_matrix[~np.eye(N, dtype=bool)])
    if min_pair_dist < 1e-6:
        return 0.0, False

    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0, False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False
    
    # Vectorized pairwise distances (reuse dist_matrix)
    np.fill_diagonal(dist_matrix, np.inf)
    
    min_score = np.inf
    for i in range(N):
        d = dist_matrix[i]
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
            score = 0.0
        else:
            diffs = np.diff(d_sorted)
            min_diff_idx = np.argmin(diffs)
            k = min_diff_idx
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            
            if D < 0.1:
                score = 0.0
            else:
                if k == 0:
                    selected = d_sorted[0:4]
                elif k == 17:
                    selected = d_sorted[15:19]
                else:
                    selected = d_sorted[k-1:k+3]
                
                ratios = np.maximum(selected / D, D / selected)
                raw = np.mean(ratios)
                score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        
        if score < min_score:
            min_score = score
            
    return min_score, True

def compute_metric_smooth(points, temperature=0.05):
    """
    Smooth surrogate for the metric using soft-min and soft-max.
    Returns a smooth scalar objective (higher is better).
    """
    N = points.shape[0]
    if N != 20:
        return -10.0
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return -10.0
    
    normalized = centered / mean_dist
    
    # Check convexity softly
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    # Soft penalty for non-convexity
    cross_abs_min = np.min(np.abs(cross))
    if cross_abs_min < 1e-10:
        return -10.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return -5.0 - 0.1 * np.sum(np.abs(cross[cross * np.sign(cross[0]) < 0]))
    
    # Check for repeated points softly
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    if min_pair_dist < 1e-6:
        return -10.0
    
    # Compute smooth vertex scores
    vertex_scores = np.zeros(N)
    for i in range(N):
        d = dist_matrix[i]
        d_sorted = np.sort(d)
        
        # Soft check for d_sorted[0] < 0.05
        if d_sorted[0] < 0.05:
            vertex_scores[i] = 0.0
            continue
        
        diffs = np.diff(d_sorted)
        # Soft argmin via softmax weights
        exp_diffs = np.exp(-diffs / temperature)
        softmax_weights = exp_diffs / np.sum(exp_diffs)
        
        # Weighted D using soft argmin
        D_values = (d_sorted[:-1] + d_sorted[1:]) / 2.0
        D = np.sum(softmax_weights * D_values)
        
        if D < 0.1:
            vertex_scores[i] = 0.0
            continue
        
        # Soft selection of 4 distances around soft k
        # Use weighted average of ratios over all possible 4-distance windows
        # Weighted by softmax of how close each window is to the soft k
        num_windows = 16  # k=0..15 maps to windows
        window_ratios = np.zeros(num_windows)
        for k_idx in range(num_windows):
            if k_idx == 0:
                selected = d_sorted[0:4]
            elif k_idx == 15:
                selected = d_sorted[15:19]
            else:
                selected = d_sorted[k_idx-1:k_idx+3]
            ratios = np.maximum(selected / D, D / selected)
            window_ratios[k_idx] = np.mean(ratios)
        
        # Weight windows by softmax of negative distance to soft k
        soft_k = np.sum(softmax_weights * np.arange(16))
        k_indices = np.arange(16)
        window_weights = np.exp(-(k_indices - soft_k)**2 / (2 * temperature * 10))
        window_weights /= np.sum(window_weights)
        
        raw = np.sum(window_weights * window_ratios)
        score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        vertex_scores[i] = score
    
    # Soft min of vertex scores
    if np.any(vertex_scores == 0):
        return -1.0
    
    # Use soft-min to approximate the minimum
    exp_scores = np.exp(vertex_scores / temperature)
    soft_min = temperature * np.log(np.sum(exp_scores)) - temperature * np.log(N)
    
    return soft_min

def equidistant_points_polygon20() -> np.ndarray:
    """
    Direct approach: Start with a regular 20-gon and use symmetry-preserving
    parameterization to optimize. The key insight is that for a convex 20-gon
    with equidistant-like properties, we can parameterize by radius and angular
    offsets from the regular polygon, then use gradient-based optimization.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_points(params):
        """
        params: [r_0, ..., r_19, theta_0, ..., theta_19]
        where r_i are radii and theta_i are angular offsets
        """
        r = np.abs(params[:n])
        theta = base_angles + params[n:]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    def points_to_params(points):
        """Convert points to parameterization (r, angular offsets)"""
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        r = np.linalg.norm(centered, axis=1)
        theta = np.arctan2(centered[:, 1], centered[:, 0])
        off = theta - base_angles
        off = np.unwrap(off)
        off = (off + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([r, off])
    
    def objective(params):
        """Objective to minimize: negative score + penalties"""
        if time.time() > deadline - safety_margin:
            return 1e10
        
        r = params[:n]
        r_abs = np.abs(r)
        
        # Penalty for small radii
        r_penalty = 0.0
        for ri in r_abs:
            if ri < 0.1:
                r_penalty += (0.1 - ri) ** 2
        
        points = get_points(params)
        score, valid = compute_metric(points)
        
        if not valid:
            return 100.0 + r_penalty + 0.01 * np.sum(r_abs ** 2)
        
        return -score + 0.001 * r_penalty
    
    def objective_smooth(params, temperature=0.05):
        """Smooth objective for gradient-based methods"""
        if time.time() > deadline - safety_margin:
            return 1e10
        
        r = params[:n]
        r_abs = np.abs(r)
        
        # Penalty for small radii
        r_penalty = 0.0
        for ri in r_abs:
            if ri < 0.1:
                r_penalty += (0.1 - ri) ** 2
        
        points = get_points(params)
        smooth_score = compute_metric_smooth(points, temperature)
        
        # Add penalty for invalid configurations
        _, valid = compute_metric(points)
        if not valid:
            return 100.0 + r_penalty + 0.01 * np.sum(r_abs ** 2)
        
        return -smooth_score + 0.001 * r_penalty
    
    # Initial candidates: regular polygon and perturbations
    best_score = -1.0
    best_params = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular polygon
    r_reg = np.ones(n)
    off_reg = np.zeros(n)
    candidates.append(np.concatenate([r_reg, off_reg]))
    
    # 2. Small perturbations of regular polygon
    for scale in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
        r_pert = 1.0 + scale * rng.normal(0, 1, n)
        off_pert = scale * 0.5 * rng.normal(0, 1, n)
        candidates.append(np.concatenate([np.abs(r_pert), off_pert]))
    
    # 3. Harmonic modes
    for freq in [1, 2, 3, 4, 5]:
        for amp in [0.05, 0.1, 0.15, 0.2]:
            for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
                r_h = 1.0 + amp * np.sin(freq * base_angles + phase)
                off_h = amp * 0.3 * np.cos(freq * base_angles + phase)
                candidates.append(np.concatenate([np.abs(r_h), off_h]))
    
    # 4. Alternating patterns
    for amp in [0.05, 0.1, 0.15]:
        for period in [2, 4, 5]:
            r_alt = 1.0 + amp * np.sin(2 * np.pi * base_angles * period / (2 * np.pi))
            off_alt = amp * 0.5 * np.cos(2 * np.pi * base_angles * period / (2 * np.pi))
            candidates.append(np.concatenate([np.abs(r_alt), off_alt]))
    
    # 5. Random structured
    for _ in range(30):
        r_rand = 1.0 + 0.2 * rng.normal(0, 1, n)
        off_rand = 0.1 * rng.normal(0, 1, n)
        candidates.append(np.concatenate([np.abs(r_rand), off_rand]))
    
    # 6. Slightly non-uniform angular spacing
    for _ in range(20):
        r_uniform = np.ones(n)
        # Perturb angles slightly
        angle_pert = 0.05 * rng.normal(0, 1, n)
        angle_pert -= np.mean(angle_pert)  # Keep total angle = 2*pi
        off_custom = angle_pert
        candidates.append(np.concatenate([r_uniform, off_custom]))
    
    # Evaluate all candidates
    scored_candidates = []
    for i, params in enumerate(candidates):
        if time.time() > deadline - 180:
            break
        
        points = get_points(params)
        score, valid = compute_metric(points)
        if valid:
            scored_candidates.append((score, params, points))
    
    # Sort by score
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Optimize top candidates with multiple strategies
    top_k = min(20, len(scored_candidates))
    
    for score, params, points in scored_candidates[:top_k]:
        if time.time() > deadline - 90:
            break
        
        # Strategy 1: Nelder-Mead on exact objective
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'adaptive': True, 'xatol': 1e-9, 'fatol': 1e-9})
            r_final = np.abs(res.x[:n])
            off_final = res.x[n:]
            params_final = np.concatenate([r_final, off_final])
            points_final = get_points(params_final)
            score_final, valid_final = compute_metric(points_final)
            
            if valid_final and score_final > best_score:
                best_score = score_final
                best_params = params_final.copy()
        except Exception:
            pass
        
        # Strategy 2: L-BFGS-B on smooth surrogate
        if time.time() > deadline - 45:
            break
        try:
            res_smooth = minimize(objective_smooth, params, method='L-BFGS-B',
                                  options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            r_smooth = np.abs(res_smooth.x[:n])
            off_smooth = res_smooth.x[n:]
            params_smooth = np.concatenate([r_smooth, off_smooth])
            points_smooth = get_points(params_smooth)
            score_smooth, valid_smooth = compute_metric(points_smooth)
            
            if valid_smooth and score_smooth > best_score:
                best_score = score_smooth
                best_params = params_smooth.copy()
        except Exception:
            pass
        
        # Strategy 3: Powell on exact objective
        if time.time() > deadline - 30:
            break
        try:
            res_powell = minimize(objective, params, method='Powell',
                                  options={'maxiter': 500, 'xtol': 1e-9, 'ftol': 1e-9})
            r_powell = np.abs(res_powell.x[:n])
            off_powell = res_powell.x[n:]
            params_powell = np.concatenate([r_powell, off_powell])
            points_powell = get_points(params_powell)
            score_powell, valid_powell = compute_metric(points_powell)
            
            if valid_powell and score_powell > best_score:
                best_score = score_powell
                best_params = params_powell.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 10 and best_params is not None:
        # Polish with Nelder-Mead at very tight tolerance
        try:
            res_polish = minimize(objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-12})
            r_p = np.abs(res_polish.x[:n])
            off_p = res_polish.x[n:]
            params_p = np.concatenate([r_p, off_p])
            points_p = get_points(params_p)
            score_p, valid_p = compute_metric(points_p)
            
            if valid_p and score_p > best_score:
                best_score = score_p
                best_params = params_p.copy()
        except Exception:
            pass
        
        # Polish with L-BFGS-B on smooth surrogate at tight tolerance
        try:
            res_polish2 = minimize(objective_smooth, best_params, method='L-BFGS-B',
                                   options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            r_p2 = np.abs(res_polish2.x[:n])
            off_p2 = res_polish2.x[n:]
            params_p2 = np.concatenate([r_p2, off_p2])
            points_p2 = get_points(params_p2)
            score_p2, valid_p2 = compute_metric(points_p2)
            
            if valid_p2 and score_p2 > best_score:
                best_score = score_p2
                best_params = params_p2.copy()
        except Exception:
            pass
        
        # Additional polish: coordinate descent on the bottleneck vertex
        if time.time() < deadline - 5:
            points_cur = get_points(best_params)
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                # Find the bottleneck vertex
                centroid = np.mean(points_cur, axis=0)
                centered = points_cur - centroid
                mean_dist = np.mean(np.linalg.norm(centered, axis=1))
                normalized = centered / mean_dist
                dist_matrix = np.linalg.norm(normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :], axis=2)
                np.fill_diagonal(dist_matrix, np.inf)
                
                vertex_scores = []
                for i in range(n):
                    d = dist_matrix[i]
                    d_sorted = np.sort(d)
                    if d_sorted[0] < 0.05:
                        vertex_scores.append(0.0)
                        continue
                    diffs = np.diff(d_sorted)
                    k = np.argmin(diffs)
                    D = (d_sorted[k] + d_sorted[k+1]) / 2.0
                    if D < 0.1:
                        vertex_scores.append(0.0)
                        continue
                    if k == 0:
                        selected = d_sorted[0:4]
                    elif k == 17:
                        selected = d_sorted[15:19]
                    else:
                        selected = d_sorted[k-1:k+3]
                    ratios = np.maximum(selected / D, D / selected)
                    raw = np.mean(ratios)
                    vertex_scores.append(min(1.0, 1.0/raw) if raw > 0 else 0.0)
                
                bottleneck = np.argmin(vertex_scores)
                if vertex_scores[bottleneck] >= 0.999:
                    break
                
                # Try small perturbations of the bottleneck vertex
                best_local = best_score
                for dr in [-0.01, -0.005, 0.005, 0.01]:
                    for dtheta in [-0.01, -0.005, 0.005, 0.01]:
                        params_test = best_params.copy()
                        params_test[bottleneck] += dr
                        params_test[n + bottleneck] += dtheta
                        points_test = get_points(params_test)
                        score_test, valid_test = compute_metric(points_test)
                        if valid_test and score_test > best_local:
                            best_local = score_test
                            best_params = params_test.copy()
                            points_cur = points_test
                
                if best_local > best_score:
                    best_score = best_local
    
    # Return best points
    if best_params is None:
        # Fallback to regular polygon
        return np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    return get_points(best_params)
