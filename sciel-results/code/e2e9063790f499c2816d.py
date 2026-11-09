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
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(normalized[i] - normalized[j])
            if d < min_pair_dist:
                min_pair_dist = d
    if min_pair_dist < 1e-6:
        return 0.0, False

    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-10):
        return 0.0, False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False
    
    # Vectorized pairwise distances
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
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

def _score_polygon_vectorized(polygon):
    """
    Fast vectorized score calculation for optimization.
    Returns the minimum vertex score, or -inf if invalid.
    """
    N = polygon.shape[0]
    if N != 20:
        return -np.inf
    
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return -np.inf
    
    normalized = centered / mean_dist
    
    # Check strictly convex (vectorized)
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-10):
        return -np.inf
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return -np.inf
    
    # Check no repeated points (vectorized min pair distance)
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    
    if np.min(dist_matrix) < 1e-6:
        return -np.inf
    
    # Sort distances for each vertex
    sorted_dists = np.sort(dist_matrix, axis=1)
    
    # Check d_0 < 0.05
    min_dists = sorted_dists[:, 0]
    invalid_mask = min_dists < 0.05
    
    # Calculate gaps and find k
    gaps = np.diff(sorted_dists, axis=1)
    k_indices = np.argmin(gaps, axis=1)
    
    # Determine D
    d_k = sorted_dists[np.arange(N), k_indices]
    d_k_plus_1 = sorted_dists[np.arange(N), k_indices + 1]
    D = (d_k + d_k_plus_1) / 2.0
    
    invalid_mask |= (D < 0.1)
    
    # Determine which 4 distances to use for each vertex
    indices = np.zeros((N, 4), dtype=int)
    
    mask_0 = (k_indices == 0)
    indices[mask_0, 0] = 0
    indices[mask_0, 1] = 1
    indices[mask_0, 2] = 2
    indices[mask_0, 3] = 3
    
    mask_17 = (k_indices == 17)
    indices[mask_17, 0] = 15
    indices[mask_17, 1] = 16
    indices[mask_17, 2] = 17
    indices[mask_17, 3] = 18
    
    mask_else = ~(mask_0 | mask_17)
    k_else = k_indices[mask_else]
    indices[mask_else, 0] = k_else - 1
    indices[mask_else, 1] = k_else
    indices[mask_else, 2] = k_else + 1
    indices[mask_else, 3] = k_else + 2
    
    row_indices = np.arange(N)[:, None]
    selected_dists = sorted_dists[row_indices, indices]
    
    # Compute raw and scores
    ratios = selected_dists / D[:, None]
    raw_components = np.where(ratios >= 1.0, ratios, 1.0 / ratios)
    raw = np.mean(raw_components, axis=1)
    
    scores = np.minimum(1.0, 1.0 / raw)
    scores[invalid_mask] = -np.inf
    
    return np.min(scores)

def _soft_min(values, temperature=0.05):
    """
    Soft-min approximation using log-sum-exp.
    Returns a smooth lower bound on the true min.
    """
    values = np.asarray(values)
    if np.any(np.isneginf(values)):
        return -np.inf
    # Clip to avoid numerical issues
    values = np.maximum(values, 1e-10)
    min_val = np.min(values)
    # soft min = -temperature * log(sum(exp(-values/temperature)))
    # But we want it to be close to min, so use:
    # soft_min ≈ min - temperature * log(N) approximately
    # More precisely: -T * log(sum(exp(-x_i/T)))
    scaled = -values / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -temperature * log_sum

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_points(r, off):
        r = np.abs(r)
        theta = angles + off
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    # Objective with soft-min surrogate for smoother optimization
    def objective(params, use_soft=False, temperature=0.05):
        if time.time() > deadline:
            return 1.0
        
        r = params[:n]
        off = params[n:]
        
        r_abs = np.abs(r)
        if np.any(r_abs < 0.1):
            return 1.0 + np.sum(r_abs**2)
            
        points = get_points(r_abs, off)
        
        if use_soft:
            # Use soft-min for smoother landscape
            score = _score_polygon_vectorized(points)
            if np.isneginf(score):
                return 1.0 + 0.1 * np.sum(r_abs**2)
            # We want to maximize the soft-min of vertex scores
            # But _score_polygon_vectorized already returns min
            # For soft-min, we need per-vertex scores
            # Let's just use the exact score but add a small penalty for low scores
            return -score
        else:
            score = _score_polygon_vectorized(points)
            if np.isneginf(score):
                return 1.0 + 0.1 * np.sum(r_abs**2)
            return -score
    
    best_score = -np.inf
    best_points = None
    best_params = None
    
    rng = np.random.default_rng(42)
    
    # Initializations
    initializations = []
    
    # 1. Regular polygon
    initializations.append((np.ones(n), np.zeros(n)))
    
    # 2. Various noise scales
    for noise_scale in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]:
        init_r = 1.0 + rng.normal(0, noise_scale, n)
        init_off = rng.normal(0, noise_scale * 0.5, n)
        initializations.append((init_r, init_off))
        
    # 3. Wave patterns
    for freq in [2, 3, 4, 5]:
        for amp in [0.05, 0.1, 0.15, 0.2]:
            init_r = 1.0 + amp * np.sin(freq * angles)
            init_off = amp * 0.3 * np.cos(freq * angles)
            initializations.append((init_r, init_off))
    
    # 4. Alternating patterns
    for amp in [0.05, 0.1, 0.15]:
        pattern = np.array([1.0 + amp if i % 2 == 0 else 1.0 - amp for i in range(n)])
        init_off = np.zeros(n)
        initializations.append((pattern, init_off))
    
    # 5. Random structured perturbations
    for _ in range(5):
        init_r = 1.0 + 0.1 * rng.random(n)
        init_off = 0.05 * rng.random(n)
        initializations.append((init_r, init_off))
    
    n_initializations = len(initializations)
    
    # Phase 1: Quick exploration
    per_restart_time = 12.0
    max_restarts = int((deadline - start_time - 60) / per_restart_time)
    
    for i in range(min(n_initializations, max_restarts)):
        if time.time() > deadline - 60:
            break
            
        init_r, init_off = initializations[i]
        init_params = np.concatenate([init_r, init_off])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 800, 'adaptive': True, 'xatol': 1e-5, 'fatol': 1e-5})
            
            final_r = np.abs(res.x[:n])
            final_off = res.x[n:]
            final_points = get_points(final_r, final_off)
            final_score, valid = compute_metric(final_points)
            
            if valid and final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
                best_params = res.x.copy()
                
        except Exception:
            pass
    
    # Phase 2: Focused polishing from best found with L-BFGS-B on coordinates
    if best_params is not None and time.time() < deadline - 40:
        try:
            # First polish with Nelder-Mead
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8})
            
            final_r2 = np.abs(res2.x[:n])
            final_off2 = res2.x[n:]
            final_points2 = get_points(final_r2, final_off2)
            final_score2, valid2 = compute_metric(final_points2)
            
            if valid2 and final_score2 > best_score:
                best_score = final_score2
                best_points = final_points2.copy()
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate ascent polishing on the best found parameters
    if best_params is not None and time.time() < deadline - 20:
        current_params = best_params.copy()
        current_score = best_score
        step_size = 0.02
        no_improve_count = 0
        
        for iteration in range(5000):
            if time.time() > deadline - 10.0:
                break
            if no_improve_count > 100:
                break
                
            improved = False
            for i in range(2 * n):
                if time.time() > deadline - 10.0:
                    break
                for delta in [step_size, -step_size, step_size/2, -step_size/2]:
                    trial_params = current_params.copy()
                    trial_params[i] += delta
                    # Ensure radii are positive
                    if i >= n:
                        trial_params[i] = max(trial_params[i], 0.1)
                    
                    trial_points = get_points(np.abs(trial_params[:n]), trial_params[n:])
                    trial_score, valid = compute_metric(trial_points)
                    
                    if valid and trial_score > current_score + 1e-10:
                        current_params = trial_params
                        current_score = trial_score
                        if current_score > best_score:
                            best_score = current_score
                            best_points = trial_points.copy()
                        improved = True
                        break
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Reduce step size if stuck
                step_size *= 0.95
                if step_size < 1e-6:
                    break
    
    # Phase 4: Final high-precision polish with tighter tolerances
    if best_params is not None and time.time() < deadline - 5:
        try:
            res3 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-10})
            
            final_r3 = np.abs(res3.x[:n])
            final_off3 = res3.x[n:]
            final_points3 = get_points(final_r3, final_off3)
            final_score3, valid3 = compute_metric(final_points3)
            
            if valid3 and final_score3 > best_score:
                best_score = final_score3
                best_points = final_points3.copy()
        except Exception:
            pass
    
    if best_points is None:
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
