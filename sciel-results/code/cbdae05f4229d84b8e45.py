import numpy as np
import time
from scipy.optimize import minimize, basinhopping

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
    # Vectorized min pair distance is expensive for small N, loop is fine
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
    if np.any(np.abs(cross) < 1e-12):
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

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    n = 20
    # Initial angles for regular polygon
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def params_to_points(params):
        r = np.abs(params[:n])
        theta = params[n:]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
        
    def points_to_params(points):
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        r = np.linalg.norm(centered, axis=1)
        theta = np.arctan2(centered[:, 1], centered[:, 0])
        # Unwrap theta to ensure continuity?
        # For Nelder-Mead, it's better to keep theta in a continuous range
        # But arctan2 returns [-pi, pi].
        # We can just use the raw values.
        return np.concatenate([r, theta])
        
    def objective(params):
        if time.time() > deadline:
            return 1.0
        r = np.abs(params[:n])
        theta = params[n:]
        
        # Penalize small radii
        if np.any(r < 0.1):
            return 1.0 + np.sum((r - 0.1)**2)
            
        points = params_to_points(params)
        score, valid = compute_metric(points)
        if not valid:
            # Calculate convexity penalty
            edges = np.roll(points, -1, axis=0) - points
            cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
            # We want all cross products to have the same sign.
            # If they are mixed, penalize the ones with the "wrong" sign relative to the majority?
            # Simpler: penalize any cross product that is close to 0 or has the opposite sign to the first one?
            # Actually, just penalize the magnitude of cross products that violate convexity.
            # If cross > 0 for all, it's CCW convex. If cross < 0 for all, CW convex.
            # If mixed, it's non-convex.
            # We can penalize the sum of squares of cross products that are "negative" if we assume CCW, or vice versa.
            # Let's just use a large penalty if invalid.
            # To guide the optimizer, let's look at the cross products.
            # If the mean cross product is positive, penalize negative ones.
            mean_cross = np.mean(cross)
            if mean_cross > 0:
                penalty = np.sum(np.minimum(cross, 0)**2)
            else:
                penalty = np.sum(np.maximum(cross, 0)**2)
            return 1.0 + penalty
        
        return -score

    # Initial guess: Regular polygon
    init_points = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    init_params = points_to_params(init_points)
    
    best_score = -1.0
    best_points = init_points.copy()
    
    # Local optimization function
    def local_optimize(x0):
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            return res.x, res.fun
        except Exception:
            return x0, objective(x0)
    
    # Initial local optimization
    if time.time() < deadline - 50:
        opt_params, opt_val = local_optimize(init_params)
        opt_points = params_to_points(opt_params)
        score, valid = compute_metric(opt_points)
        if valid and score > best_score:
            best_score = score
            best_points = opt_points.copy()
            best_params = opt_params.copy()
        else:
            best_params = init_params.copy()

    # Basin Hopping
    if time.time() < deadline - 20:
        def step(x):
            # Random perturbation
            dx = np.random.normal(0, 0.1, len(x))
            return x + dx
            
        # We need to track the best state
        # basinhopping returns the best state found
        # We need to wrap it to check time
        
        # Since basinhopping doesn't have a built-in timeout, we'll run it in a loop
        # or just use it with a limited number of iterations.
        
        # Let's try a manual basin hopping loop for better time control
        current_params = best_params.copy()
        current_score = best_score
        
        for i in range(50): # Max 50 basin hops
            if time.time() > deadline - 10:
                break
                
            # Perturb
            new_params = current_params + np.random.normal(0, 0.2, len(current_params))
            
            # Optimize
            opt_params, opt_val = local_optimize(new_params)
            opt_points = params_to_points(opt_params)
            score, valid = compute_metric(opt_points)
            
            if valid:
                if score > current_score:
                    current_score = score
                    current_params = opt_params.copy()
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
                        best_params = opt_params.copy()
                else:
                    # Accept with probability exp(-delta/T) ?
                    # Simulated Annealing style
                    # T should decrease
                    T = 0.1 * (0.9 ** i)
                    delta = score - current_score
                    if delta < 0:
                        if np.random.rand() < np.exp(delta / T):
                            current_params = opt_params.copy()
                            current_score = score
                    else:
                        current_params = opt_params.copy()
                        current_score = score
                        if score > best_score:
                            best_score = score
                            best_points = opt_points.copy()
                            best_params = opt_params.copy()
            else:
                # If invalid, reject
                pass

    # Final polish
    if time.time() < deadline - 5:
        opt_params, opt_val = local_optimize(best_params)
        opt_points = params_to_points(opt_params)
        score, valid = compute_metric(opt_points)
        if valid and score > best_score:
            best_score = score
            best_points = opt_points.copy()

    return best_points

