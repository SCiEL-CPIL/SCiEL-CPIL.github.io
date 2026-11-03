import numpy as np
import time
import math
from scipy.optimize import minimize

N = 300

def compute_score(points):
    """
    Compute the exact packing radius.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair = np.sqrt(np.min(dists_sq))
    
    return min(min_pair, min_bound)

def soft_min_value_and_grad(points, k=50.0):
    """
    Compute the value and gradient of the soft-min surrogate objective.
    Objective to maximize: S = -1/k * log(sum(exp(-k * d_j)))
    We minimize -S, so we return -S and -grad(S).
    """
    # Calculate all distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Pairwise distances (upper triangle)
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    pair_dists = dists[i_idx, j_idx]
    
    # Boundary distances
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min calculation
    # To avoid overflow/underflow, shift by min
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    exp_vals = np.exp(-k * shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return min_val, np.zeros_like(points)
        
    weights = exp_vals / sum_exp
    soft_min_val = min_val - np.log(sum_exp) / k
    
    # Gradient of soft_min_val w.r.t points
    # d(soft_min)/dp = sum(weights * d(dist)/dp)
    grads = np.zeros_like(points)
    
    # Pairwise part
    pair_weights = weights[:len(pair_dists)]
    # d_{ij} = ||p_i - p_j||
    # grad_{p_i} d_{ij} = (p_i - p_j) / d_{ij}
    # grad_{p_j} d_{ij} = (p_j - p_i) / d_{ij}
    
    # Vectorized update
    # v_ij = (p_i - p_j) / d_ij
    d_pair = pair_dists
    mask_nonzero = d_pair > 1e-9
    if np.any(mask_nonzero):
        i_m = i_idx[mask_nonzero]
        j_m = j_idx[mask_nonzero]
        w_m = pair_weights[mask_nonzero]
        
        v_ij = (points[i_m] - points[j_m]) / d_pair[mask_nonzero, np.newaxis]
        
        # grad_i += w * v_ij
        # grad_j -= w * v_ij
        np.add.at(grads, i_m, v_ij * w_m[:, np.newaxis])
        np.add.at(grads, j_m, -v_ij * w_m[:, np.newaxis])
        
    # Boundary part
    bound_weights = weights[len(pair_dists):]
    rad_safe = np.where(rad > 1e-9, rad, 1.0)
    # grad_{p_i} (1 - ||p_i||) = - p_i / ||p_i||
    bound_grads = -(points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
    grads += bound_grads
    
    # We want to MINIMIZE -soft_min_val
    # So objective = -soft_min_val
    # Gradient of objective = -grads
    return -soft_min_val, -grads

def polish_lbfgsb(points, time_budget):
    """
    Polish the points using L-BFGS-B on the soft-min surrogate.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = compute_score(points)
    
    k = 100.0 # Sharpness of soft min
    
    def objective_and_grad(x):
        pts = x.reshape(N, 2)
        val, grad = soft_min_value_and_grad(pts, k)
        return val, grad.reshape(-1)
        
    # Run minimize
    # We use a loop to potentially restart or adjust k if time permits, 
    # but usually one good run with L-BFGS-B is sufficient for polishing.
    
    res = minimize(objective_and_grad, points.reshape(-1), jac=True, method='L-BFGS-B')
    
    if res.success or res.fun < 0: # res.fun is -soft_min, so lower is better
        new_points = res.x.reshape(N, 2)
        # Project inside circle just in case
        rad = np.linalg.norm(new_points, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            new_points[mask] /= rad[mask, np.newaxis]
            
        new_score = compute_score(new_points)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            
    # If time remains, try a second polish with higher k
    if time.time() - start_time < time_budget - 1.0:
        k = 500.0
        def objective_and_grad_high(x):
            pts = x.reshape(N, 2)
            val, grad = soft_min_value_and_grad(pts, k)
            return val, grad.reshape(-1)
            
        res2 = minimize(objective_and_grad_high, best_points.reshape(-1), jac=True, method='L-BFGS-B')
        if res2.success or res2.fun < 0:
            new_points = res2.x.reshape(N, 2)
            rad = np.linalg.norm(new_points, axis=1)
            mask = rad > 1.0
            if np.any(mask):
                new_points[mask] /= rad[mask, np.newaxis]
                
            new_score = compute_score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
                
    return best_points, best_score

def simulated_annealing(points, time_budget):
    """
    Simulated Annealing to escape local minima.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.01
    cooling_rate = 0.999
    step_size = 0.01
    
    while time.time() - start_time < time_budget:
        if T < 1e-6:
            break
            
        # Pick random point
        idx = np.random.randint(N)
        
        # Generate move
        delta = np.random.normal(0, step_size, 2)
        new_points = current_points.copy()
        new_points[idx] += delta
        
        # Check boundary
        rad = np.linalg.norm(new_points[idx])
        if rad > 1.0:
            new_points[idx] /= rad
        
        # Evaluate new score
        new_score = compute_score(new_points)
        
        # Acceptance criterion
        delta_score = new_score - current_score
        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                
        T *= cooling_rate
        
        # Adapt step size
        # If we are improving a lot, increase step size slightly?
        # If stuck, decrease?
        # Simple: keep constant or decay slowly.
        step_size *= 0.9999
        
    return best_points, best_score

def init_hex():
    """
    Initialize with hexagonal grid points.
    """
    best_points = None
    best_score = -np.inf
    
    # Try a few scales
    for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
        # Approximate spacing for N points in unit circle
        # Area = pi, N circles. s ~ sqrt(pi/N) * constant
        s = 0.06 * scale
        
        y_step = s * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / s) + 2
        
        points_list = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else s / 2
            for i in range(-max_i, max_i + 1):
                x = i * s + offset
                if np.hypot(x, y) < 0.98:
                    points_list.append([x, y])
                    
        if len(points_list) < N:
            continue
            
        points = np.array(points_list)
        
        if len(points) > N:
            # Keep points closest to center
            dists_to_center = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists_to_center)[:N]
            points = points[idx]
            
        score = compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
    if best_points is None:
        # Fallback random
        best_points = np.random.uniform(-0.9, 0.9, (N, 2))
        best_score = compute_score(best_points)
        
    return best_points, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    # 1. Initialization
    points, score = init_hex()
    
    # 2. Simulated Annealing
    # Use 60% of time for SA
    sa_time = time_limit * 0.6
    elapsed = time.time() - start_time
    if elapsed < sa_time:
        points, score = simulated_annealing(points, sa_time - elapsed)
        
    # 3. Polish
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining > 1.0:
        points, score = polish_lbfgsb(points, remaining)
        
    return points
