# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def compute_distances(points):
    """Compute all pairwise distances using scipy."""
    return spd.pdist(points)

def smooth_min_max_ratio(points, beta):
    """
    Compute a smooth surrogate for (d_min / d_max)^2.
    Uses log-sum-exp for soft max and min.
    """
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    
    max_d = np.max(dists)
    min_d = np.min(dists)
    
    exp_d = np.exp((dists - max_d) / beta)
    soft_max = max_d + beta * np.log(np.sum(exp_d))
    
    exp_neg_d = np.exp((min_d - dists) / beta)
    soft_min = min_d - beta * np.log(np.sum(exp_neg_d))
    
    if soft_max == 0:
        return 0.0
        
    ratio = soft_min / soft_max
    return ratio * ratio

def smooth_min_max_ratio_grad(points, beta):
    """
    Compute the smooth surrogate and its gradient.
    Returns (value, grad) where grad has shape (n, 2).
    """
    n = points.shape[0]
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0, np.zeros((n, 2))
    
    max_d = np.max(dists)
    min_d = np.min(dists)
    
    exp_d = np.exp((dists - max_d) / beta)
    soft_max = max_d + beta * np.log(np.sum(exp_d))
    
    exp_neg_d = np.exp((min_d - dists) / beta)
    soft_min = min_d - beta * np.log(np.sum(exp_neg_d))
    
    if soft_max == 0:
        return 0.0, np.zeros((n, 2))
        
    ratio = soft_min / soft_max
    value = ratio * ratio
    
    sum_exp_max = np.sum(exp_d)
    sum_exp_min = np.sum(exp_neg_d)
    
    # Build index mapping for pdist
    # pdist index for pair (i,j) with i<j: i*(2n-i-1)//2 + (j-i-1)
    grad = np.zeros((n, 2))
    
    # Precompute unit vectors and weights for each pair
    pair_indices = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    
    for (i, j) in pair_indices:
        d_ij = dists[i * (2*n - i - 1) // 2 + (j - i - 1)]
        if d_ij < 1e-12:
            continue
        
        k = i * (2*n - i - 1) // 2 + (j - i - 1)
        w_max = exp_d[k] / sum_exp_max
        w_min = exp_neg_d[k] / sum_exp_min
        
        diff = points[i] - points[j]
        unit_diff = diff / d_ij
        
        # d(soft_max)/d(p_i) += w_max * unit_diff
        # d(soft_max)/d(p_j) -= w_max * unit_diff
        # d(soft_min)/d(p_i) -= w_min * unit_diff  (because d(soft_min)/d(d_k) is negative, and d(d_k)/d(p_i) = unit_diff)
        # d(soft_min)/d(p_j) += w_min * unit_diff
        
        # d(value)/d(p_i) = 2*ratio/soft_max^2 * (d(soft_min)/d(p_i)*soft_max - soft_min*d(soft_max)/d(p_i))
        # = 2*ratio/soft_max^2 * (-w_min*unit_diff*soft_max - soft_min*w_max*unit_diff)
        # = -2*ratio/soft_max^2 * unit_diff * (w_min*soft_max + soft_min*w_max)
        
        coeff_max = 2 * ratio * w_max / (soft_max * soft_max)
        coeff_min = 2 * ratio * w_min * soft_max / (soft_max * soft_max)
        coeff_total = coeff_max + coeff_min
        
        grad[i] += -coeff_total * unit_diff
        grad[j] += coeff_total * unit_diff
    
    return value, grad

def complex_to_points(z):
    """Convert complex numbers to 2D points, with z[0] fixed at origin."""
    # z[0] is fixed at 0 (origin)
    # z[1:] are the other 15 points
    n = len(z)
    points = np.zeros((n, 2))
    for i in range(1, n):
        points[i, 0] = z[i].real
        points[i, 1] = z[i].imag
    return points

def points_to_complex(points):
    """Convert 2D points to complex numbers, with first point at origin."""
    n = len(points)
    z = np.zeros(n, dtype=complex)
    # Translate so first point is at origin
    offset = points[0]
    for i in range(1, n):
        z[i] = complex(points[i, 0] - offset[0], points[i, 1] - offset[1])
    return z

def objective_flat(x_flat, beta=0.1):
    """
    Objective function for flat representation of all 16 points.
    x_flat is a real array of length 32.
    """
    n = 16
    points = x_flat.reshape(n, 2)
    return -smooth_min_max_ratio(points, beta)

def objective_flat_grad(x_flat, beta=0.1):
    """
    Objective function and gradient for flat representation.
    """
    n = 16
    points = x_flat.reshape(n, 2)
    val, grad = smooth_min_max_ratio_grad(points, beta)
    return -val, -grad.flatten()

def lbfgsb_optimize(points, time_limit, beta=0.1, seed=None):
    """
    Optimize using L-BFGS-B with analytic gradient.
    """
    n = 16
    x_flat = points.flatten().copy()
    
    start_time = time.time()
    best_x_flat = x_flat.copy()
    best_score = calculate_score(points)
    
    time_exceeded = [False]
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            time_exceeded[0] = True
            raise ValueError("Time limit exceeded")
        return False
    
    try:
        res = minimize(objective_flat, x_flat, 
                      method='L-BFGS-B',
                      jac=lambda x: objective_flat_grad(x, beta)[1],
                      callback=callback,
                      options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40, 'maxfun': 5000})
        
        if res.success or not time_exceeded[0]:
            trial_score = calculate_score(res.x.reshape(n, 2))
            if trial_score > best_score:
                best_score = trial_score
                best_x_flat = res.x.copy()
    except (ValueError, Exception):
        pass
    
    best_points = best_x_flat.reshape(n, 2)
    return best_points, best_score

def local_search(points, time_limit, seed=None, use_smooth=False, beta=0.1):
    """
    Perform local search to maximize the score.
    Uses simulated annealing with adaptive step size.
    """
    n = points.shape[0]
    current_points = points.copy()
    
    if use_smooth:
        current_score = smooth_min_max_ratio(current_points, beta)
    else:
        current_score = calculate_score(current_points)
        
    best_points = current_points.copy()
    best_score = current_score
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
        
    start_time = time.time()
    temp = 0.05
    min_temp = 1e-6
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        idx = rng.randint(n)
        
        # Adaptive step size based on current temperature
        step_size = 0.1 * temp + 0.005
        perturbation = rng.randn(2) * step_size
        
        new_point = current_points[idx].copy()
        new_point += perturbation
        
        trial_points = current_points.copy()
        trial_points[idx] = new_point
        
        if use_smooth:
            trial_score = smooth_min_max_ratio(trial_points, beta)
        else:
            trial_score = calculate_score(trial_points)
            
        if trial_score >= current_score:
            current_points = trial_points
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            delta = current_score - trial_score
            if delta < 1e-10:
                current_points = trial_points
                current_score = trial_score
            else:
                if rng.rand() < np.exp(-delta / (temp + 1e-10)):
                    current_points = trial_points
                    current_score = trial_score
        
        temp *= cooling_rate
        if temp < min_temp:
            temp = min_temp
            
        if time.time() - start_time > time_limit * 0.95:
            break

    final_score = calculate_score(best_points)
    return best_points, final_score

def nelder_mead_polish(points, time_limit, seed=None):
    """
    Use Nelder-Mead on the smooth surrogate to polish the points.
    """
    n = points.shape[0]
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(n, 2)
        return -smooth_min_max_ratio(pts, beta=0.01)
        
    start_time = time.time()
    best_pts = points.copy()
    best_score = calculate_score(best_pts)
    
    time_exceeded = [False]
    def callback(xk):
        if time.time() - start_time > time_limit:
            time_exceeded[0] = True
            raise ValueError("Time limit exceeded")
        return False
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       callback=callback,
                       options={'maxiter': 2000, 'fatol': 1e-14, 'xatol': 1e-14, 'initial_simplex': 1e-4, 'adaptive': True})
        
        if res.success or not time_exceeded[0]:
            trial_pts = res.x.reshape(n,2)
            trial_score = calculate_score(trial_pts)
            if trial_score > best_score:
                best_pts = trial_pts
                best_score = trial_score
    except (ValueError, Exception):
        pass
        
    return best_pts, best_score

def optimize_geometry(points, time_limit, seed=None):
    """
    A structured optimizer combining L-BFGS-B, local search, and Nelder-Mead polish.
    """
    n = points.shape[0]
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
        
    start_time = time.time()
    
    while time.time() - start_time < time_limit - 3:
        remaining = remaining_time_calc(start_time, time_limit) - 3
        if remaining < 5:
            break
        
        # Phase 1: L-BFGS-B with analytic gradient (most effective)
        chunk1 = min(remaining * 0.40, 30)
        if chunk1 > 3:
            # Use progressively smaller beta for convergence
            beta_val = 0.08 if remaining > 30 else 0.04
            res_points, res_score = lbfgsb_optimize(current_points, chunk1, beta=beta_val, seed=seed)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                current_points = best_points.copy()
        
        # Phase 2: Smooth local search
        remaining = remaining_time_calc(start_time, time_limit) - 3
        if remaining < 5:
            break
        chunk2 = min(remaining * 0.20, 12)
        if chunk2 > 3:
            res_points, res_score = local_search(current_points, chunk2, seed=seed+100, use_smooth=True, beta=0.06)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                current_points = best_points.copy()
                
        # Phase 3: Nelder-Mead polish
        remaining = remaining_time_calc(start_time, time_limit) - 3
        if remaining < 5:
            break
        chunk3 = min(remaining * 0.20, 12)
        if chunk3 > 3:
            res_points, res_score = nelder_mead_polish(current_points, chunk3, seed=seed+200)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                current_points = best_points.copy()
                
        # Phase 4: Exact local search
        remaining = remaining_time_calc(start_time, time_limit) - 3
        if remaining < 5:
            break
        chunk4 = min(remaining * 0.15, 10)
        if chunk4 > 2:
            res_points, res_score = local_search(current_points, chunk4, seed=seed+300, use_smooth=False)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                current_points = best_points.copy()
                
        # Perturb for next restart (gentler perturbations)
        num_perturb = rng.randint(1, 3)
        indices = rng.choice(n, num_perturb, replace=False)
        for idx in indices:
            current_points[idx] += rng.randn(2) * 0.06
            
    return best_points, best_score

def remaining_time_calc(start_time, total_limit):
    return total_limit - (time.time() - start_time)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    global_deadline = time.time() + 625
    
    candidates = []
    
    # Candidate 1: 4x4 Square Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    candidates.append(grid_points)
    
    # Candidate 2: Points on two concentric circles
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(np.pi/8, 2*np.pi, 8, endpoint=False)
    r1 = 1.0
    r2 = 0.5
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    circle_points = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
    candidates.append(circle_points)
    
    # Candidate 3: Hexagonal lattice points (first 16 in a hex grid)
    hex_grid = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = q + 0.5 * r
            y = np.sqrt(3)/2 * r
            hex_grid.append([x, y])
    hex_grid = np.array(hex_grid)
    dists_to_origin = np.linalg.norm(hex_grid, axis=1)
    idx = np.argsort(dists_to_origin)[:16]
    hex_candidate = hex_grid[idx]
    candidates.append(hex_candidate)
    
    # Candidate 4: Symmetric configuration with 1 center + 6 + 9
    center = np.array([0.0, 0.0])
    ring1_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = np.linspace(0, 2*np.pi, 9, endpoint=False)
    r2 = np.sqrt(3)
    ring2 = r2 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    symmetric_points = np.vstack([center, ring1, ring2])
    candidates.append(symmetric_points)
    
    best_points = None
    best_score = 0
    
    num_candidates = len(candidates)
    
    # Allocate time: 4 candidates, ~100s each
    base_time = 100
    
    for i, cand in enumerate(candidates):
        if time.time() > global_deadline - 60:
            break
            
        cand = cand.astype(float)
        
        # Ensure we don't exceed deadline
        max_allowed = global_deadline - 60 - time.time()
        time_per_candidate = min(base_time, max_allowed)
        
        if time_per_candidate < 10:
            continue
        
        res_points, res_score = optimize_geometry(cand, time_per_candidate, seed=1000+i)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # Final high-precision polish with remaining time
    if best_points is not None and time.time() < global_deadline - 10:
        remaining_time = global_deadline - 10 - time.time()
        if remaining_time > 20:
            # Use very small beta for high-precision L-BFGS-B
            res_points, res_score = lbfgsb_optimize(best_points, remaining_time * 0.6, beta=0.02, seed=8888)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
            
            # Final Nelder-Mead polish
            remaining_time = global_deadline - 10 - time.time()
            if remaining_time > 10:
                res_points, res_score = nelder_mead_polish(best_points, remaining_time, seed=7777)
                if res_score > best_score:
                    best_points = res_points
                    best_score = res_score

    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))
        
    return best_points

# EVOLVE-BLOCK-END
