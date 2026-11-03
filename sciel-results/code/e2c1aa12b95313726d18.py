"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    min_pair = np.min(pdist(centers))
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_hex_grid(n_rows=None):
    """Generate a hexagonal lattice packing, centered in the square."""
    if n_rows is None:
        n_rows = 14
    
    n_cols = int(np.ceil(N / n_rows))
    
    r1 = 1.0 / (2.0 * n_cols)
    r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
    r = min(r1, r2)
    s = 2 * r
    
    centers = []
    for i in range(n_rows):
        for j in range(n_cols):
            if len(centers) >= N:
                break
            x = -0.5 + r + j * s + (i % 2) * s / 2
            y = -0.5 + r + i * s * np.sqrt(3) / 2
            centers.append([x, y])
        if len(centers) >= N:
            break
    
    centers = np.array(centers[:N])
    
    # Center the grid
    cx = (centers[:, 0].max() + centers[:, 0].min()) / 2
    cy = (centers[:, 1].max() + centers[:, 1].min()) / 2
    centers[:, 0] -= cx
    centers[:, 1] -= cy
    
    return centers

def _generate_best_hex():
    """Try multiple row counts and return the best hex grid."""
    best_centers = None
    best_r = 0.0
    
    for n_rows in range(10, 20):
        centers = _generate_hex_grid(n_rows=n_rows)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers

def _soft_min_objective_and_grad(flat_x, beta=100.0):
    """
    Smooth objective: -soft_min of all pairwise distances/2 and wall distances.
    Returns (objective, gradient) for L-BFGS-B.
    """
    centers = flat_x.reshape(N, 2)
    
    # Compute all pairwise distances
    d = pdist(centers)
    pair_dists = d / 2.0  # candidate radii from pairs
    
    # Wall distances
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft minimum
    d_min = np.min(all_dists)
    shifted = -(all_dists - d_min) * beta
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted))
    soft_min_val = d_min - log_sum / beta
    
    # We want to maximize soft_min, so objective = -soft_min
    objective = -soft_min_val
    
    # Gradient computation
    # d(soft_min)/d(all_dists[k]) = -exp_shifted[k] / sum(exp_shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    grad = np.zeros_like(flat_x)
    
    # Pairwise distance gradients
    # For pair (i,j) at index k in d, d(pair_dist[k])/d(c_i) = (c_i - c_j)/|c_i-c_j| * 0.5
    # We need to map pair indices back to point indices
    n_pairs = len(pair_dists)
    n_points = N
    
    # Build pair index mapping
    # pdist uses condensed form: pair (i,j) with i<j is at index i*n - i*(i+1)//2 + j - i - 1
    pair_idx = np.arange(n_pairs)
    # Invert condensed form
    ii = np.zeros(n_pairs, dtype=int)
    jj = np.zeros(n_pairs, dtype=int)
    for k in range(n_pairs):
        # Find i, j such that k = i*n - i*(i+1)//2 + j - i - 1
        # This is slow for large n, use vectorized approach
        pass
    
    # Vectorized inversion of condensed form
    # For condensed index k, find i and j
    # k = i*(n-1) - i*(i+1)//2 + j - i - 1 for 0-indexed
    # Better: use the formula
    # i = floor((n - 1 - sqrt(2*n*(n-1) - 8*k - 7)) / 2)
    n = n_points
    k_arr = pair_idx.astype(float)
    i_arr = np.floor((n - 1 - np.sqrt(2.0 * n * (n - 1) - 8.0 * k_arr - 7.0)) / 2.0).astype(int)
    j_arr = k_arr + i_arr + 1 + i_arr * (i_arr + 1) / 2.0 - i_arr * (n - 1)
    j_arr = np.round(j_arr).astype(int)
    # Verify
    # For safety, let's just do it the direct way for correctness
    i_arr = np.zeros(n_pairs, dtype=int)
    j_arr = np.zeros(n_pairs, dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            i_arr[idx] = i
            j_arr[idx] = j
            idx += 1
    
    # Gradient for pair distances
    centers_diff = centers[i_arr] - centers[j_arr]
    pair_dists_raw = np.sqrt(np.sum(centers_diff**2, axis=1))
    pair_dists_raw = np.maximum(pair_dists_raw, 1e-12)
    
    # weight for each pair
    pair_weights = weights[:n_pairs] * 0.5  # factor of 0.5 from d(pair_dist)/d(r)
    
    # Accumulate gradient
    for i in range(n_pairs):
        w = pair_weights[i]
        diff = centers_diff[i] / pair_dists_raw[i]
        grad[i_arr[i] * 2: i_arr[i] * 2 + 2] += w * diff
        grad[j_arr[i] * 2: j_arr[i] * 2 + 2] -= w * diff
    
    # Wall distance gradients
    wall_weights = weights[n_pairs:]
    for i in range(N):
        w = wall_weights[i]
        for dim in range(2):
            val = centers[i, dim]
            if val < 0:
                # wall_dist = 0.5 + val, d/d(val) = 1
                grad[i * 2 + dim] += w
            else:
                # wall_dist = 0.5 - val, d/d(val) = -1
                grad[i * 2 + dim] -= w
    
    return objective, grad

def _lbfgs_optimize(centers, beta=100.0, max_iter=500, time_budget=60.0):
    """Run L-BFGS-B optimization on the smooth objective."""
    flat_x = centers.flatten().copy()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    start_time = time.time()
    
    # Callback to check time
    def callback(xk):
        elapsed = time.time() - start_time
        return elapsed > time_budget
    
    result = minimize(
        _soft_min_objective_and_grad,
        flat_x,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={
            'maxiter': max_iter,
            'ftol': 1e-14,
            'gtol': 1e-10,
            'maxls': 50,
            'callback': callback,
            'maxfun': 5000,
        }
    )
    
    optimized = result.x.reshape(N, 2)
    return optimized

def _multi_beta_optimize(centers):
    """
    Run L-BFGS-B with increasing beta (annealing) to refine the packing.
    Start with smooth (low beta) and gradually sharpen.
    """
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    betas = [20, 50, 100, 200, 400, 800]
    
    for beta in betas:
        if _time_left() < 30.0:
            break
        
        time_budget = min(20.0, _time_left() - 25.0)
        if time_budget < 5.0:
            break
        
        try:
            optimized = _lbfgs_optimize(best_centers, beta=beta, max_iter=300, time_budget=time_budget)
        except Exception:
            break
        
        r = _compute_radius(optimized)
        if r > best_r:
            best_r = r
            best_centers = optimized.copy()
    
    return best_centers, best_r

def _local_perturbation(centers, n_restarts=3):
    """
    Try small random perturbations to escape local optima.
    """
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    for restart in range(n_restarts):
        if _time_left() < 30.0:
            break
        
        # Perturb a random subset of points
        n_perturb = 5 + restart * 3
        idx = np.random.choice(N, size=n_perturb, replace=False)
        perturbed = centers.copy()
        noise = np.random.normal(0, 0.005, (n_perturb, 2))
        perturbed[idx] += noise
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        r = _compute_radius(perturbed)
        
        # Quick refinement
        if _time_left() > 35.0:
            time_budget = min(15.0, _time_left() - 30.0)
            if time_budget > 5.0:
                try:
                    optimized = _lbfgs_optimize(perturbed, beta=100, max_iter=200, time_budget=time_budget)
                    r = _compute_radius(optimized)
                    if r > best_r:
                        best_r = r
                        best_centers = optimized.copy()
                except Exception:
                    pass
    
    return best_centers, best_r

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Start with best hexagonal grid
    centers = _generate_best_hex()
    r = _compute_radius(centers)
    
    # Phase 1: L-BFGS-B with annealed beta (main optimization)
    if _time_left() > 60.0:
        centers, r = _multi_beta_optimize(centers)
    
    # Phase 2: Local perturbation to escape local optima
    if _time_left() > 40.0:
        centers, r = _local_perturbation(centers, n_restarts=2)
    
    # Phase 3: Final high-beta polish
    if _time_left() > 25.0:
        time_budget = min(20.0, _time_left() - 20.0)
        if time_budget > 5.0:
            try:
                optimized = _lbfgs_optimize(centers, beta=800, max_iter=500, time_budget=time_budget)
                r_new = _compute_radius(optimized)
                if r_new > r:
                    centers = optimized
                    r = r_new
            except Exception:
                pass
    
    # Final verification and clipping
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
