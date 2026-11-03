import numpy as np
import time
import scipy.optimize as opt

N = 300
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def _hex_init(N):
    s = 0.11
    points = []
    dy = s * np.sqrt(3.0) / 2.0
    k = int(1.0 / s) + 2
    for j in range(-k, k + 1):
        y = j * dy
        if abs(y) > 1.0:
            continue
        x_offset = 0.0 if j % 2 == 0 else s / 2.0
        for i in range(-k, k + 1):
            x = i * s + x_offset
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) > N:
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # Fallback: create random points in circle
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        points = np.c_[radii * np.cos(angles), radii * np.sin(angles)]

    points += np.random.normal(0, 0.005, points.shape)
    return points

def _soft_min_value_and_grad(points, beta):
    """
    Returns value to minimize (-soft_min) and its gradient.
    """
    N = len(points)
    # Boundary terms
    norms = np.linalg.norm(points, axis=1)
    # Avoid division by zero
    norms_safe = np.where(norms == 0, 1e-9, norms)
    z_bound = 1.0 - norms
    grad_bound = -points / norms_safe[:, np.newaxis] # shape (N, 2)
    
    # Pair terms
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    dists_safe = np.where(dists == 0, 1e-9, dists)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    # We need gradients for each point i from all pairs (i, j)
    # grad_pair[i] = sum_{j != i} weight_ij * 0.5 * (p_i - p_j) / dist_ij
    
    # Calculate weights
    # z_pair = dists / 2
    z_pair = dists / 2.0
    
    # We need to combine z_bound and z_pair into a single list for the softmax
    # To do this efficiently, we can compute the log-sum-exp for all terms.
    
    # Flatten z_pair (upper triangle)
    z_pair_flat = z_pair[mask]
    # Flatten diff for upper triangle? 
    # It's easier to work with the full matrix and mask later, or extract indices.
    i_idx, j_idx = np.where(mask)
    diff_flat = diff[i_idx, j_idx]
    dists_flat = dists[i_idx, j_idx]
    
    all_z = np.concatenate([z_bound, z_pair_flat])
    
    # Softmax weights
    z_min = np.min(all_z)
    # Shift
    z_shifted = all_z - z_min
    exp_terms = np.exp(-beta * z_shifted)
    sum_exp = np.sum(exp_terms)
    
    # Weights
    w_bound = exp_terms[:N] / sum_exp
    w_pair = exp_terms[N:] / sum_exp
    
    # Value
    log_sum = np.log(sum_exp)
    soft_min_val = z_min - (1.0 / beta) * log_sum
    value = -soft_min_val
    
    # Gradient
    # grad_total = sum(w_bound * grad_bound) + sum(w_pair * grad_pair_terms)
    
    # Boundary gradient contribution
    grad_total = np.sum(w_bound[:, np.newaxis] * grad_bound, axis=0)
    
    # Pair gradient contribution
    # For each pair (i, j) with weight w_k:
    # grad p_i += w_k * 0.5 * (p_i - p_j) / dist_ij
    # grad p_j += w_k * 0.5 * (p_j - p_i) / dist_ij
    
    # diff_flat is p_i - p_j
    # dists_flat is |p_i - p_j|
    # grad_vec = 0.5 * diff_flat / dists_flat[:, np.newaxis]
    grad_vec = 0.5 * diff_flat / dists_flat[:, np.newaxis]
    
    # Weighted grad vec
    weighted_grads = w_pair[:, np.newaxis] * grad_vec
    
    # Accumulate to points
    # For each k, add to i_idx[k] and subtract from j_idx[k]
    # Using np.add.at for accumulation
    np.add.at(grad_total, i_idx, weighted_grads)
    np.subtract.at(grad_total, j_idx, weighted_grads)
    
    return value, grad_total

def construct():
    global START_TIME
    pts = _hex_init(N)
    best_pts = pts.copy()
    best_score = -1e9
    
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        def objective_and_jac(x):
            p = x.reshape(-1, 2)
            val, grad = _soft_min_value_and_grad(p, beta)
            return val, grad.flatten()
            
        try:
            res = opt.minimize(
                objective_and_jac, 
                pts.flatten(), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            if np.all(np.isfinite(res.x)):
                new_pts = res.x.reshape(-1, 2)
                # Calculate actual score
                d_bound = 1.0 - np.linalg.norm(new_pts, axis=1)
                diff = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
                d_pair = np.linalg.norm(diff, axis=2) / 2.0
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                min_pair = np.min(d_pair[mask])
                min_bound = np.min(d_bound)
                r = min(min_pair, min_bound)
                
                if r > best_score:
                    best_score = r
                    best_pts = new_pts
                    pts = new_pts
        except Exception:
            pass

    # Final check
    return best_pts
