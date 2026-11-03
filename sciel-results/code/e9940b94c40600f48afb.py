"""Combined program: Iterative Radius Increase with Energy Minimization."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100
HALF = 0.5
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _compute_radius(C):
    """Compute the maximum possible radius for a set of centers C."""
    if len(C) < 2:
        return HALF - np.max(np.abs(C))
    dist_to_bound = HALF - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    tree = cKDTree(C)
    distances, _ = tree.query(C, k=2)
    min_dist = np.min(distances[:, 1])
    r_pair = min_dist / 2.0
    return min(r_bound, r_pair)

def _soft_min_objective(flat_pos, beta=50.0):
    """
    Compute the negative soft-minimum distance and its gradient.
    The soft-min approximates min( min_{i<j} |c_i - c_j|/2, min_i min_k (1/2 - |c_ik|) ).
    We maximize this, so we minimize the negative.
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    d_walls = HALF - np.abs(pos)
    wall_dists = d_walls.flatten()
    
    # Effective distances: radius candidates
    # r_pair = d / 2
    # r_wall = d_wall
    eff_dists = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Soft-min: -1/beta * log(sum(exp(-beta * x)))
    # To avoid overflow/underflow, use log-sum-exp trick
    # S = sum(exp(-beta * x))
    # log(S) = max(-beta*x) + log(sum(exp(-beta*x - max(-beta*x))))
    
    scaled = -beta * eff_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    
    soft_min_val = -log_sum_exp / beta
    
    # Gradient
    # d(soft_min)/dx_i = -1/beta * d(log_sum_exp)/dx_i
    # d(log_sum_exp)/dx_i = 1/S * d(S)/dx_i = 1/S * S * (-beta) * exp(-beta*x_i)/exp(-beta*x_i) ... wait
    # Let u_i = -beta * x_i. S = sum(exp(u_i)).
    # d(log S)/dx_i = 1/S * sum_j (d S/dx_i) ... no, S depends on x_i only through term i.
    # d S / d x_i = exp(u_i) * (-beta)
    # d log S / d x_i = exp(u_i) / S * (-beta)
    # weights w_i = exp(u_i) / S
    # d soft_min / d x_i = -1/beta * w_i * (-beta) = w_i
    
    # So gradient of soft_min w.r.t x_i is w_i.
    # We want to maximize soft_min, so we minimize -soft_min.
    # Gradient of -soft_min w.r.t x_i is -w_i.
    # We need gradient w.r.t pos.
    
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_pos = np.zeros_like(pos)
    
    # Pairwise part
    # x_pair_ij = d_ij / 2
    # d x_pair_ij / d pos_i = (pos_i - pos_j) / (2 * d_ij)
    # d (-soft_min) / d pos_i = -w_pair_ij * (pos_i - pos_j) / (2 * d_ij)
    
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx]
        d = pair_dists[idx]
        if d > 1e-14:
            vec = pos[i] - pos[j]
            coeff = w / (2.0 * d)
            # grad = -w * vec / (2d)
            grad_pos[i] -= coeff * vec
            grad_pos[j] += coeff * vec
            
    # Wall part
    # x_wall_ik = 0.5 - |pos_ik|
    # d x_wall_ik / d pos_ik = -sign(pos_ik)
    # d (-soft_min) / d pos_ik = -w_wall_ik * (-sign(pos_ik)) = w_wall_ik * sign(pos_ik)
    
    wall_weights = weights[len(pair_dists):]
    for idx in range(N * 3):
        i = idx // 3
        k = idx % 3
        w = wall_weights[idx]
        s = np.sign(pos[i, k])
        if s == 0: s = 1.0
        grad_pos[i, k] += w * s
        
    return -soft_min_val, -grad_pos.flatten()

def _polish(pos, max_iter=300):
    """Polish position using L-BFGS-B on the soft-min objective."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _soft_min_objective(x)[0],
            pos.flatten(),
            jac=lambda x: _soft_min_objective(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if np.isfinite(res.fun):
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def _simulated_annealing(pos, rng, start_temp=0.05, end_temp=0.001, steps=2000):
    """
    Simulated annealing to escape local minima.
    Objective: maximize soft-min radius.
    """
    current_pos = pos.copy()
    current_val, _ = _soft_min_objective(current_pos.flatten())
    current_val = -current_val # We minimized -val, so actual val is -min_val
    
    T = start_temp
    cooling_rate = (end_temp / start_temp) ** (1.0 / steps)
    
    for _ in range(steps):
        if _check_time():
            break
            
        # Random move
        i = rng.integers(N)
        # Scale move with temperature
        move_scale = T * 2.0
        move = rng.normal(0, move_scale, 3)
        new_pos = current_pos.copy()
        new_pos[i] += move
        
        # Clip to bounds (simple constraint handling)
        new_pos[i] = np.clip(new_pos[i], -HALF, HALF)
        
        new_val, _ = _soft_min_objective(new_pos.flatten())
        new_val = -new_val
        
        delta = new_val - current_val
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_pos = new_pos
            current_val = new_val
            
        T *= cooling_rate
        
    return current_pos

def _generate_initial_points(seed):
    """Generate diverse initial points."""
    rng = np.random.default_rng(seed)
    # Mix of BCC, FCC, and Random
    method = seed % 3
    if method == 0:
        # BCC-like
        size = 4
        raw = []
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    raw.append([i, j, k])
                    raw.append([i + 0.5, j + 0.5, k + 0.5])
        raw = np.array(raw)
        center = np.array([size/2 - 0.25, size/2 - 0.25, size/2 - 0.25])
        raw = raw - center
        max_coord = np.max(np.abs(raw))
        scale = 0.45 / max_coord
        raw *= scale
        idx = rng.permutation(len(raw))
        return raw[idx[:N]].copy()
    elif method == 1:
        # FCC-like
        size = 4
        raw = []
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    raw.append([i, j, k])
                    raw.append([i + 0.5, j + 0.5, k])
                    raw.append([i + 0.5, j, k + 0.5])
                    raw.append([i, j + 0.5, k + 0.5])
        raw = np.array(raw)
        center = np.array([size/2 - 0.25, size/2 - 0.25, size/2 - 0.25])
        raw = raw - center
        max_coord = np.max(np.abs(raw))
        scale = 0.45 / max_coord
        raw *= scale
        idx = rng.permutation(len(raw))
        return raw[idx[:N]].copy()
    else:
        # Random
        return rng.uniform(-0.45, 0.45, (N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)
    
    best_radius = 0.0
    best_centers = None
    
    # Try multiple seeds with different initializations
    seeds = [42, 123, 456, 789, 101112, 202021, 303031, 404041]
    
    for seed in seeds:
        if _check_time():
            break
            
        centers = _generate_initial_points(seed)
        centers = np.clip(centers, -HALF + 1e-9, HALF - 1e-9)
        
        # 1. Initial Polish to get a local optimum
        polished = _polish(centers, max_iter=100)
        r_curr = _compute_radius(polished)
        
        # 2. Simulated Annealing to escape local minima
        if time.time() < deadline - 30.0:
            rng = np.random.default_rng(seed)
            # Adjust SA parameters based on remaining time
            # Use a moderate number of steps
            sa_steps = 1000
            sa_polished = _simulated_annealing(polished, rng, start_temp=0.05, end_temp=0.001, steps=sa_steps)
            sa_polished = _polish(sa_polished, max_iter=100)
            r_sa = _compute_radius(sa_polished)
            
            if r_sa > r_curr:
                polished = sa_polished
                r_curr = r_sa
        
        # 3. Final high-precision polish
        if not _check_time():
            final_polished = _polish(polished, max_iter=300)
            r_final = _compute_radius(final_polished)
            
            if r_final > r_curr:
                r_curr = r_final
                polished = final_polished
        
        if r_curr > best_radius:
            best_radius = r_curr
            best_centers = polished.copy()
            
        # If we have time, try a second round of SA from the best found so far
        if time.time() < deadline - 60.0 and best_centers is not None:
            rng2 = np.random.default_rng(seed + 999)
            # Perturb best centers slightly and SA again
            perturbed = best_centers + rng2.normal(0, 0.01, (N, 3))
            perturbed = np.clip(perturbed, -HALF + 1e-9, HALF - 1e-9)
            
            sa_polished2 = _simulated_annealing(perturbed, rng2, start_temp=0.02, end_temp=0.0005, steps=500)
            sa_polished2 = _polish(sa_polished2, max_iter=100)
            r_sa2 = _compute_radius(sa_polished2)
            
            if r_sa2 > best_radius:
                best_radius = r_sa2
                best_centers = sa_polished2.copy()
                
                # Final polish on the new best
                if not _check_time():
                    final_polished2 = _polish(best_centers, max_iter=300)
                    r_final2 = _compute_radius(final_polished2)
                    if r_final2 > best_radius:
                        best_radius = r_final2
                        best_centers = final_polished2.copy()

    best_centers = np.clip(best_centers, -HALF + 1e-10, HALF - 1e-10)
    return best_centers
