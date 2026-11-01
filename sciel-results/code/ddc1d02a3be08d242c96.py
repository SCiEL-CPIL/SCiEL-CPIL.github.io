import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(p):
    """Compute the valid radius for a configuration."""
    if p.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(p)):
        return 0.0
    if np.any(np.abs(p) > 0.5 + 1e-12):
        return 0.0
    
    # Boundary constraint
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    
    # Pairwise constraint
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    
    return min(min_b, min_d)

def _softmin_radius_and_grad(p, beta=50.0):
    """
    Smooth approximation of min radius using softmin, with analytic gradient.
    
    Returns (softmin_value, gradient_wrt_p) where gradient is d(softmin)/dp.
    """
    # Boundary distances: b_i = 0.5 - max_j |p_ij|
    abs_p = np.abs(p)
    max_abs = np.max(abs_p, axis=1)  # (N,)
    b = 0.5 - max_abs  # (N,)
    
    # Pairwise distances / 2
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]  # (N,N,3)
    dists = np.linalg.norm(diffs, axis=2)  # (N,N)
    pd = dists / 2.0  # (N,N)
    
    # We want softmin over all constraints: b_i and pd_ij for i < j
    # Let's collect all constraint values
    b_vals = b  # (N,)
    pd_upper = pd[np.triu_indices_from(pd, k=1)]  # (N*(N-1)/2,)
    
    all_vals = np.concatenate([b_vals, pd_upper])
    
    # Softmin: -logsumexp(-beta * x) / beta
    # = min(x) - log(sum(exp(-beta*(x - min(x)))))/beta
    
    min_val = np.min(all_vals)
    shifted = all_vals - min_val  # >= 0
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = min_val - log_sum / beta
    
    # Gradient computation
    # d(softmin)/dx_k = -exp(-beta*(x_k - min_val)) / sum(exp(-beta*(x_j - min_val))) * (-beta) * (1/beta)
    # Wait, let me derive properly.
    # softmin(x) = min(x) - (1/beta) * log(sum_j exp(-beta*(x_j - min(x))))
    # This is tricky because min(x) is not differentiable.
    
    # Better approach: use the standard log-sum-exp formulation
    # softmin(x) = -(1/beta) * log(sum_j exp(-beta * x_j))
    # This is smooth and differentiable.
    # d(softmin)/dx_k = exp(-beta*x_k) / sum_j exp(-beta*x_j)
    
    # For numerical stability, shift by min_val
    # softmin = min_val - (1/beta) * log(sum_j exp(-beta*(x_j - min_val)))
    # d(softmin)/dx_k = -exp(-beta*(x_k - min_val)) / sum_j exp(-beta*(x_j - min_val)) * (-beta) * (1/beta)
    #                  = exp(-beta*(x_k - min_val)) / sum_j exp(-beta*(x_j - min_val))
    
    weights = exp_terms / np.sum(exp_terms)  # (N + N*(N-1)/2,)
    
    # Now compute gradient w.r.t. p
    grad = np.zeros_like(p)
    
    # Boundary part: b_i = 0.5 - max_j |p_ij|
    # db_i/dp_ik = -sign(p_ik) if |p_ik| = max_j |p_ij|, else 0
    # For simplicity, use the argmax axis
    for i in range(N):
        max_idx = np.argmax(abs_p[i])
        grad[i, max_idx] += weights[i] * (-np.sign(p[i, max_idx]))
    
    # Pairwise part: pd_ij = |p_i - p_j| / 2
    # dpd_ij/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
    # dpd_ij/dp_j = (p_j - p_i) / (2 * |p_i - p_j|)
    
    # Index mapping: b_vals is indices 0..N-1, pd_upper starts at N
    # pd_upper[k] corresponds to pair (i,j) where i < j
    # We need to map k -> (i,j)
    
    pair_idx = 0
    for i in range(N):
        for j in range(i+1, N):
            w = weights[N + pair_idx]
            if dists[i, j] > 1e-10:
                unit_vec = diffs[i, j] / dists[i, j]  # (p_i - p_j)/|p_i-p_j|
                grad[i] += w * unit_vec / 2.0
                grad[j] -= w * unit_vec / 2.0
            pair_idx += 1
    
    return softmin_val, grad

def _smooth_obj_and_grad(p, beta=50.0):
    """Objective to minimize: negative softmin radius, with gradient."""
    val, grad = _softmin_radius_and_grad(p, beta)
    return -val, -grad

def _repulsion_anneal(p0, target_r, n_steps=200, time_limit=None, start_time=None):
    """Repulsion-based local search to push points apart while respecting target radius."""
    p = p0.copy()
    limits = 0.5 - target_r
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        forces = np.zeros_like(p)
        
        # Boundary forces
        for axis in range(3):
            over_pos = p[:, axis] > limits
            over_neg = p[:, axis] < -limits
            forces[over_pos, axis] -= 1.0
            forces[over_neg, axis] += 1.0
        
        # Pairwise repulsion
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        mask = dists < 2 * target_r
        np.fill_diagonal(mask, False)
        
        if np.any(mask):
            safe_dists = np.where(dists > 1e-10, dists, 1.0)
            unit_vecs = diffs / safe_dists[:, :, np.newaxis]
            unit_vecs = unit_vecs * mask[:, :, np.newaxis]
            pair_forces = np.sum(unit_vecs, axis=1)
            forces += pair_forces
        
        # Adaptive step size
        step_size = 0.02 * (1.0 - 0.5 * step / n_steps)
        p += step_size * forces
        
        # Clamp to cube
        p = np.clip(p, -0.5, 0.5)
    
    return p

def _seed_fcc():
    """Generate FCC lattice points and select the most compact cluster."""
    points = []
    max_idx = 8
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    return points[idx[:N]].copy()

def _seed_grid():
    """Generate a 3D grid of points."""
    # 4x4x4 = 64 points, take 50
    coords = np.linspace(-0.4, 0.4, 4)
    grid = np.array([[x, y, z] for x in coords for y in coords for z in coords])
    grid -= np.mean(grid, axis=0)
    
    dists = np.linalg.norm(grid, axis=1)
    idx = np.argsort(dists)
    
    return grid[idx[:N]].copy()

def _seed_random(seed=42):
    """Generate random points."""
    rng = np.random.RandomState(seed)
    p = rng.uniform(-0.4, 0.4, size=(N, 3))
    return p

def _polish(p0, time_limit, start_time):
    """
    High-precision polishing using analytic gradients.
    Multi-scale beta with L-BFGS-B.
    """
    p = p0.copy()
    best_r = _compute_radius(p)
    best_p = p.copy()
    
    betas = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        def obj_and_grad(x):
            val, grad = _smooth_obj_and_grad(x.reshape(N, 3), beta=beta)
            return val, grad.flatten()
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result = minimize(obj_and_grad, p.flatten(), method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 100, 'gtol': 1e-8})
            p_new = result.x.reshape(N, 3)
            r_new = _compute_radius(p_new)
            
            if r_new > best_r + 1e-10:
                best_r = r_new
                best_p = p_new.copy()
                p = p_new
        except Exception:
            pass
    
    return best_p, best_r

def _local_search(p0, time_limit, start_time):
    """
    Combined local search:
    1. Repulsion annealing for initial spreading.
    2. Multi-scale soft-min gradient polishing.
    3. Final high-precision polish.
    """
    p = p0.copy()
    best_r = _compute_radius(p)
    best_p = p.copy()
    
    # Phase 1: Repulsion annealing to spread points
    if time.time() - start_time < time_limit * 0.3:
        p_rep = _repulsion_anneal(p, best_r, n_steps=50,
                                  time_limit=time_limit * 0.3, start_time=start_time)
        r_rep = _compute_radius(p_rep)
        if r_rep > best_r:
            best_r = r_rep
            best_p = p_rep.copy()
            p = p_rep
    
    # Phase 2: Multi-scale polishing
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        p_polished, r_polished = _polish(p, remaining_time, start_time)
        if r_polished > best_r:
            best_r = r_polished
            best_p = p_polished.copy()
    
    return best_p, best_r

def construct():
    start_time = time.time()
    total_time_limit = 140.0  # Safety margin from 150s
    
    # Try multiple seeds and pick the best
    seeds = [
        ('fcc', _seed_fcc),
        ('grid', _seed_grid),
        ('random', _seed_random),
    ]
    
    best_p = None
    best_r = -1
    
    # Allocate time per seed
    time_allocations = [0.5, 0.3, 0.2]
    available_time = total_time_limit - 5.0
    
    for i, (seed_name, seed_func) in enumerate(seeds):
        if time.time() - start_time > total_time_limit - 5.0:
            break
        
        p0 = seed_func()
        
        # Scale to fit in cube with some margin
        max_coord = np.max(np.abs(p0))
        if max_coord > 0:
            scale = 0.42 / max_coord
            p0 = p0 * scale
        
        # Local search
        seed_time_limit = time_allocations[i] * available_time
        seed_time_limit = min(seed_time_limit, total_time_limit - 5.0 - (time.time() - start_time))
        if seed_time_limit < 5.0:
            break
        
        p_improved, r_improved = _local_search(p0, seed_time_limit, start_time)
        
        if r_improved > best_r:
            best_r = r_improved
            best_p = p_improved
    
    # Final high-precision polish with very high beta
    if best_p is not None and time.time() - start_time < total_time_limit - 3.0:
        p = best_p.copy()
        
        def obj_and_grad(x):
            val, grad = _smooth_obj_and_grad(x.reshape(N, 3), beta=500.0)
            return val, grad.flatten()
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result = minimize(obj_and_grad, p.flatten(), method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 100, 'gtol': 1e-9})
            p_smooth = result.x.reshape(N, 3)
            r_smooth = _compute_radius(p_smooth)
            
            if r_smooth > best_r:
                best_r = r_smooth
                best_p = p_smooth
        except:
            pass
    
    # Final validation
    final_r = _compute_radius(best_p)
    if final_r > best_r:
        best_r = final_r
    
    return best_p
