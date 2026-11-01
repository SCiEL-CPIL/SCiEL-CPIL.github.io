"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_objective(params, beta=50.0):
    """
    Smooth surrogate for the negative radius using soft-min (log-sum-exp).
    We want to maximize min(pairwise_dists/2, boundary_dists).
    Soft-min approximates the minimum: -soft_min = soft_max of negatives.
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    # Pairwise distances / 2
    dists = pdist(c, metric='euclidean')
    pair_vals = dists / 2.0
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    
    # Combine all constraints into one array
    all_vals = np.concatenate([pair_vals, boundary_dists.flatten()])
    
    # Soft-min: -log-sum-exp(-beta * x) / beta approximates min(x) for large beta
    # We want to maximize the soft-min, so minimize negative soft-min
    # Numerically stable computation
    min_val = np.min(all_vals)
    shifted = beta * (all_vals - min_val)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -700, 700)
    lse = np.log(np.sum(np.exp(shifted))) - beta * min_val
    soft_min_val = min_val + lse / beta
    
    return -soft_min_val


def _active_set_polish(c0, max_time=15.0, tol=1e-12):
    """
    High-precision polish using only active (tight) constraints with analytical gradients.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    n = c.shape[0]
    
    def _get_active_sets(r_current, margin_factor):
        dists = pdist(c, metric='euclidean')
        pair_vals = dists / 2.0
        boundary_dists = 0.5 - np.abs(c)
        
        margin = margin_factor * r_current + 1e-6
        active_mask = pair_vals < (r_current + margin)
        active_pairs = np.where(active_mask)[0]
        active_boundary = np.where(boundary_dists.flatten() < (r_current + margin))[0]
        
        active_i = []
        active_j = []
        for idx in active_pairs:
            i = 0
            while i < n - 1:
                count = n - i - 1
                if idx < i * count + count:
                    break
                idx -= count
                i += 1
            j = i + 1 + idx
            active_i.append(i)
            active_j.append(j)
        
        return np.array(active_i), np.array(active_j), np.array(active_boundary)

    def _soft_min_active(params, a_i, a_j, a_b, beta=500.0):
        c_local = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        
        vals = []
        if len(a_i) > 0:
            diffs = c_local[a_i] - c_local[a_j]
            dists = np.sqrt(np.sum(diffs**2, axis=1) + 1e-16)
            vals.append(dists / 2.0)
        if len(a_b) > 0:
            bi = a_b // 2
            bj = a_b % 2
            bd = 0.5 - np.abs(c_local[bi, bj])
            vals.append(bd)
            
        if not vals:
            return 0.0, np.zeros_like(params)
            
        all_vals = np.concatenate(vals)
        min_val = np.min(all_vals)
        shifted = beta * (all_vals - min_val)
        shifted = np.clip(shifted, -700, 700)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        lse = np.log(sum_exp) - beta * min_val
        soft_min_val = min_val + lse / beta
        obj = -soft_min_val
        
        probs = exp_shifted / sum_exp
        grad_params = np.zeros_like(params)
        
        # Gradient for pairs
        if len(a_i) > 0:
            diffs = c_local[a_i] - c_local[a_j]
            dists = np.sqrt(np.sum(diffs**2, axis=1) + 1e-16)
            # d(obj)/dc = - sum p_k * d(v_k)/dc
            # d(v_k)/dc_i = 0.5 * (c_i - c_j) / dist
            # d(v_k)/dc_j = -0.5 * (c_i - c_j) / dist
            for k in range(len(a_i)):
                i = a_i[k]
                j = a_j[k]
                d = diffs[k]
                dist = dists[k]
                if dist < 1e-10: continue
                factor = -probs[k] * 0.5 / dist
                grad_params[i] += factor * d
                grad_params[j] -= factor * d
                
        # Gradient for boundaries
        if len(a_b) > 0:
            bi = a_b // 2
            bj = a_b % 2
            # v = 0.5 - |x|, dv/dx = -sign(x)
            # d(obj)/dx = - p * dv/dx = p * sign(x)
            signs = np.sign(c_local[bi, bj])
            # Handle 0 sign
            signs[signs == 0] = 0.0
            grad_params[bi, bj] += probs * signs
            
        return obj, grad_params

    # Pass 1
    a_i, a_j, a_b = _get_active_sets(r, 0.05)
    if len(a_i) > 0 or len(a_b) > 0:
        try:
            result = minimize(
                lambda p: _soft_min_active(p, a_i, a_j, a_b, beta=200.0),
                c.flatten(),
                jac=lambda p: _soft_min_active(p, a_i, a_j, a_b, beta=200.0)[1],
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > r:
                c = c_new
                r = r_new
        except Exception:
            pass

    # Pass 2 (tighter)
    if time.time() < start_time + max_time - 2:
        a_i, a_j, a_b = _get_active_sets(r, 0.02)
        if len(a_i) > 0 or len(a_b) > 0:
            try:
                result = minimize(
                    lambda p: _soft_min_active(p, a_i, a_j, a_b, beta=500.0),
                    c.flatten(),
                    jac=lambda p: _soft_min_active(p, a_i, a_j, a_b, beta=500.0)[1],
                    method='L-BFGS-B',
                    bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                    options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 100}
                )
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > r:
                    c = c_new
                    r = r_new
            except Exception:
                pass

    return c, r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.06, 0.08, 15):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _force_directed_init(c0, n_iter=500, k=0.005):
    """
    Force-directed relaxation: circles repel each other and are attracted to center.
    This helps untangle overlaps from random initialization.
    """
    c = c0.copy()
    n = c.shape[0]
    
    for it in range(n_iter):
        # Compute repulsive forces between all pairs
        # For efficiency, only consider pairs closer than some threshold
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion: F = k / dist^2 * direction (capped to avoid blowup)
        min_dist = 0.01
        safe_dists = np.maximum(dists, min_dist)
        forces = np.zeros_like(c)
        
        # Only compute for close pairs to save time
        mask = safe_dists < 0.15
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    d = safe_dists[i, j]
                    f = k / (d * d)
                    forces[i] += f * (c[i] - c[j]) / d
                    forces[j] -= f * (c[i] - c[j]) / d
        
        # Attraction to center (weak)
        forces -= 0.001 * c
        
        # Damping
        damping = 0.9 * (1.0 - it / n_iter) + 0.1
        c += damping * forces
        
        # Clip to bounds
        c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return c


def _local_optimize_smooth(c0, max_time=25.0, beta_start=20.0, beta_end=200.0):
    """
    Local optimization using L-BFGS-B with a smooth surrogate objective.
    Uses annealing of beta to gradually sharpen the soft-min toward the true min,
    followed by an active-set polish for high precision.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Phase 1: Smooth optimization with increasing beta
    betas = [beta_start, 40.0, 80.0, 150.0, beta_end]
    
    for beta in betas:
        if time.time() > start_time + max_time - 8:
            break
        
        def _obj(params):
            return _soft_min_objective(params, beta=beta)
        
        try:
            result = minimize(
                _obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Phase 2: Active-set polish for high precision
    if time.time() < start_time + max_time - 3:
        remaining = max_time - (time.time() - start_time) - 3
        if remaining > 2:
            c_polish, r_polish = _active_set_polish(best_c, max_time=remaining)
            if r_polish > best_r:
                best_c = c_polish
                best_r = r_polish
    
    return best_c, best_r


def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with smooth surrogate."""
    return _local_optimize_smooth(c0, max_time=max_time)


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    # Initialize with hexagonal pattern
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Optimize hexagonal initialization with generous time
    if time.time() < deadline - 30:
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(50.0, deadline - time.time() - 25))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Try force-directed initialization from random starts
    np.random.seed(123)
    for trial in range(3):
        if time.time() > deadline - 35:
            break
        
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        # Apply force-directed relaxation
        c_fd = _force_directed_init(c_rand, n_iter=200, k=0.005)
        r_fd = _compute_radius(c_fd)
        
        if r_fd > best_radius:
            best_radius = r_fd
            best_c = c_fd.copy()
        
        if time.time() < deadline - 25:
            c_opt, r_opt = _local_optimize_smooth(c_fd, max_time=min(30.0, deadline - time.time() - 15))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Final active-set polish on the best solution found
    if best_c is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 3:
            c_polish, r_polish = _active_set_polish(best_c, max_time=remaining)
            if r_polish > best_radius:
                best_radius = r_polish
                best_c = c_polish.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
