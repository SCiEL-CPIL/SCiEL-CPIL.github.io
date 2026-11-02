import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """Initialize points in a hexagonal lattice."""
    # Better estimate for hex lattice spacing
    # Area per hex cell = s^2 * sqrt(3)/2, total area = pi
    # n * s^2 * sqrt(3)/2 = pi => s = sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    # Add a small margin to ensure enough points
    s *= 0.98
    max_k = int(1.0 / (s * 0.5)) + 3
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    if len(candidates) < n:
        # Try slightly larger spacing
        s *= 1.02
        max_k = int(1.0 / (s * 0.5)) + 4
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        # Select points closest to center, but ensure good spread
        radii = np.linalg.norm(candidates, axis=1)
        # Sort by radius and take first n
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        # Fallback to random
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    # Small perturbation to break symmetry
    points += np.random.normal(0, 1e-5, points.shape)
    return points

def _ring_init(n):
    """Initialize points in concentric rings with better spacing."""
    points = []
    # Estimate ring spacing based on hex lattice
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    num_rings = int(1.0 / s) + 1
    ring_counts = []
    
    for k in range(num_rings):
        if k == 0:
            count = 1
        else:
            # Circumference of ring k is 2*pi*k*s, spacing is 2*s
            # So count = circumference / spacing = pi*k
            count = int(np.pi * k)
            count = max(1, count)
        ring_counts.append(count)
    
    # Adjust to get exactly n points
    diff = n - sum(ring_counts)
    if diff > 0:
        # Add points to outer rings
        for i in range(len(ring_counts)-1, -1, -1):
            if diff <= 0:
                break
            add = min(diff, max(1, ring_counts[i]//4))
            ring_counts[i] += add
            diff -= add
    elif diff < 0:
        # Remove points from outer rings
        for i in range(len(ring_counts)-1, -1, -1):
            if diff >= 0:
                break
            remove = min(-diff, ring_counts[i]-1)
            ring_counts[i] -= remove
            diff += remove
    
    # Ensure sum is exactly n
    while sum(ring_counts) > n:
        for i in range(len(ring_counts)-1, -1, -1):
            if ring_counts[i] > 1:
                ring_counts[i] -= 1
                break
    while sum(ring_counts) < n:
        ring_counts[-1] += 1
    
    idx = 0
    for k, count in enumerate(ring_counts):
        if k == 0:
            if count >= 1:
                points.append([0.0, 0.0])
                idx += 1
        else:
            radius = k * s
            phase = (k % 2) * np.pi / count
            for m in range(count):
                if idx >= n:
                    break
                angle = 2 * np.pi * m / count + phase
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                if x*x + y*y <= 1.0:
                    points.append([x, y])
                    idx += 1
    
    points = np.array(points)
    if len(points) < n:
        # Fill remaining with random points
        while len(points) < n:
            x, y = np.random.uniform(-0.95, 0.95, 2)
            if x*x + y*y <= 1.0:
                points = np.vstack([points, [[x, y]]])
    elif len(points) > n:
        points = points[:n]
    
    points += np.random.normal(0, 1e-5, points.shape)
    return points

def _random_init(n):
    """Random initialization inside unit circle."""
    points = []
    while len(points) < n:
        p = np.random.uniform(-1, 1, 2)
        if np.linalg.norm(p) < 1.0:
            points.append(p)
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 148.0
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        vecs = p[i_idx] - p[j_idx]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
        
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        log_sum_exp = max_z + np.log(sum_exp)
        val = log_sum_exp / beta
        
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        coeff = w_pairs / (2.0 * safe_pair_dists)
        
        np.add.at(grad, i_idx, -coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, coeff[:, np.newaxis] * vecs)
        
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = p / safe_norms[:, np.newaxis]
        
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        return val, grad

    def active_set_polish(points, deadline):
        """Polish using active set constraints with tighter tolerance."""
        if time.time() > deadline - 5:
            return points
            
        current_r = _compute_score(points)
        p = points
        
        vecs = p[i_idx] - p[j_idx]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Active set: constraints within 0.1% of current minimum (tighter)
        active_pairs = np.where(pair_dists < current_r * 1.001)[0]
        active_bounds = np.where(bound_dists < current_r * 1.001)[0]
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            return points
            
        def active_objective_and_grad(x):
            pp = x.reshape(-1, 2)
            
            if len(active_pairs) > 0:
                a_vecs = pp[i_idx[active_pairs]] - pp[j_idx[active_pairs]]
                a_pair_dists = np.sqrt(np.sum(a_vecs**2, axis=1)) / 2.0
            else:
                a_pair_dists = np.array([])
                a_vecs = np.empty((0, 2))
            
            if len(active_bounds) > 0:
                a_norms = np.linalg.norm(pp[active_bounds], axis=1)
                a_bound_dists = 1.0 - a_norms
            else:
                a_bound_dists = np.array([])
            
            beta_active = 200000.0  # Higher beta for tighter fit
            z_pairs = -beta_active * a_pair_dists if len(a_pair_dists) > 0 else np.array([-np.inf])
            z_bounds = -beta_active * a_bound_dists if len(a_bound_dists) > 0 else np.array([-np.inf])
            
            all_z = np.concatenate([z_pairs, z_bounds])
            max_z = np.max(all_z)
            exp_all = np.exp(all_z - max_z)
            sum_exp = np.sum(exp_all)
            log_sum_exp = max_z + np.log(sum_exp)
            val = log_sum_exp / beta_active
            
            grad = np.zeros_like(pp)
            w_all = exp_all / sum_exp
            
            if len(active_pairs) > 0:
                w_pairs = w_all[:len(a_pair_dists)]
                safe_pd = np.where(a_pair_dists > 1e-12, a_pair_dists, 1.0)
                coeff = w_pairs / (2.0 * safe_pd)
                np.add.at(grad, i_idx[active_pairs], -coeff[:, np.newaxis] * a_vecs)
                np.add.at(grad, j_idx[active_pairs], coeff[:, np.newaxis] * a_vecs)
            
            if len(active_bounds) > 0:
                w_bounds = w_all[len(a_pair_dists):]
                safe_bn = np.where(a_norms > 1e-12, a_norms, 1.0)
                bound_grad_vecs = pp[active_bounds] / safe_bn[:, np.newaxis]
                grad[active_bounds] += w_bounds[:, np.newaxis] * bound_grad_vecs
            
            return val, grad
        
        try:
            res = minimize(
                lambda x: active_objective_and_grad(x)[0],
                points.flatten(),
                jac=lambda x: active_objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,  # More iterations
                    'gtol': 1e-15,    # Tighter tolerance
                    'ftol': 1e-17,
                    'maxls': 200,
                    'disp': False
                }
            )
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > _compute_score(points):
                return candidate
        except Exception:
            pass
            
        return points

    best_points = None
    best_score = -np.inf
    
    inits = [_hex_lattice_init(N), _ring_init(N), _random_init(N)]
    
    # Adaptive beta schedule: start lower, increase gradually
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0, 10000.0, 50000.0, 100000.0]
    
    for init_points in inits:
        if time.time() > deadline - 40:
            break
            
        points = init_points
        score = _compute_score(points)
        
        if score > best_score:
            best_points = points.copy()
            best_score = score
            
        for beta in betas:
            if time.time() > deadline - 25:
                break
                
            try:
                # Adjust maxiter and gtol based on beta
                if beta >= 50000:
                    maxiter = 800
                    gtol = 1e-13
                elif beta >= 10000:
                    maxiter = 600
                    gtol = 1e-12
                elif beta >= 1000:
                    maxiter = 500
                    gtol = 1e-11
                elif beta >= 100:
                    maxiter = 400
                    gtol = 1e-10
                else:
                    maxiter = 300
                    gtol = 1e-9
                    
                res = minimize(
                    lambda x: objective_and_grad(x, beta)[0],
                    points.flatten(),
                    jac=lambda x: objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': maxiter,
                        'gtol': gtol,
                        'ftol': 1e-16,
                        'maxls': 100,
                        'disp': False
                    }
                )
                
                candidate = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                
                if candidate_score > score:
                    points = candidate
                    score = candidate_score
                    
                if score > best_score:
                    best_points = points.copy()
                    best_score = score
                    
            except Exception:
                pass
    
    # Final polish on the best solution
    if best_points is not None and time.time() < deadline - 10:
        polished = active_set_polish(best_points, deadline)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
            
    # Multiple perturbation rounds if time permits
    if best_points is not None and time.time() < deadline - 15:
        for perturb_scale in [1e-5, 5e-6, 1e-6]:
            if time.time() > deadline - 10:
                break
            perturbed = best_points + np.random.normal(0, perturb_scale, best_points.shape)
            # Quick optimize with high beta
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, 100000.0)[0],
                    perturbed.flatten(),
                    jac=lambda x: objective_and_grad(x, 100000.0)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-13, 'ftol': 1e-16, 'disp': False}
                )
                candidate = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
            except Exception:
                pass
    
    # Final active set polish
    if best_points is not None and time.time() < deadline - 5:
        polished = active_set_polish(best_points, deadline)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
            
    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        
    best_score = _compute_score(best_points)
    
    return best_points
