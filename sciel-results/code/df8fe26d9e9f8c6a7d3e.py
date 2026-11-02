# EVOLVE-BLOCK-START

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

def _hex_lattice_init(n, jitter_scale=1e-4):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    points = []
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    max_k = int(1.0 / (s * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 3
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    points += np.random.normal(0, jitter_scale, points.shape)
    return points

def _ring_init(n):
    """
    Initialize points in concentric rings with optimized angular phases.
    """
    points = []
    num_rings = int(np.sqrt(n * 2 / np.sqrt(3)))
    
    ring_counts = []
    remaining = n
    for k in range(num_rings):
        if k == 0:
            count = 1
        else:
            count = int(2 * np.pi * k / (2 * np.sqrt(3) * np.sqrt(n)))
            count = max(1, count)
        ring_counts.append(count)
    
    diff = n - sum(ring_counts)
    if diff > 0:
        for i in range(len(ring_counts)-1, -1, -1):
            add = min(diff, max(1, ring_counts[i]//3))
            ring_counts[i] += add
            diff -= add
            if diff <= 0:
                break
    elif diff < 0:
        for i in range(len(ring_counts)-1, -1, -1):
            remove = min(-diff, ring_counts[i]-1)
            ring_counts[i] -= remove
            diff += remove
            if diff >= 0:
                break
    
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
            radius = k * np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
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
        while len(points) < n:
            x, y = np.random.uniform(-0.9, 0.9, 2)
            if x*x + y*y <= 1.0:
                points = np.vstack([points, [[x, y]]])
    elif len(points) > n:
        points = points[:n]
    
    points += np.random.normal(0, 1e-4, points.shape)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    # Generate multiple diverse initializations
    inits = [
        _hex_lattice_init(N, jitter_scale=1e-4),
        _hex_lattice_init(N, jitter_scale=1e-3),
        _ring_init(N)
    ]
    
    best_points = None
    best_score = -np.inf
    
    # Evaluate initial scores
    init_scores = []
    for init_points in inits:
        score = _compute_score(init_points)
        init_scores.append((score, init_points))
        if score > best_score:
            best_score = score
            best_points = init_points.copy()

    # Sort initializations by score
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        i_idx, j_idx = iu
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

    def run_optimization(points, betas, max_iter_per_beta, time_limit):
        current_points = points.copy()
        current_score = _compute_score(current_points)
        
        for beta in betas:
            if time.time() > time_limit:
                break
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, beta)[0],
                    current_points.flatten(),
                    jac=lambda x: objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': max_iter_per_beta,
                        'gtol': 1e-12,
                        'ftol': 1e-15,
                        'maxls': 50,
                        'disp': False
                    }
                )
                candidate = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                
                if candidate_score > current_score:
                    current_points = candidate
                    current_score = candidate_score
            except Exception:
                pass
        return current_points, current_score

    # Optimize the top 2 initializations
    for score, init_points in init_scores[:2]:
        if time.time() > deadline - 30:
            break
        
        # Adaptive beta schedule
        betas = [50.0, 200.0, 1000.0, 5000.0, 20000.0, 100000.0]
        
        opt_points, opt_score = run_optimization(init_points, betas, 300, deadline - 10)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points

    # Final polish with very high beta
    if time.time() < deadline - 5 and best_points is not None:
        try:
            final_beta = 500000.0
            res = minimize(
                lambda x: objective_and_grad(x, final_beta)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, final_beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'gtol': 1e-14,
                    'ftol': 1e-16,
                    'maxls': 100,
                    'disp': False
                }
            )
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
        except Exception:
            pass

    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
