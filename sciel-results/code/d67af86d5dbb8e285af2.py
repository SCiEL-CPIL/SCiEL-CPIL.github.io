import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    # Try multiple rotations and spacings to find a good subset
    for attempt in range(10):
        # Base spacing for packing N circles in radius 1
        # Area per circle ~ pi * r^2. Total area ~ pi.
        # N * pi * r^2 < pi => r < 1/sqrt(N).
        # Spacing d = 2r.
        base_spacing = 2.0 / np.sqrt(n_points) * 0.98
        spacing = base_spacing * (0.97 + 0.01 * attempt)
        
        k = int(1.0 / spacing) + 5
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
        
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                # Hexagonal lattice coordinates
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                
                # Rotate
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                r = np.hypot(x_rot, y_rot)
                
                # Must fit inside unit circle (center distance < 1 - radius)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))

        if len(rows) < n_points:
            continue

        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        
        # Select points closest to center to maximize symmetry and packing density
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        
        # Add small noise for subsequent attempts to break symmetry
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.02, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]

        # Compute current radius
        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        # Fallback: simple grid or hex without rotation
        spacing = 2.0 / np.sqrt(n_points) * 0.9
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]

    return best_pts

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def _softmin_obj_grad(pts, beta):
    """
    Compute soft-min radius and its gradient.
    r_soft = -1/beta * log(sum(exp(-beta * f_k)))
    where f_k are the constraints (distances/2 and boundary distances).
    We want to maximize r_soft, so we minimize -r_soft.
    """
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    bound_dist = 1.0 - norms
    
    # f_k values
    # We only care about the minimum, so large values contribute exp(-beta * large) ~ 0.
    # To avoid underflow/overflow issues in logsumexp, we can shift by min, but 
    # for beta ~ 10-50, and distances ~ 0.05, exp(-beta*d) is very small.
    # It's safer to use scipy.special.logsumexp if available, or manual stable logsumexp.
    
    # Let's collect all f_k
    # Upper triangle of d for pairs
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = d[triu_idx] / 2.0
    
    # Combine
    all_f = np.concatenate([pair_dists, bound_dist])
    
    # Stable logsumexp: log(sum(exp(-beta * f)))
    # = -beta * min_f + log(sum(exp(-beta * (f - min_f))))
    min_f = all_f.min()
    shifted = -beta * (all_f - min_f)
    # Clip to avoid overflow in exp, though shifted <= 0
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        # Fallback, shouldn't happen
        r_soft = min_f
        grad = np.zeros_like(pts)
        return -r_soft, grad

    log_sum = np.log(sum_exp)
    r_soft = -1.0 / beta * (log_sum - beta * min_f)
    # Wait, log(sum(exp(-beta f))) = -beta min_f + log(sum(exp(-beta (f-min_f))))
    # So r_soft = -1/beta * (-beta min_f + log_sum_shifted) = min_f - 1/beta * log_sum_shifted
    
    # Gradient of r_soft w.r.t pts
    # d r_soft / d f_k = -1/beta * d/df_k log(sum(exp(-beta f_m)))
    # = -1/beta * (1/sum_exp) * sum_m (d/df_k exp(-beta f_m))
    # = -1/beta * (1/sum_exp) * (-beta exp(-beta f_k))  (if k is in the sum)
    # = exp(-beta f_k) / sum_exp
    # So d r_soft / d f_k = w_k where w_k = exp(-beta f_k) / sum_exp
    # Note: w_k sums to 1.
    
    weights = exp_shifted / sum_exp
    
    # Split weights back
    n_pairs = len(pair_dists)
    w_pairs = weights[:n_pairs]
    w_bound = weights[n_pairs:]
    
    grad = np.zeros_like(pts)
    
    # Gradient from pairs
    # d r_soft / d pts_i = sum_j w_ij * d f_ij / d pts_i
    # f_ij = ||p_i - p_j|| / 2
    # d f_ij / d p_i = (p_i - p_j) / (2 * ||p_i - p_j||) = (p_i - p_j) / (4 * f_ij) ? No.
    # d ||p_i - p_j|| / d p_i = (p_i - p_j) / ||p_i - p_j||
    # So d f_ij / d p_i = (p_i - p_j) / (2 * ||p_i - p_j||)
    
    # We need to accumulate gradients for each point i from all j != i.
    # Using vectorization:
    # diff is (N, N, 2)
    # d is (N, N)
    # w_pairs is mapped to triu. We need a full weight matrix W (N, N) where W[i,j] = w_ij.
    
    W = np.zeros((n, n))
    W[triu_idx] = w_pairs
    W.T[triu_idx] = w_pairs # Symmetric
    
    # Zero diagonal
    np.fill_diagonal(W, 0.0)
    
    # d f_ij / d p_i = (p_i - p_j) / (2 * d_ij)
    # d_ij is d (with inf on diag).
    # To avoid div by zero, mask where d is inf or 0.
    valid_mask = np.isfinite(d) & (d > 1e-10)
    inv_2d = np.zeros_like(d)
    inv_2d[valid_mask] = 1.0 / (2.0 * d[valid_mask])
    
    # grad_i = sum_j W[i,j] * (p_i - p_j) * inv_2d[i,j]
    # = sum_j W[i,j] * inv_2d[i,j] * p_i - sum_j W[i,j] * inv_2d[i,j] * p_j
    # Let M = W * inv_2d (elementwise).
    M = W * inv_2d
    # sum_j M[i,j]
    row_sums = M.sum(axis=1)
    # sum_j M[i,j] * p_j
    # This is M @ pts
    M_pts = M.dot(pts)
    
    grad_pair = row_sums[:, np.newaxis] * pts - M_pts
    
    # Gradient from boundary
    # f_i = 1 - ||p_i||
    # d f_i / d p_i = -p_i / ||p_i||
    # d r_soft / d p_i += w_bound_i * (-p_i / ||p_i||)
    
    norms_safe = np.where(norms > 1e-10, norms, 1.0)
    grad_bound = -w_bound[:, np.newaxis] * pts / norms_safe[:, np.newaxis]
    
    total_grad = grad_pair + grad_bound
    
    # We want to MINIMIZE -r_soft
    # So objective is -r_soft
    # Gradient is -total_grad
    
    return -r_soft, -total_grad

def _local_polish_softmin(pts, t_init, deadline):
    """
    Polish using soft-min log-sum-exp formulation.
    """
    n_pts = N
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Annealing schedule for beta
    # Start with low beta (smooth), increase to high beta (sharp/min)
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    x0 = best_pts.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        def obj_func(x):
            pts_x = x.reshape(n_pts, 2)
            val, _ = _softmin_obj_grad(pts_x, beta)
            return val
        
        def grad_func(x):
            pts_x = x.reshape(n_pts, 2)
            _, grad = _softmin_obj_grad(pts_x, beta)
            return grad.ravel()
        
        try:
            res = minimize(
                obj_func, 
                x0.ravel(), 
                jac=grad_func,
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 1000,
                    'disp': False
                }
            )
            
            new_pts = res.x.reshape(n_pts, 2)
            
            # Check validity
            if np.all(np.isfinite(new_pts)):
                # Project back inside if slightly outside? 
                # The soft-min penalizes going outside, but might not strictly prevent it.
                norms = np.linalg.norm(new_pts, axis=1)
                if np.any(norms > 1.0):
                    # Hard clip to be safe for the final score, but this might break smoothness
                    # Better to rely on the penalty. If it goes outside, r=0.
                    # Let's check the true radius.
                    pass
                
                true_r = _compute_radius(new_pts)
                if true_r > best_r:
                    best_r = true_r
                    best_pts = new_pts.copy()
                    x0 = new_pts.copy()
        except Exception:
            pass
            
    return best_pts, best_r

def _force_relax(pts, n_steps=1000, deadline=None):
    """Fast force-directed relaxation using cKDTree for neighbor search."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    r_est = best_r

    lr = 0.01
    cooldown = 0.999

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = np.zeros_like(pos)
        target = 2.0 * r_est

        try:
            tree = cKDTree(pos)
            pairs = tree.query_pairs(r=target, output_type='ndarray')
            for i, j in pairs:
                diff = pos[i] - pos[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    continue
                excess = target - d
                if excess > 0:
                    # Soft repulsion
                    force_mag = 1.0 * excess
                    unit_vec = diff / d
                    forces[i] += force_mag * unit_vec
                    forces[j] -= force_mag * unit_vec
        except Exception:
            pass

        # Boundary force
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.05
        close_to_wall = norms > (1.0 - wall_margin)
        if np.any(close_to_wall):
            excess_b = np.maximum(norms[close_to_wall] - (1.0 - wall_margin), 0)
            f_mags = 10.0 * excess_b
            directions = -pos[close_to_wall] / norms[close_to_wall, np.newaxis]
            forces[close_to_wall] += f_mags[:, np.newaxis] * directions

        pos = pos + lr * forces

        # Project back into unit disk
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.999

        if step % 20 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r

        lr *= cooldown

    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_pts = None
    best_r = 0.0

    # Phase 1: Initialization and Coarse Relaxation
    seeds = [42, 123, 7, 99, 256, 777, 1234, 5678, 1111, 2222, 3333, 4444]
    
    for seed in seeds:
        if time.time() > deadline - 40.0:
            break

        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)

        # Quick relaxation to remove initial overlaps
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=500, deadline=deadline - 35.0)
        
        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)

    # Phase 2: Soft-Min Polish
    # Use the best point found so far
    if time.time() < deadline - 10.0:
        polished_pts, polished_r = _local_polish_softmin(best_pts, best_r, deadline=deadline - 2.0)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
    
    # Phase 3: Final High-Precision Polish on the result of Phase 2
    # Run again with even tighter tolerances if time permits
    if time.time() < deadline - 1.0:
        polished_pts2, polished_r2 = _local_polish_softmin(best_pts, best_r, deadline=deadline - 0.5)
        if polished_r2 > best_r:
            best_r = polished_r2
            best_pts = polished_pts2.copy()

    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    
    norms = np.linalg.norm(best_pts, axis=1)
    if not np.all(norms < 1.0) or not np.all(np.isfinite(best_pts)):
        # Fallback
        best_pts = _hex_init(N, seed=0)
        norms = np.linalg.norm(best_pts, axis=1)
        mask = norms < 1.0
        if not np.all(mask):
             # If even fallback fails (shouldn't happen), return zero
             best_pts = np.zeros((N, 2))

    return best_pts
