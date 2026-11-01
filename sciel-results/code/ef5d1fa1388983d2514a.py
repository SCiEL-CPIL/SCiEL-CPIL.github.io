"""Packing 200 equal circles in a unit square using smooth optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist


def _compute_radius(c):
    """Compute the exact largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = float(np.min(boundary_dists))
    if n < 2:
        return min_boundary
    dists = pdist(c, metric='euclidean')
    min_pair_dist = float(np.min(dists))
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_radius(c, beta):
    """
    Compute soft approximation of min radius using log-sum-exp.
    Returns a smooth approximation to min(all constraint terms).
    """
    # Boundary terms: distance from each point to each of 4 walls
    b = np.column_stack([
        0.5 - c[:, 0],  # left
        0.5 + c[:, 0],  # right
        0.5 - c[:, 1],  # bottom
        0.5 + c[:, 1],  # top
    ])  # shape (N, 4)
    
    # Pairwise terms: half distances
    d = pdist(c, metric='euclidean')
    p = 0.5 * d  # shape (N*(N-1)/2,)
    
    # Combine all terms
    all_terms = np.concatenate([b.ravel(), p])
    
    # Soft min via log-sum-exp
    m = np.min(all_terms)
    # For stability, shift by m
    shifted = all_terms - m
    # exp(-beta * shifted) where shifted >= 0
    # For large beta, only terms close to m contribute
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    soft_min = m - (1.0 / beta) * log_sum
    
    return soft_min


def _objective_vec(x, beta):
    """
    Objective: minimize -t where t is the soft-min radius.
    x = [c_1x, c_1y, ..., c_Nx, c_Ny, t]
    We use epigraph: maximize t subject to t <= all constraint terms (smoothed).
    Equivalent: minimize -t + penalty for t exceeding soft constraints.
    
    Actually, let's just use: minimize -soft_min_radius(c)
    where c = x[:2N].reshape(N,2)
    """
    c = x[:2 * N].reshape(N, 2)
    r = _soft_min_radius(c, beta)
    return -r


def _objective_vec_grad(x, beta):
    """
    Objective and gradient for L-BFGS-B.
    """
    c = x[:2 * N].reshape(N, 2)
    n = c.shape[0]
    
    # Compute soft min and its gradient
    b = np.column_stack([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1],
    ])
    d = pdist(c, metric='euclidean')
    p = 0.5 * d
    
    all_terms = np.concatenate([b.ravel(), p])
    m = np.min(all_terms)
    shifted = all_terms - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    # Soft min value
    soft_min = m - (1.0 / beta) * np.log(sum_exp)
    
    # Gradient of soft min w.r.t. each term:
    # d(soft_min)/d(term_k) = -exp(-beta*(term_k - m)) / sum_exp
    weights = -exp_vals / sum_exp  # shape (4N + N(N-1)/2,)
    
    # Now compute gradient w.r.t. c
    grad_c = np.zeros_like(c)
    
    # Boundary contributions
    # term indices 0..4N-1 correspond to b.ravel()
    # b[i, 0] = 0.5 - c[i,0] -> d/d(c[i,0]) = -1
    # b[i, 1] = 0.5 + c[i,0] -> d/d(c[i,0]) = +1
    # b[i, 2] = 0.5 - c[i,1] -> d/d(c[i,1]) = -1
    # b[i, 3] = 0.5 + c[i,1] -> d/d(c[i,1]) = +1
    w_b = weights[:4 * N].reshape(N, 4)
    grad_c[:, 0] += -w_b[:, 0] + w_b[:, 1]
    grad_c[:, 1] += -w_b[:, 2] + w_b[:, 3]
    
    # Pairwise contributions
    # p[k] = 0.5 * ||c_i - c_j|| for pair (i,j)
    # d(p_k)/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # d(p_k)/d(c_j) = 0.5 * (c_j - c_i) / ||c_i - c_j||
    w_p = weights[4 * N:]
    
    # Build pair index mapping
    # pdist ordering: for n points, pairs are (0,1),(0,2),...,(0,n-1),(1,2),...,(n-2,n-1)
    # We need to map each pair index to (i, j)
    pair_count = n * (n - 1) // 2
    if pair_count > 0 and len(w_p) > 0:
        # Compute pairwise differences and distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (n, n, 2)
        dist_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist_mat, 1.0)  # avoid div by zero
        
        # Unit vectors
        unit_vec = diff / dist_mat[:, :, np.newaxis]
        
        # For each pair (i,j) with i<j, contribution to grad_c[i] and grad_c[j]
        # Pair index in pdist: we need to enumerate
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if idx < len(w_p):
                    w = w_p[idx]
                    # p = 0.5 * dist, dp/dc_i = 0.5 * (c_i - c_j)/dist
                    grad_c[i] += w * 0.5 * unit_vec[i, j]
                    grad_c[j] += w * 0.5 * unit_vec[j, i]
                    idx += 1
    
    # Objective is -soft_min, so gradient is -grad(soft_min)
    grad_obj = -grad_c.ravel()
    
    return -soft_min, grad_obj


def _objective_with_grad(x, beta):
    """Wrapper for L-BFGS-B that returns (f, g)."""
    c = x[:2 * N].reshape(N, 2)
    n = c.shape[0]
    
    b = np.column_stack([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1],
    ])
    d = pdist(c, metric='euclidean')
    p = 0.5 * d
    
    all_terms = np.concatenate([b.ravel(), p])
    m = np.min(all_terms)
    shifted = all_terms - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    soft_min = m - (1.0 / beta) * np.log(sum_exp)
    
    weights = -exp_vals / sum_exp
    grad_c = np.zeros_like(c)
    
    w_b = weights[:4 * N].reshape(N, 4)
    grad_c[:, 0] += -w_b[:, 0] + w_b[:, 1]
    grad_c[:, 1] += -w_b[:, 2] + w_b[:, 3]
    
    w_p = weights[4 * N:]
    pair_count = n * (n - 1) // 2
    if pair_count > 0 and len(w_p) > 0:
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist_mat, 1.0)
        unit_vec = diff / dist_mat[:, :, np.newaxis]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if idx < len(w_p):
                    w = w_p[idx]
                    grad_c[i] += w * 0.5 * unit_vec[i, j]
                    grad_c[j] += w * 0.5 * unit_vec[j, i]
                    idx += 1
    
    grad_obj = -grad_c.ravel()
    # Pad gradient to full x size (x has 2N+1 elements, but we only optimize first 2N)
    full_grad = np.zeros(len(x))
    full_grad[:2 * N] = grad_obj
    
    return -soft_min, full_grad


def _initialize_hexagonal(seed_offset=0):
    """Initialize with a hexagonal pattern with optional perturbation."""
    best_c = None
    best_r = 0.0
    
    np.random.seed(42 + seed_offset)
    
    # Try multiple hexagonal spacings
    for d in np.linspace(0.054, 0.072, 15):
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
            # Take first N points, possibly with slight random perturbation
            c = points[:N].copy()
            # Add small random perturbation to break symmetry
            c += np.random.normal(0, 0.002, size=(N, 2))
            c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _initialize_grid_perturbed(seed_offset=0):
    """Initialize with a perturbed grid."""
    np.random.seed(123 + seed_offset)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    c = grid[:N].copy()
    c += np.random.normal(0, 0.003, size=(N, 2))
    c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
    return c


def _optimize_smooth(c0, beta_schedule, max_time=30.0):
    """
    Multi-stage smooth optimization.
    beta_schedule: list of beta values to use in sequence.
    """
    start_time = time.time()
    c = c0.copy()
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (2 * N)
    
    for beta in beta_schedule:
        if time.time() - start_time > max_time - 2:
            break
        
        x0 = c.flatten()
        
        def obj_and_grad(x):
            return _objective_with_grad(x, beta)
        
        try:
            res = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 100,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxfun': 200,
                }
            )
            c = res.x[:2 * N].reshape(N, 2)
            c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        except Exception:
            pass
    
    return c


def _polish_high_beta(c0, max_time=10.0):
    """Final polish with very high beta for precision."""
    start_time = time.time()
    c = c0.copy()
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (2 * N)
    
    # Use very high beta for near-exact min
    beta = 2000.0
    x0 = c.flatten()
    
    def obj_and_grad(x):
        return _objective_with_grad(x, beta)
    
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 50,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': 100,
            }
        )
        c = res.x[:2 * N].reshape(N, 2)
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
    except Exception:
        pass
    
    return c


def _local_refinement(c0, max_time=5.0):
    """
    Local refinement: identify active constraints and do a focused optimization.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    
    # Try small random perturbations and re-optimize
    np.random.seed(777)
    best_c = c.copy()
    best_r = r
    
    for _ in range(3):
        if time.time() - start_time > max_time - 1:
            break
        
        c_pert = c + np.random.normal(0, 0.001, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Quick optimize with moderate beta
        c_opt = _optimize_smooth(c_pert, [100.0, 500.0], max_time=2.0)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    return best_c


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (120s limit, 10s buffer)
    
    best_c = None
    best_r = 0.0
    
    # Stage 1: Multiple initializations with multi-stage smooth optimization
    init_configs = [
        _initialize_hexagonal(seed_offset=0),
        _initialize_hexagonal(seed_offset=1),
        _initialize_grid_perturbed(seed_offset=0),
    ]
    
    # Beta schedule: coarse to fine
    beta_schedule = [20.0, 50.0, 100.0, 200.0, 500.0]
    
    for i, c_init in enumerate(init_configs):
        if time.time() > deadline - 20:
            break
        
        r_init = _compute_radius(c_init)
        if r_init > best_r:
            best_r = r_init
            best_c = c_init.copy()
        
        # Time budget per config
        remaining = deadline - time.time() - 15
        if remaining < 5:
            break
        time_per_config = min(25.0, remaining / max(1, len(init_configs) - i))
        
        c_opt = _optimize_smooth(c_init, beta_schedule, max_time=time_per_config)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Stage 2: Local refinement
    if best_c is not None and time.time() < deadline - 10:
        remaining = deadline - time.time() - 8
        if remaining > 3:
            c_ref = _local_refinement(best_c, max_time=min(5.0, remaining))
            r_ref = _compute_radius(c_ref)
            if r_ref > best_r:
                best_r = r_ref
                best_c = c_ref.copy()
    
    # Stage 3: High-precision polish
    if best_c is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 2:
            c_pol = _polish_high_beta(best_c, max_time=min(5.0, remaining))
            r_pol = _compute_radius(c_pol)
            if r_pol > best_r:
                best_r = r_pol
                best_c = c_pol.copy()
    
    # Final: ensure validity
    if best_c is None:
        best_c = _initialize_hexagonal(seed_offset=0)
    
    best_c = np.clip(best_c, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_c
# EVOLVE-BLOCK-END
