import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _normalize_batch(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fib_sample(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_hull(config):
    """
    Compute covering radius in degrees using convex hull facets.
    Returns (radius_deg, circumcenters, facet_indices).
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([]), np.array([], dtype=int)
    
    simplices = hull.simplices
    n_facets = len(simplices)
    
    p0 = config[simplices[:, 0]]
    p1 = config[simplices[:, 1]]
    p2 = config[simplices[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    c = np.cross(v1, v2)
    c_norms = np.linalg.norm(c, axis=1)
    
    valid_mask = c_norms > 1e-12
    if not np.any(valid_mask):
        return 0.0, np.array([]), np.array([], dtype=int)
    
    c = c[valid_mask]
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    valid_facets = simplices[valid_mask]
    
    dots0 = np.einsum('ij,ij->i', c, p0[valid_mask])
    dots1 = np.einsum('ij,ij->i', c, p1[valid_mask])
    dots2 = np.einsum('ij,ij->i', c, p2[valid_mask])
    
    dots = np.column_stack([dots0, dots1, dots2])
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    angle_spread = np.max(angles, axis=1) - np.min(angles, axis=1)
    valid_circ = angle_spread < 1e-6
    
    if not np.any(valid_circ):
        grid = _fib_sample(10000)
        dots_grid = grid @ config.T
        min_dots = np.min(dots_grid, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle)), np.array([]), np.array([], dtype=int)
    
    c = c[valid_circ]
    valid_facets = valid_facets[valid_circ]
    
    dots_all = c @ config.T
    dots_all = np.clip(dots_all, -1.0, 1.0)
    max_dots = np.max(dots_all, axis=1)
    min_angles = np.arccos(max_dots)
    
    max_angle = np.max(min_angles)
    radius_deg = np.degrees(max_angle)
    
    return radius_deg, c, valid_facets

def _spectral_energy(config):
    """
    Compute spectral energy: maximize trace of Gram matrix = sum of pairwise dot products.
    This is equivalent to minimizing the sum of squared chordal distances.
    Energy = -sum_{i<j} dot(p_i, p_j)
    """
    G = config @ config.T
    # Trace is N (all diagonal are 1), so sum of all entries = N + 2*sum_{i<j} dot
    # We want to maximize sum_{i<j} dot(p_i, p_j)
    # So minimize -sum_{i<j} dot = -(np.sum(G) - N) / 2
    total = np.sum(G) - N  # = 2 * sum_{i<j} dot
    return -total / 2.0

def _spectral_gradient(config):
    """
    Gradient of spectral energy wrt config.
    E = -1/2 * sum_{i<j} dot(p_i, p_j)
    dE/dp_k = -1/2 * sum_{j != k} (-p_j) ... wait let me redo.
    
    E = -1/2 * sum_{i<j} (p_i . p_j)
    dE/dp_k = -1/2 * [sum_{j<k} (-p_j) + sum_{j>k} (-p_j)] 
            = -1/2 * [-sum_{j != k} p_j]
            = 1/2 * sum_{j != k} p_j
    
    But we need to project to tangent space.
    """
    G = config @ config.T
    # sum_{j != k} p_j = sum_j p_j - p_k
    sum_all = np.sum(config, axis=0)
    grad = 0.5 * (sum_all[None, :] - config)  # (N, 3)
    
    # Project to tangent space
    proj = np.einsum('ij,ij->i', grad, config)
    grad = grad - proj[:, np.newaxis] * config
    
    return grad

def _spectral_relax(config, deadline, min_time=10.0):
    """
    Optimize spectral energy using L-BFGS-B.
    """
    x0 = config.flatten()
    
    def fun_grad(x):
        cfg = x.reshape(N, 3)
        cfg = _normalize_batch(cfg)
        val = _spectral_energy(cfg)
        grad = _spectral_gradient(cfg)
        return val, grad.flatten()
    
    best_x = x0
    best_val, _ = fun_grad(x0)
    
    # Run multiple L-BFGS passes
    while time.time() < deadline - min_time:
        res = minimize(fun_grad, best_x, method='L-BFGS-B', jac=True,
                       options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if res.fun < best_val - 1e-12:
            best_val = res.fun
            best_x = res.x
        else:
            break
    
    best_cfg = _normalize_batch(best_x.reshape(N, 3))
    return best_cfg

def _lloyd_step(config, grid):
    """One Lloyd iteration: move each point to centroid of its Voronoi cell."""
    dots = grid @ config.T
    max_idx = np.argmax(dots, axis=1)
    
    new_config = np.zeros_like(config)
    for i in range(N):
        mask = (max_idx == i)
        count = np.sum(mask)
        if count > 0:
            center = np.mean(grid[mask], axis=0)
            new_config[i] = _normalize(center)
        else:
            new_config[i] = config[i]
    
    return new_config

def _lloyd_optimize(config, deadline, min_time=15.0):
    """Run Lloyd's algorithm for several iterations."""
    M = 15000
    grid = _fib_sample(M)
    
    for iteration in range(100):
        if time.time() > deadline - min_time:
            break
        new_config = _lloyd_step(config, grid)
        if np.allclose(config, new_config, atol=1e-8):
            break
        config = new_config
    
    return config

def _local_optimize(config, deadline, min_time_remaining=10.0):
    """
    Local optimization using circumcenter-based critical points.
    """
    config = config.copy()
    current_score, _, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.04
    stagnation = 0
    max_stagnation = 40
    
    while time.time() < deadline - min_time_remaining:
        score, circumcenters, facets = _covering_radius_hull(config)
        
        if len(circumcenters) == 0:
            break
        
        current_score = score
        
        dots_all = circumcenters @ config.T
        dots_all = np.clip(dots_all, -1.0, 1.0)
        max_dots = np.max(dots_all, axis=1)
        min_angles = np.arccos(max_dots)
        
        worst_idx = np.argmax(min_angles)
        worst_point = circumcenters[worst_idx]
        worst_angle = min_angles[worst_idx]
        
        dots_to_worst = np.clip(worst_point @ config.T, -1.0, 1.0)
        sorted_idx = np.argsort(-dots_to_worst)
        
        improved = False
        
        for trial in range(3):
            if trial >= len(sorted_idx):
                break
            idx = sorted_idx[trial]
            
            v = worst_point - config[idx]
            v_proj = v - np.dot(v, config[idx]) * config[idx]
            v_norm = np.linalg.norm(v_proj)
            
            if v_norm < 1e-10:
                continue
            
            v_dir = v_proj / v_norm
            
            for ss in [step_size, step_size * 0.5, step_size * 2.0]:
                new_pt = config[idx] + ss * v_dir
                new_pt = _normalize(new_pt)
                
                old_pt = config[idx].copy()
                config[idx] = new_pt
                
                new_score, _, _ = _covering_radius_hull(config)
                
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    stagnation = 0
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_config = config.copy()
                    break
                else:
                    config[idx] = old_pt
            
            if improved:
                break
        
        if not improved:
            stagnation += 1
            step_size *= 0.85
            if step_size < 0.001 or stagnation > max_stagnation:
                break
    
    return best_config

def _polish(config, deadline, min_time_remaining=5.0):
    """
    High-precision polish pass.
    """
    config = config.copy()
    current_score, _, _ = _covering_radius_hull(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.005
    stagnation = 0
    
    while time.time() < deadline - min_time_remaining:
        score, circumcenters, facets = _covering_radius_hull(config)
        
        if len(circumcenters) == 0:
            break
        
        current_score = score
        
        dots_all = circumcenters @ config.T
        dots_all = np.clip(dots_all, -1.0, 1.0)
        max_dots = np.max(dots_all, axis=1)
        min_angles = np.arccos(max_dots)
        
        threshold = np.max(min_angles) - 0.001
        worst_mask = min_angles > threshold
        worst_indices = np.where(worst_mask)[0]
        
        if len(worst_indices) == 0:
            break
        
        improved = False
        
        for wi in worst_indices[:5]:
            worst_point = circumcenters[wi]
            
            dots_to_worst = np.clip(worst_point @ config.T, -1.0, 1.0)
            sorted_idx = np.argsort(-dots_to_worst)
            
            for trial in range(2):
                if trial >= len(sorted_idx):
                    break
                idx = sorted_idx[trial]
                
                v = worst_point - config[idx]
                v_proj = v - np.dot(v, config[idx]) * config[idx]
                v_norm = np.linalg.norm(v_proj)
                
                if v_norm < 1e-10:
                    continue
                
                v_dir = v_proj / v_norm
                
                new_pt = config[idx] + step_size * v_dir
                new_pt = _normalize(new_pt)
                
                old_pt = config[idx].copy()
                config[idx] = new_pt
                
                new_score, _, _ = _covering_radius_hull(config)
                
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    stagnation = 0
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_config = config.copy()
                    break
                else:
                    config[idx] = old_pt
            
            if improved:
                break
        
        if not improved:
            stagnation += 1
            step_size *= 0.9
            if step_size < 0.0005 or stagnation > 50:
                break
    
    return best_config

def _soft_max_surrogate(config, beta=50.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp on grid points.
    Returns (value, gradient).
    """
    grid = _fib_sample(3000)
    dots = grid @ config.T
    min_dots = np.min(dots, axis=1)
    angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
    
    m = np.max(angles)
    exp_angles = np.exp(beta * (angles - m))
    sum_exp = np.sum(exp_angles)
    val = m + np.log(sum_exp) / beta
    
    w = exp_angles / sum_exp
    
    argmins = np.argmin(dots, axis=1)
    dots_min = dots[np.arange(len(grid)), argmins]
    dots_min_clipped = np.clip(dots_min, -1.0 + 1e-9, 1.0 - 1e-9)
    denom = np.sqrt(1.0 - dots_min_clipped**2)
    
    grad = np.zeros_like(config)
    grad_contrib = (-w / denom)[:, np.newaxis] * grid
    
    np.add.at(grad, argmins, grad_contrib)
    
    grad_proj = np.einsum('ij,ij->i', grad, config)
    grad = grad - grad_proj[:, np.newaxis] * config
    
    return val, grad

def _lbfgs_stage(config, deadline, min_time=15.0):
    """
    L-BFGS optimization with soft-max surrogate.
    """
    x0 = config.flatten()
    
    def fun_grad(x):
        cfg = x.reshape(N, 3)
        cfg = _normalize_batch(cfg)
        val, grad = _soft_max_surrogate(cfg)
        return val, grad.flatten()
    
    best_x = x0
    best_val, _ = fun_grad(x0)
    
    while time.time() < deadline - min_time:
        res = minimize(fun_grad, best_x, method='L-BFGS-B', jac=True,
                       options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-9})
        
        if res.success and res.fun < best_val - 1e-12:
            best_val = res.fun
            best_x = res.x
        else:
            break
    
    best_cfg = _normalize_batch(best_x.reshape(N, 3))
    return best_cfg

def _repulsion_relax(config, deadline, min_time=10.0):
    """
    Add a repulsion force to prevent coincident points and improve uniformity.
    """
    config = config.copy()
    step_size = 0.01
    
    for iteration in range(200):
        if time.time() > deadline - min_time:
            break
        
        # Compute repulsion forces
        forces = np.zeros_like(config)
        for i in range(N):
            for j in range(i + 1, N):
                d = config[i] - config[j]
                dist = np.linalg.norm(d)
                if dist < 0.1:  # Only repel when close
                    f = d / (dist + 1e-10) * (0.1 - dist)
                    forces[i] += f
                    forces[j] -= f
        
        # Project to tangent space
        for i in range(N):
            forces[i] -= np.dot(forces[i], config[i]) * config[i]
        
        # Update
        for i in range(N):
            new_pt = config[i] + step_size * forces[i]
            config[i] = _normalize(new_pt)
        
        # Decrease step size
        step_size *= 0.95
        if step_size < 0.001:
            break
    
    return config

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_config = None
    best_score = float('inf')
    
    # Multiple starting configurations
    starts = [
        _fib_sample(N),
        _fib_sample(N, offset=0.3),
        _fib_sample(N, offset=0.7),
    ]
    
    # Add a random start
    rng = np.random.default_rng(42)
    random_start = rng.standard_normal((N, 3))
    starts.append(_normalize_batch(random_start))
    
    for start_config in starts:
        if time.time() > deadline - 30:
            break
        
        config = start_config.copy()
        
        # Phase 1: Spectral energy optimization (new structural approach)
        if time.time() < deadline - 60:
            config = _spectral_relax(config, deadline, min_time=20.0)
        
        # Phase 2: Lloyd's algorithm
        if time.time() < deadline - 40:
            config = _lloyd_optimize(config, deadline, min_time=20.0)
        
        # Phase 3: Repulsion relaxation
        if time.time() < deadline - 30:
            config = _repulsion_relax(config, deadline, min_time=15.0)
        
        # Phase 4: L-BFGS with soft-max
        if time.time() < deadline - 25:
            config = _lbfgs_stage(config, deadline, min_time=15.0)
        
        # Phase 5: Local optimization
        if time.time() < deadline - 15:
            config = _local_optimize(config, deadline, min_time=10.0)
        
        # Phase 6: Polish
        if time.time() < deadline - 5:
            config = _polish(config, deadline, min_time=3.0)
        
        config = _normalize_batch(config)
        
        # Evaluate
        score, _, _ = _covering_radius_hull(config)
        
        if score < best_score:
            best_score = score
            best_config = config.copy()
    
    if best_config is None:
        best_config = _fib_sample(N)
    
    # Final normalization and validation
    best_config = _normalize_batch(best_config)
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_config[i] - best_config[j]) < 1e-6:
                best_config[j] = best_config[j] + 1e-4 * np.random.randn(3)
                best_config[j] = _normalize(best_config[j])
    
    return best_config
