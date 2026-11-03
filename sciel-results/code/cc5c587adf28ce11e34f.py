"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _score(c):
    """
    Compute the actual score (largest common radius) for a configuration.
    c: (N, 3) array of centers.
    Returns the radius r.
    """
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check boundaries: min distance to each face
    face_dists = 0.5 - np.abs(c)
    min_face = np.min(face_dists)
    if min_face < -1e-12:
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_pair / 2.0, min_face)
    return r


def _objective_and_grad(x, beta):
    """
    Objective and gradient for optimization.
    x: 1D array of length 3*N
    Returns (obj, grad) where obj = -soft_min(all distances)
    """
    c = x.reshape(N, 3)
    
    # Wall clearances
    wall = 0.5 - np.abs(c)  # shape (N, 3)
    wall_flat = wall.ravel()  # shape (3N,)
    
    # Pairwise distances / 2
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0
    i, j = np.triu_indices(N, k=1)
    d_half_upper = d_half[i, j]  # shape (N*(N-1)/2,)
    
    # Combine all constraints
    vals = np.concatenate([wall_flat, d_half_upper])
    n_vals = len(vals)
    
    # Soft-min via log-sum-exp
    m = np.min(vals)
    shifted = vals - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = np.log(sum_exp)
    soft_min_val = m - log_sum_exp / beta
    
    # Weights for each constraint in the soft-min
    weights = exp_vals / sum_exp  # shape (n_vals,)
    
    # Gradient w.r.t. x
    grad_x = np.zeros_like(c)  # shape (N, 3)
    
    # Wall contribution: dv/dx = -sign(c) for each coordinate
    wall_weights = weights[:3*N]  # shape (3N,)
    wall_weights_reshaped = wall_weights.reshape(N, 3)
    sign_c = np.sign(c)
    grad_x -= wall_weights_reshaped * sign_c
    
    # Pair contribution
    pair_weights = weights[3*N:]  # shape (N*(N-1)/2,)
    
    ci = c[i]  # shape (n_pairs, 3)
    cj = c[j]  # shape (n_pairs, 3)
    diff = ci - cj  # shape (n_pairs, 3)
    norms = np.linalg.norm(diff, axis=1, keepdims=True)  # shape (n_pairs, 1)
    norms = np.maximum(norms, 1e-15)
    
    # dv/d(ci) = diff / (2 * norm)
    grad_ci = diff / (2.0 * norms)  # shape (n_pairs, 3)
    grad_cj = -grad_ci  # shape (n_pairs, 3)
    
    weighted_grad_ci = pair_weights[:, np.newaxis] * grad_ci
    weighted_grad_cj = pair_weights[:, np.newaxis] * grad_cj
    
    np.add.at(grad_x, i, weighted_grad_ci)
    np.add.at(grad_x, j, weighted_grad_cj)
    
    # Objective is -soft_min
    f = -soft_min_val
    g = -grad_x.ravel()
    
    return f, g


def _generate_bcc_lattice(n_points):
    """
    Generate points on a BCC lattice scaled to fit in [-0.5, 0.5]^3.
    """
    n_div = 5
    points = []
    
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([i / n_div, j / n_div, k / n_div])
    
    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                points.append([(i + 0.5) / n_div, (j + 0.5) / n_div, (k + 0.5) / n_div])
    
    points = np.array(points)
    points = points - 0.5
    
    rng = np.random.default_rng(123)
    indices = np.arange(len(points))
    rng.shuffle(indices)
    points_shuffled = points[indices]
    
    selected = [points_shuffled[0]]
    for i in range(1, len(points_shuffled)):
        if len(selected) >= n_points:
            break
        p = points_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.08:
            selected.append(p)
    
    while len(selected) < n_points:
        p = rng.uniform(-0.45, 0.45, size=3)
        selected.append(p)
    
    return np.array(selected[:n_points])


def _generate_fcc_lattice(n_points):
    """
    Generate points on an FCC lattice scaled to fit in [-0.5, 0.5]^3.
    """
    n_div = 4
    points = []
    
    # Corner and face centers
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([i / n_div, j / n_div, k / n_div])
    
    # Edge centers
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div):
                points.append([i / n_div, j / n_div, (k + 0.5) / n_div])
    
    for i in range(n_div + 1):
        for j in range(n_div):
            for k in range(n_div + 1):
                points.append([i / n_div, (j + 0.5) / n_div, k / n_div])
    
    for i in range(n_div):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([(i + 0.5) / n_div, j / n_div, k / n_div])
    
    # Body center
    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                points.append([(i + 0.5) / n_div, (j + 0.5) / n_div, (k + 0.5) / n_div])
    
    points = np.array(points)
    points = points - 0.5
    
    rng = np.random.default_rng(456)
    indices = np.arange(len(points))
    rng.shuffle(indices)
    points_shuffled = points[indices]
    
    selected = [points_shuffled[0]]
    for i in range(1, len(points_shuffled)):
        if len(selected) >= n_points:
            break
        p = points_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.08:
            selected.append(p)
    
    while len(selected) < n_points:
        p = rng.uniform(-0.45, 0.45, size=3)
        selected.append(p)
    
    return np.array(selected[:n_points])


def _generate_grid_init():
    """Generate a simple grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _optimize_continuation(c_init, deadline):
    """
    Run continuation optimization from an initial configuration.
    Returns (best_c, best_val)
    """
    best_c = c_init.copy()
    best_val = _score(best_c)
    
    # Continuation schedule with more stages
    betas = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
        
        def obj_grad(x):
            f, g = _objective_and_grad(x, beta)
            return f, g
        
        try:
            res = minimize(obj_grad, best_c.ravel(), method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8,
                                    'maxls': 50})
            
            new_c = res.x.reshape(N, 3)
            new_c = np.clip(new_c, -0.4995, 0.4995)
            
            new_val = _score(new_c)
            
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
                
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        for beta_polish in [5000, 10000]:
            if time.time() > deadline - 5:
                break
            
            def obj_grad_polish(x):
                f, g = _objective_and_grad(x, beta_polish)
                return f, g
            
            try:
                res = minimize(obj_grad_polish, best_c.ravel(), method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10,
                                        'maxls': 30})
                
                new_c = res.x.reshape(N, 3)
                new_c = np.clip(new_c, -0.49999, 0.49999)
                
                new_val = _score(new_c)
                
                if new_val > best_val:
                    best_val = new_val
                    best_c = new_c.copy()
                    
            except Exception:
                pass
    
    return best_c, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Generate multiple initializations
    inits = []
    
    # BCC initialization
    try:
        c_bcc = _generate_bcc_lattice(N)
        inits.append(c_bcc)
    except Exception:
        pass
    
    # FCC initialization
    try:
        c_fcc = _generate_fcc_lattice(N)
        inits.append(c_fcc)
    except Exception:
        pass
    
    # Grid initialization
    try:
        c_grid = _generate_grid_init()
        inits.append(c_grid)
    except Exception:
        pass
    
    # Random initialization
    rng = np.random.default_rng(42)
    c_rand = rng.uniform(-0.4, 0.4, size=(N, 3))
    inits.append(c_rand)
    
    best_c = inits[0].copy()
    best_score = _score(best_c)
    
    # Optimize from each initialization with continuation
    for c_init in inits:
        if time.time() > deadline - 30.0:
            break
        
        c_opt, score = _optimize_continuation(c_init, deadline)
        if score > best_score:
            best_score = score
            best_c = c_opt.copy()
    
    # Final very high precision polish on the best result
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, 20000.0),
                best_c.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40}
            )
            c_polished = res.x.reshape(N, 3)
            c_polished = np.clip(c_polished, -0.499999, 0.499999)
            
            score = _score(c_polished)
            if score > best_score:
                best_score = score
                best_c = c_polished.copy()
        except Exception:
            pass
    
    return best_c
# EVOLVE-BLOCK-END
