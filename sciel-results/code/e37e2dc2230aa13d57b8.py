import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n_points, rotation=0.0):
    """Generate a hexagonal lattice of points inside the unit circle."""
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    k = int(1.0 / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if rotation != 0.0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        points = points @ R.T
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    
    return points

def _radial_ring_init(n_points):
    """Initialize points on concentric rings."""
    points = []
    remaining = n_points
    ring_idx = 0
    current_count = 1
    points.append([0.0, 0.0])
    remaining -= 1
    
    while remaining > 0:
        ring_idx += 1
        count_in_ring = 6 * ring_idx
        
        if count_in_ring > remaining:
            count_in_ring = remaining
            
        r = np.sqrt(ring_idx / (n_points / 6.0)) * 0.9
        if r > 0.95:
             r = 0.95
             
        for i in range(count_in_ring):
            angle = 2 * np.pi * i / count_in_ring
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
            
        remaining -= count_in_ring
        
    points = np.array(points)
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient.
    x is a flat array of shape (2*N,).
    """
    C = x.reshape(N, 2)
    
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    L_safe = np.where(L > 1e-12, L, 1.0)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs = L_safe[i_idx, j_idx] / 2.0
    
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    x_all = np.concatenate([x_pairs, x_walls])
    
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    grad = np.zeros_like(C)
    
    P_pairs = P[:len(i_idx)]
    D_flat = D[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    U = D_flat / L_flat_safe[:, np.newaxis]
    W = P_pairs * 0.5
    
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    P_walls = P[len(i_idx):]
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    grad += P_walls[:, np.newaxis] * U_walls
    
    return -val, -grad

def _epigraph_obj_and_grad(x, beta):
    """
    Epigraph formulation: minimize t + (1/beta)*log(sum(exp(-beta*(x_i - t))))
    where x_i are the constraint margins (pair distances/2 and wall distances).
    x is flat (2N+1,): first 2N are coords, last is t.
    """
    C = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    L_safe = np.where(L > 1e-12, L, 1.0)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs = L_safe[i_idx, j_idx] / 2.0
    
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    x_all = np.concatenate([x_pairs, x_walls])
    
    # Soft-min of (x_i - t)
    shifted = x_all - t
    # Clip for numerical stability
    shifted = np.clip(shifted, -50.0/beta, 50.0/beta)
    exp_terms = np.exp(-beta * shifted)
    Z = np.sum(exp_terms)
    
    obj = t + (1.0/beta) * np.log(Z)
    
    # Gradient
    P = exp_terms / Z
    
    grad_C = np.zeros_like(C)
    grad_t = 1.0 - np.sum(P)  # d/dt [t + (1/beta)log(Z)] = 1 + (1/beta)*sum(-beta*P) = 1 - sum(P)
    
    P_pairs = P[:len(i_idx)]
    D_flat = D[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    U = D_flat / L_flat_safe[:, np.newaxis]
    W = P_pairs * 0.5
    
    np.add.at(grad_C, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad_C, j_idx, -W[:, np.newaxis] * U)
    
    P_walls = P[len(i_idx):]
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    grad_C += P_walls[:, np.newaxis] * U_walls
    
    grad = np.concatenate([grad_C.flatten(), [grad_t]])
    
    return obj, grad

def _expand_symmetric(params):
    """
    Expand symmetric parameters (25 points in a 30-degree sector) to 300 points.
    params: shape (25, 2) -> [r, theta]
    """
    n_orbits = 25
    symmetry = 12
    points = []
    for i in range(n_orbits):
        r, theta = params[i]
        # Generate 12 copies
        for k in range(symmetry):
            angle = theta + 2 * np.pi * k / symmetry
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
    return np.array(points)

def _symmetric_obj(params, beta):
    """
    Objective for symmetric optimization.
    params: shape (50,) flat (r1, t1, r2, t2, ...)
    """
    P = params.reshape(25, 2)
    # Ensure angles are within [0, pi/6] for uniqueness, though optimizer handles bounds
    # r must be >= 0
    r = np.maximum(P[:, 0], 0.0)
    theta = P[:, 1]
    
    points = _expand_symmetric(np.column_stack([r, theta]))
    
    # Compute soft-min score
    # Reuse logic from _soft_min_obj_and_grad but without gradient for now (use numerical)
    # Or just compute true score for simplicity if beta is high?
    # Let's use soft-min for smoothness.
    
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    L_safe = np.where(L > 1e-12, L, 1.0)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs = L_safe[i_idx, j_idx] / 2.0
    
    L_wall = np.linalg.norm(points, axis=1)
    x_walls = 1.0 - L_wall
    
    x_all = np.concatenate([x_pairs, x_walls])
    
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    return -val

def _optimize_symmetric(time_limit):
    """
    Optimize using 12-fold symmetry assumption.
    """
    start_time = time.time()
    
    # Initialize 25 points in the sector [0, pi/6]
    # Simple initialization: hexagonal lattice points within the sector
    # We can generate a dense hex lattice and pick the ones in the sector
    # Or just place them in a grid.
    
    # Let's use a grid initialization in polar coords
    # 25 points. 5 rings of 5?
    # r values: 0.1, 0.3, 0.5, 0.7, 0.9
    # theta values: 0, 0.1, 0.2, 0.3, 0.4 (rad) ~ 0, 5.7, 11.4, 17.1, 22.8 deg
    # This covers the sector [0, 30 deg].
    
    r_vals = np.linspace(0.1, 0.9, 5)
    theta_vals = np.linspace(0.0, np.pi/6, 5)
    
    R, T = np.meshgrid(r_vals, theta_vals)
    R_flat = R.flatten()
    T_flat = T.flatten()
    
    # Ensure we have exactly 25 points
    if len(R_flat) > 25:
        R_flat = R_flat[:25]
        T_flat = T_flat[:25]
    elif len(R_flat) < 25:
        # Pad with random points in sector
        diff = 25 - len(R_flat)
        R_pad = np.random.uniform(0.1, 0.9, diff)
        T_pad = np.random.uniform(0, np.pi/6, diff)
        R_flat = np.concatenate([R_flat, R_pad])
        T_flat = np.concatenate([T_flat, T_pad])
        
    params_init = np.column_stack([R_flat, T_flat]).flatten()
    
    # Bounds
    # r in [0, 1], theta in [0, pi/6]
    bounds = []
    for i in range(25):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, np.pi/6))
        
    # Optimize
    best_params = params_init.copy()
    best_score = -_symmetric_obj(params_init, 100.0)
    
    # Annealing beta
    betas = [100, 500, 2000, 10000]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
            
        def obj(x):
            return _symmetric_obj(x, beta)
            
        try:
            res = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-10})
            new_params = res.x
            new_score = -_symmetric_obj(new_params, beta)
            
            if new_score > best_score:
                best_score = new_score
                best_params = new_params.copy()
                
        except Exception:
            pass
            
    # Convert to points
    best_points = _expand_symmetric(best_params.reshape(25, 2))
    
    # Verify validity
    if not np.all(np.isfinite(best_points)):
        return None, 0.0
        
    true_score = _compute_true_score(best_points)
    if true_score < 0: # Should not happen
        return None, 0.0
        
    return best_points, true_score

def _optimize_soft_min(points_init, time_limit):
    """
    Optimize the packing using soft-min with annealing beta.
    """
    start_time = time.time()
    best_points = points_init.copy()
    best_score = _compute_true_score(points_init)
    current_points = points_init.copy()
    
    betas = [10, 50, 200, 1000, 5000, 20000, 100000]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
            
        def obj_grad(x):
            return _soft_min_obj_and_grad(x, beta)
        
        try:
            max_iter = 200
            res = minimize(obj_grad, current_points.flatten(), method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12})
            
            new_points = res.x.reshape(N, 2)
            
            if not np.all(np.isfinite(new_points)):
                continue
                
            new_score = _compute_true_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                
            if np.all(np.isfinite(new_points)):
                current_points = new_points.copy()
                
        except Exception:
            pass
            
    return best_points, best_score

def _optimize_epigraph(points_init, time_limit):
    """
    Optimize using epigraph formulation with auxiliary variable t.
    """
    start_time = time.time()
    best_points = points_init.copy()
    best_score = _compute_true_score(points_init)
    
    # Initial t: slightly below the current min constraint
    current_score = _compute_true_score(points_init)
    t_init = current_score - 0.001
    
    betas = [50, 200, 1000, 5000, 20000, 100000]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
        
        def obj_grad(x):
            return _epigraph_obj_and_grad(x, beta)
        
        # x = [coords (2N), t]
        x_init = np.concatenate([best_points.flatten(), [t_init]])
        
        # Bounds: coords can be anything reasonable, t in [0, 1]
        bounds = [(None, None)] * (2*N) + [(0.0, 1.0)]
        
        try:
            max_iter = 300
            res = minimize(obj_grad, x_init, method='L-BFGS-B', jac=True,
                          bounds=bounds,
                          options={'maxiter': max_iter, 'gtol': 1e-9, 'ftol': 1e-12})
            
            new_x = res.x
            new_points = new_x[:2*N].reshape(N, 2)
            
            if not np.all(np.isfinite(new_points)):
                continue
                
            new_score = _compute_true_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                t_init = new_score - 0.001
                
        except Exception:
            pass
            
    return best_points, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    initializers = [
        lambda: _hex_lattice_init(N, rotation=0.0),
        lambda: _hex_lattice_init(N, rotation=np.pi/12),
        lambda: _radial_ring_init(N)
    ]
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Standard soft-min optimization on multiple initializers
    for i, init_func in enumerate(initializers):
        elapsed = time.time() - start_time
        if i == 0:
            slice_time = 35.0
        elif i == 1:
            slice_time = 35.0
        else:
            slice_time = time_limit - (time.time() - start_time) - 30.0
            
        if slice_time < 5.0:
            break
            
        try:
            points_init = init_func()
            points_opt, score_opt = _optimize_soft_min(points_init, slice_time)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                
        except Exception:
            continue
            
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_true_score(best_points)
        
    # Phase 2: Epigraph optimization (auxiliary variable t) with remaining time
    elapsed = time.time() - start_time
    if elapsed < time_limit - 15.0:
        try:
            ep_points, ep_score = _optimize_epigraph(best_points, time_limit - elapsed - 5.0)
            if ep_score > best_score:
                best_score = ep_score
                best_points = ep_points
        except Exception:
            pass

    # Phase 3: Final high-precision polish with very high beta
    elapsed = time.time() - start_time
    if elapsed < time_limit - 3.0:
        polish_time = time_limit - elapsed - 2.0
        if polish_time > 5.0:
            beta_final = 200000
            def obj_grad_final(x):
                return _soft_min_obj_and_grad(x, beta_final)
                
            try:
                res = minimize(obj_grad_final, best_points.flatten(), method='L-BFGS-B', jac=True,
                              options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-15})
                new_points = res.x.reshape(N, 2)
                new_score = _compute_true_score(new_points)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
            except Exception:
                pass

    # Phase 4: One more epigraph polish if time allows
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5.0:
        try:
            ep2_points, ep2_score = _optimize_epigraph(best_points, time_limit - elapsed - 2.0)
            if ep2_score > best_score:
                best_score = ep2_score
                best_points = ep2_points
        except Exception:
            pass

    if not np.all(np.isfinite(best_points)):
        best_points = _hex_lattice_init(N)
        
    dists = np.linalg.norm(best_points, axis=1)
    if np.max(dists) > 1.0:
        scale = 0.99 / np.max(dists)
        best_points = best_points * scale
        
    return best_points
