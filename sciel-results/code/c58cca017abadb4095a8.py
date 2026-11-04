"""
Optimized packer for 100 equal spheres in a unit cube.
Combines violation-energy BFGS/L-BFGS-B optimization with soft-min smoothing,
multi-start initializations, contact-graph refinement, and a final Adam-based
gradient descent pass with a learnable temperature for the soft-min surrogate.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair_dist = np.min(dists[:, 1])
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _energy_and_grad(x, target_r):
    """
    Energy for a fixed target radius: sum of squared violations of
    pairwise separation (2*r) and wall distance (r).
    Returns (energy, gradient_flat).
    """
    c = x.reshape(N, 3)

    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)

    active = dists < 2.0 * target_r
    active[np.diag_indices_from(active)] = False

    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    energy_pair = 0.5 * np.sum(violations ** 2)

    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]

    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)

    wall_dists = HALF - np.abs(c)
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    energy_wall = np.sum(wall_violations ** 2)

    signs = np.sign(c)
    grad_wall = 2.0 * wall_violations * signs

    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()

    return total_energy, total_grad

def _soft_min_radius_and_grad(x, beta):
    """
    Smooth soft-min radius surrogate.
    Returns (energy, gradient) where energy = -soft_min_radius.
    """
    c = x.reshape(N, 3)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e10)
    
    iu = np.triu_indices(N, k=1)
    pair_radii = dists[iu] / 2.0
    
    wall_dists = HALF - np.abs(c)
    wall_radii = wall_dists.flatten()
    
    all_radii = np.concatenate([pair_radii, wall_radii])
    
    max_r = np.max(all_radii)
    shifted = -beta * (all_radii - max_r)
    
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 500)
    
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = max_r - (1.0 / beta) * log_sum
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros_like(c)
    
    pair_weights = weights[:len(pair_radii)]
    
    w_mat = np.zeros((N, N))
    w_mat[iu] = pair_weights
    
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_diffs = diff / safe_dists[:, :, np.newaxis]
    
    grad_pair = -0.5 * np.sum(w_mat[:, :, np.newaxis] * unit_diffs, axis=1)
    
    wall_weights = weights[len(pair_radii):].reshape(N, 3)
    signs = np.sign(c)
    grad_wall = wall_weights * signs
    
    total_grad = (grad_pair + grad_wall).flatten()
    energy = -soft_min_val
    
    return energy, total_grad

def _local_optimize_violation(init_pts, target_r, deadline, max_iter=2000):
    """
    L-BFGS-B minimization of the violation energy for a fixed target radius.
    """
    def objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)

    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts

def _optimize_soft_min(init_pts, beta_start=10.0, beta_end=80.0, n_beta_steps=5, deadline=None, max_iter_per_beta=100):
    """
    Optimize using soft-min radius objective with annealing.
    """
    if deadline is None:
        deadline = time.time() + 30.0
    
    x = init_pts.flatten().copy()
    
    for beta in np.linspace(beta_start, beta_end, n_beta_steps):
        if time.time() > deadline - 1.0:
            break
        
        def objective(x):
            if time.time() > deadline:
                raise TimeoutError()
            return _soft_min_radius_and_grad(x, beta)
        
        try:
            result = minimize(
                objective,
                x,
                method='BFGS',
                jac=True,
                options={'maxiter': max_iter_per_beta, 'gtol': 1e-10, 'disp': False}
            )
            x = result.x
        except (TimeoutError, Exception):
            break
    
    opt_pts = x.reshape(N, 3)
    return opt_pts

def _adam_soft_min_optimize(init_pts, beta_start=5.0, beta_end=100.0, n_steps=500, lr=0.001, deadline=None):
    """
    Adam gradient descent with annealed beta for soft-min radius.
    This provides a differentiable, smooth path through the landscape.
    """
    if deadline is None:
        deadline = time.time() + 30.0
    
    x = init_pts.flatten().copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    for step in range(n_steps):
        if time.time() > deadline - 0.5:
            break
        
        # Anneal beta
        beta = beta_start + (beta_end - beta_start) * (step / max(n_steps - 1, 1))
        beta = max(beta, 1.0)
        
        # Adaptive learning rate
        lr_t = lr / (1.0 + 0.01 * step)
        
        energy, grad = _soft_min_radius_and_grad(x, beta)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))
        
        x = x - lr_t * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project back to bounds
        x = np.clip(x, -HALF + 1e-9, HALF - 1e-9)
    
    opt_pts = x.reshape(N, 3)
    return opt_pts

def _contact_graph_refine(centers, radius, deadline):
    """
    Identify near-contact pairs and refine only the involved spheres.
    """
    if time.time() > deadline - 3.0:
        return centers, _compute_radius(centers)

    centers = centers.copy()
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.1 * radius)

    if len(pairs) == 0:
        return centers, _compute_radius(centers)

    involved = set()
    for i, j in pairs:
        involved.add(i)
        involved.add(j)
    involved = sorted(involved)

    if len(involved) < 2:
        return centers, _compute_radius(centers)

    idx_array = np.array(involved)
    n_involved = len(involved)
    
    def sub_objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(n_involved, 3)
        
        # Compute min distance among involved pairs and to walls
        min_d = np.inf
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
        
        # Also check distances from involved to non-involved
        if n_involved < N:
            non_involved = np.setdiff1d(np.arange(N), idx_array)
            for i in idx_array:
                for j in non_involved:
                    d = np.linalg.norm(c[i] - c[j])
                    if d < min_d:
                        min_d = d
        
        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        min_wall = np.min(wall)
        
        return -(min_d / 2.0 + min_wall)

    def sub_grad(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(n_involved, 3)
        
        # Find the binding constraint
        min_d = np.inf
        min_pair = None
        min_type = 'pair'
        
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
                min_pair = (i, j)
                min_type = 'pair'
        
        if n_involved < N:
            non_involved = np.setdiff1d(np.arange(N), idx_array)
            for i in idx_array:
                for j in non_involved:
                    d = np.linalg.norm(c[i] - c[j])
                    if d < min_d:
                        min_d = d
                        min_pair = (i, j)
                        min_type = 'pair'
        
        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        min_wall_idx = np.argmin(wall)
        min_wall_val = wall[min_wall_idx]
        
        grad_full = np.zeros((N, 3))
        
        # If wall is the binding constraint
        if min_wall_val < min_d / 2.0:
            i_wall = idx_array[min_wall_idx]
            abs_c = np.abs(c[i_wall])
            k_max = np.argmax(abs_c)
            grad_full[i_wall, k_max] += -np.sign(c[i_wall, k_max])
        elif min_pair is not None:
            i, j = min_pair
            d = c[i] - c[j]
            dist = min_d
            if dist > 1e-12:
                grad_full[i] += d / (2.0 * dist)
                grad_full[j] -= d / (2.0 * dist)
        
        return grad_full[idx_array].flatten()

    x0 = centers[idx_array].flatten()
    bounds_sub = [(-HALF + 1e-9, HALF - 1e-9)] * (n_involved * 3)

    try:
        result = minimize(
            sub_objective,
            x0,
            jac=sub_grad,
            method='L-BFGS-B',
            bounds=bounds_sub,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_sub = result.x.reshape(n_involved, 3)
        centers[idx_array] = new_sub
    except TimeoutError:
        pass
    except Exception:
        pass

    centers = np.clip(centers, -HALF + 1e-9, HALF - 1e-9)
    return centers, _compute_radius(centers)

def _generate_grid():
    """Generate a 5x4x5 grid initialization."""
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points[:N].copy()

def _generate_bcc():
    """Generate a BCC-like initialization."""
    points = []
    for ix in range(5):
        for iy in range(4):
            for iz in range(5):
                x = (ix + 0.5) / 5.0 - 0.5
                y = (iy + 0.5) / 4.0 - 0.5
                z = (iz + 0.5) / 5.0 - 0.5
                points.append([x, y, z])
    return np.array(points[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_points = None
    best_r = 0.0
    
    rng = np.random.default_rng(42)
    
    # Generate initializations
    initializations = []
    grid_centers = _generate_grid()
    initializations.append(grid_centers)
    
    bcc_centers = _generate_bcc()
    initializations.append(bcc_centers)
    
    # Perturbed versions
    for i in range(4):
        base = _generate_grid()
        noise_scale = 0.02 + 0.01 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)
    
    # Random initializations
    for i in range(2):
        random_c = rng.uniform(-0.45, 0.45, size=(N, 3))
        initializations.append(random_c)
    
    # Target radii to try (progressively increasing)
    target_radii = [0.108, 0.110, 0.112, 0.114, 0.116, 0.118]
    
    # Phase 1: Multi-start with violation-energy L-BFGS-B optimization
    for init in initializations:
        if time.time() > deadline - 40.0:
            break
        
        for target_r in target_radii:
            if time.time() > deadline - 35.0:
                break
            
            noise = rng.normal(0, 0.002, size=(N, 3))
            init_pts = np.clip(init + noise, -0.49, 0.49)
            
            remaining = deadline - time.time() - 5.0
            if remaining < 3.0:
                break
            
            opt_pts = _local_optimize_violation(init_pts, target_r, deadline)
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                r_opt = _compute_radius(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_points = opt_pts.copy()
    
    # Phase 2: Soft-min optimization with annealing
    if best_points is not None and time.time() < deadline - 30.0:
        for trial in range(3):
            if time.time() > deadline - 25.0:
                break
            
            noise = rng.normal(0, 0.005, size=(N, 3))
            init_pts = np.clip(best_points + noise, -0.49, 0.49)
            
            remaining = deadline - time.time() - 10.0
            if remaining < 5.0:
                break
            
            opt_pts = _optimize_soft_min(
                init_pts, 
                beta_start=5.0, 
                beta_end=60.0, 
                n_beta_steps=4,
                deadline=deadline - 5.0,
                max_iter_per_beta=150
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                r_opt = _compute_radius(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_points = opt_pts.copy()
    
    # Phase 3: Adam-based soft-min optimization (new differentiable approach)
    if best_points is not None and time.time() < deadline - 20.0:
        for trial in range(2):
            if time.time() > deadline - 15.0:
                break
            
            noise = rng.normal(0, 0.003, size=(N, 3))
            init_pts = np.clip(best_points + noise, -0.49, 0.49)
            
            remaining = deadline - time.time() - 8.0
            if remaining < 5.0:
                break
            
            opt_pts = _adam_soft_min_optimize(
                init_pts,
                beta_start=5.0,
                beta_end=80.0,
                n_steps=300,
                lr=0.0005,
                deadline=deadline - 5.0
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                r_opt = _compute_radius(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_points = opt_pts.copy()
    
    # Phase 4: Squeeze phase - incremental radius increase
    if best_points is not None:
        num_squeezes = 15
        for i in range(num_squeezes):
            if time.time() > deadline - 15.0:
                break
            
            success_improved = False
            increments = [0.003, 0.002, 0.001, 0.0005, 0.0002]
            
            for inc in increments:
                if time.time() > deadline - 10.0:
                    break
                
                target_r = best_r * (1.0 + inc)
                
                noise = rng.normal(0, 0.001, size=(N, 3))
                init_pts = np.clip(best_points + noise, -0.49, 0.49)
                
                remaining = deadline - time.time() - 5.0
                if remaining < 2.0:
                    break
                
                opt_pts = _local_optimize_violation(
                    init_pts, 
                    target_r, 
                    deadline
                )
                
                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                    r_opt = _compute_radius(opt_pts)
                    
                    if r_opt > best_r:
                        best_r = r_opt
                        best_points = opt_pts.copy()
                        success_improved = True
                        break
            
            if not success_improved and rng.random() < 0.3:
                noise = rng.normal(0, 0.004, size=(N, 3))
                init_pts = np.clip(best_points + noise, -0.49, 0.49)
                target_r = best_r * 1.002
                remaining = deadline - time.time() - 3.0
                if remaining > 2.0:
                    opt_pts = _local_optimize_violation(
                        init_pts, 
                        target_r, 
                        deadline
                    )
                    if np.all(np.isfinite(opt_pts)):
                        opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                        r_opt = _compute_radius(opt_pts)
                        if r_opt > best_r:
                            best_r = r_opt
                            best_points = opt_pts.copy()
    
    # Phase 5: Contact graph refinement
    if best_points is not None and time.time() < deadline - 8.0:
        refined_pts, r_refined = _contact_graph_refine(
            best_points, best_r, deadline
        )
        if r_refined > best_r:
            best_r = r_refined
            best_points = refined_pts.copy()
    
    # Phase 6: Final high-precision polish
    if best_points is not None and time.time() < deadline - 3.0:
        target_r = best_r * 1.0001
        remaining = deadline - time.time() - 1.0
        if remaining > 2.0:
            polished_pts = _local_optimize_violation(
                best_points, 
                target_r, 
                deadline,
                max_iter=3000
            )
            if np.all(np.isfinite(polished_pts)):
                polished_pts = np.clip(polished_pts, -HALF + 1e-9, HALF - 1e-9)
                r_polished = _compute_radius(polished_pts)
                if r_polished > best_r:
                    best_r = r_polished
                    best_points = polished_pts.copy()
    
    if best_points is None:
        best_points = _generate_grid()
    
    return best_points
