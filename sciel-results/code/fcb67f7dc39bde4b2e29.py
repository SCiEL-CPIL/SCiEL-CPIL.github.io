import numpy as np
import time
from scipy.optimize import minimize

N = 100
iu = np.triu_indices(N, k=1)
NUM_PAIRS = len(iu[0])

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_wall_dist, min_pair_dist / 2.0)


def _soft_min_objective_and_grad(x, beta):
    """
    Objective: soft-min approximation of the radius constraints.
    We minimize G(x) = (1/beta) * ln(sum_k exp(-beta * c_k(x)))
    where c_k are the constraint margins (wall distances and pair distances/2).
    As beta -> inf, G(x) -> -min_k c_k(x). Minimizing G maximizes min c_k.
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: c_w = 0.5 - |c_i|
    wall_vals = 0.5 - np.abs(c)  # (N, 3)
    wall_vals_flat = wall_vals.flatten()  # (3N,)
    
    # Pair constraints: c_p = dist_ij / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (N, N, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, N)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists[iu] / 2.0  # (num_pairs,)
    
    all_vals = np.concatenate([wall_vals_flat, pair_vals])
    
    # Compute weights w_k = exp(-beta * c_k) / S
    # To avoid overflow/underflow, subtract max
    neg_beta_c = -beta * all_vals
    max_neg_beta_c = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_neg_beta_c)
    S = np.sum(exp_vals)
    weights = exp_vals / S  # (num_constraints,)
    
    # Objective value
    obj = (max_neg_beta_c + np.log(S)) / beta
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Wall gradient: dG/dc[i,k] = w_{i,k} * sign(c[i,k])
    wall_weights = weights[:3*N].reshape(-1, 3)
    grad += wall_weights * np.sign(c)
    
    # Pair gradient: dG/dc_i = - sum_j w_{ij} * (c_i - c_j) / dist_ij
    pair_weights = weights[3*N:]
    
    # Efficiently compute pair gradient
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    unit_diff = diff * inv_dists[:, :, np.newaxis]  # (N, N, 3)
    
    # Create weight matrix (N, N)
    w_mat = np.zeros((N, N))
    w_mat[iu] = pair_weights
    w_mat[iu[1], iu[0]] = pair_weights  # symmetric
    
    # grad[i] = - sum_j w_mat[i,j] * unit_diff[i,j]
    grad_pairs = -np.sum(w_mat[:, :, np.newaxis] * unit_diff, axis=1)
    grad += grad_pairs
    
    return obj, grad.ravel()


def _optimize_soft_min(x0, deadline, beta_start=10.0, beta_end=5000.0, n_steps=6):
    """
    Run L-BFGS-B optimization with increasing beta (annealing).
    """
    best_x = x0.copy()
    best_r = _compute_radius(x0.reshape(-1, 3))
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                lambda x: _soft_min_objective_and_grad(x, beta)[0],
                best_x,
                jac=lambda x: _soft_min_objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
            )
            
            new_x = np.clip(res.x, -0.5, 0.5)
            new_r = _compute_radius(new_x.reshape(-1, 3))
            
            if new_r > best_r:
                best_r = new_r
                best_x = new_x
        except Exception:
            pass
    
    return best_x, best_r


def _force_directed_refinement(centers, n_iters=200, deadline=None):
    """
    Force-directed method: each sphere repels others and is attracted to center.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Repulsion from other spheres
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Strong repulsion when close
        mask = dists < 2.5 * r_current
        force_mag = np.zeros_like(dists)
        force_mag[mask] = 1.0 / (dists[mask] ** 2 + 1e-8)
        
        unit_diff = np.zeros_like(diff)
        unit_diff[mask] = diff[mask] / dists[mask, :, np.newaxis]
        
        forces = np.sum(force_mag[:, :, np.newaxis] * unit_diff, axis=1)
        
        # Attraction to center (weak)
        forces -= 0.001 * c
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_mask = wall_dists < r_current
        wall_force = np.zeros_like(c)
        
        for i in range(N):
            for k in range(3):
                if wall_mask[i, k]:
                    wall_force[i, k] = -np.sign(c[i, k]) * (r_current - wall_dists[i, k])
        
        forces += wall_force
        
        # Adaptive step size
        step = 0.01 / (1.0 + it * 0.01)
        c += step * forces
        c = np.clip(c, -0.5, 0.5)
        
        new_r = _compute_radius(c)
        if new_r > r_current:
            r_current = new_r
    
    return c, r_current


def _init_grid():
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.375, 0.375, 4)
    return np.array([(x[i], y[j], z[k]) for i in range(5) for j in range(5) for k in range(4)])


def _init_fcc_like():
    """FCC-like initialization with 100 points."""
    centers = _init_grid()
    np.random.seed(42)
    centers += np.random.normal(0, 0.02, centers.shape)
    return centers


def _init_random(seed=0):
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (N, 3))


def _init_halton():
    """Halton sequence initialization."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    centers = np.zeros((N, 3))
    for i in range(N):
        for j in range(3):
            centers[i, j] = 2 * halton(i + 1, [2, 3, 5][j]) - 1
    return centers


def _init_fcc_dense():
    """Denser FCC-like lattice initialization."""
    # Try to create a more structured initial packing
    centers = []
    # Use a 4x4x4 grid with FCC offsets
    a = 1.0 / 4.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Corner
                centers.append([i*a - 0.5 + a/2, j*a - 0.5 + a/2, k*a - 0.5 + a/2])
                # Face centers
                if i < 3:
                    centers.append([(i+0.5)*a - 0.5 + a/2, j*a - 0.5 + a/2, k*a - 0.5 + a/2])
                if j < 3:
                    centers.append([i*a - 0.5 + a/2, (j+0.5)*a - 0.5 + a/2, k*a - 0.5 + a/2])
                if k < 3:
                    centers.append([i*a - 0.5 + a/2, j*a - 0.5 + a/2, (k+0.5)*a - 0.5 + a/2])
    
    # Take first 100
    centers = np.array(centers[:100])
    np.random.seed(123)
    centers += np.random.normal(0, 0.01, centers.shape)
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s margin
    
    # Multiple initializations
    inits = [
        _init_grid(),
        _init_fcc_like(),
        _init_halton(),
        _init_fcc_dense(),
        _init_random(0),
    ]
    
    best_r = -1
    best_centers = None
    
    for i, init_centers in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Allocate remaining time divided among remaining inits
        remaining_inits = len(inits) - i
        time_per_init = min(25, (deadline - 10 - time.time()) / max(1, remaining_inits))
        
        # Phase 1: Soft-min optimization
        if time.time() < deadline - 5:
            centers, r = _optimize_soft_min(init_centers, time.time() + time_per_init * 0.6)
        else:
            centers, r = init_centers, _compute_radius(init_centers)
        
        # Phase 2: Force-directed refinement
        if time.time() < deadline - 3:
            centers_fd, r_fd = _force_directed_refinement(
                centers, 
                n_iters=100, 
                deadline=time.time() + min(5, (deadline - 3 - time.time()))
            )
            if r_fd > r:
                centers, r = centers_fd, r_fd
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    if best_centers is None:
        best_centers = _init_grid()
    
    return best_centers
