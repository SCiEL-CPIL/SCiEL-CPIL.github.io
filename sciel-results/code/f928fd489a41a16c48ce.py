import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)


def _soft_min_objective(x_flat, beta):
    """
    Soft-min objective: minimize -(1/beta)*ln(sum(exp(-beta*constraints)))
    to approximate maximizing the minimum constraint value.
    """
    c = x_flat.reshape(-1, 3)
    
    # Wall constraints: 0.5 - |c_ik|
    wall_vals = 0.5 - np.abs(c)
    wall_flat = wall_vals.ravel()
    
    # Pairwise constraints: |c_i - c_j| / 2 for i < j
    dists = cdist(c, c, metric='euclidean')
    iu_i, iu_j = np.triu_indices(N, k=1)
    pair_vals = dists[iu_i, iu_j] / 2.0
    
    all_vals = np.concatenate([wall_flat, pair_vals])
    
    # Soft min: (1/beta) * ln(sum(exp(-beta * all_vals)))
    neg_beta_x = -beta * all_vals
    max_nb = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_nb)
    sum_exp = np.sum(exp_vals)
    obj = (max_nb + np.log(sum_exp)) / beta
    
    return obj


def _soft_min_grad(x_flat, beta):
    """Gradient of the soft-min objective."""
    c = x_flat.reshape(-1, 3)
    
    # Wall constraints
    wall_vals = 0.5 - np.abs(c)
    wall_flat = wall_vals.ravel()
    
    # Pairwise constraints
    dists = cdist(c, c, metric='euclidean')
    iu_i, iu_j = np.triu_indices(N, k=1)
    pair_vals = dists[iu_i, iu_j] / 2.0
    
    all_vals = np.concatenate([wall_flat, pair_vals])
    
    neg_beta_x = -beta * all_vals
    max_nb = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_nb)
    sum_exp = np.sum(exp_vals)
    weights = -exp_vals / sum_exp  # d(obj)/d(all_vals)
    
    # Wall gradient
    wall_weights = weights[:3*N].reshape(N, 3)
    signs = np.sign(c)
    grad_walls = -wall_weights * signs
    
    # Pair gradient
    pair_weights = weights[3*N:]
    grad_pairs = np.zeros_like(c)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    unit_diff = diff * inv_dists[:, :, np.newaxis]
    
    pair_w_mat = np.zeros((N, N))
    pair_w_mat[iu_i, iu_j] = pair_weights
    grad_pairs = np.sum(pair_w_mat[:, :, np.newaxis] * unit_diff, axis=1)
    
    grad = (grad_walls + grad_pairs).ravel()
    return grad


def _optimize_with_soft_min(x0, deadline, beta_schedule=None):
    """Run L-BFGS-B with soft-min objective and increasing beta."""
    best_x = x0.copy()
    best_r = _compute_radius(best_x.reshape(-1, 3))
    
    if beta_schedule is None:
        beta_schedule = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for beta in beta_schedule:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _soft_min_objective,
                best_x,
                args=(beta,),
                jac=_soft_min_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 20}
            )
            new_x = np.clip(res.x, -0.5, 0.5)
            r = _compute_radius(new_x.reshape(-1, 3))
            if r > best_r:
                best_r = r
                best_x = new_x
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    return best_x, best_r


def _init_fcc_distorted(rng):
    """
    Initialize with a distorted FCC-like lattice.
    FCC has packing fraction ~0.74 in infinite space.
    We place points on FCC lattice positions within the cube, then distort.
    """
    # FCC lattice: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) + integer translations
    # Scale to fit in [-0.5, 0.5]^3 with ~100 points
    # Use a 5x5x4 grid as base, then add FCC-like offsets
    
    nx, ny, nz = 5, 5, 4
    centers = np.zeros((N, 3))
    idx = 0
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # Base position
                x = -0.5 + (ix + 0.5) / nx
                y = -0.5 + (iy + 0.5) / ny
                z = -0.5 + (iz + 0.5) / nz
                
                # Add FCC-like perturbation based on position
                # Alternate between two sublattices
                sublattice = (ix + iy + iz) % 2
                if sublattice == 0:
                    offset = np.array([0.02, 0.02, 0.02])
                else:
                    offset = np.array([-0.02, -0.02, -0.02])
                
                centers[idx] = [x + offset[0], y + offset[1], z + offset[2]]
                idx += 1
    
    # Add small random noise
    centers += rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers


def _init_bcc_like(rng):
    """Initialize with BCC-like structure."""
    nx, ny, nz = 5, 5, 4
    centers = np.zeros((N, 3))
    idx = 0
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = -0.5 + (ix + 0.5) / nx
                y = -0.5 + (iy + 0.5) / ny
                z = -0.5 + (iz + 0.5) / nz
                
                # BCC: corner + body center offset
                offset = 0.015 * np.array([1, 1, 1]) * ((ix + iy + iz) % 2 * 2 - 1)
                centers[idx] = [x + offset[0], y + offset[1], z + offset[2]]
                idx += 1
    
    centers += rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_grid_perturbed(rng, nx, ny, nz, noise_scale=0.01):
    """Perturbed grid initialization."""
    xs = np.linspace(-0.5 + 0.03, 0.5 - 0.03, nx)
    ys = np.linspace(-0.5 + 0.03, 0.5 - 0.03, ny)
    zs = np.linspace(-0.5 + 0.03, 0.5 - 0.03, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _local_search_push_apart(centers, n_iters=100, deadline=None):
    """
    Local search: iteratively find the most violated constraint and fix it.
    Uses a more sophisticated approach than simple push-apart.
    """
    c = centers.copy()
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all constraints
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        
        dists = cdist(c, c, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        r_current = min(min_wall, min_pair / 2.0)
        
        # Find the tightest constraint
        if min_wall <= min_pair / 2.0:
            # Wall constraint is tight
            flat_idx = np.argmin(wall_dists)
            pi = flat_idx // 3
            pk = flat_idx % 3
            
            # Move inward
            direction = -np.sign(c[pi, pk])
            if direction == 0:
                direction = 1.0
            c[pi, pk] += direction * 0.003
        else:
            # Pair constraint is tight
            idx = np.argmin(dists)
            i = idx // N
            j = idx % N
            if i == j:
                continue
            
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction = direction / norm
            
            # Push apart, but also consider walls
            step = 0.003
            new_i = c[i] + direction * step
            new_j = c[j] - direction * step
            
            # Check if new positions violate walls
            if np.any(np.abs(new_i) > 0.499) or np.any(np.abs(new_j) > 0.499):
                step *= 0.5
                new_i = c[i] + direction * step
                new_j = c[j] - direction * step
            
            c[i] = np.clip(new_i, -0.499, 0.499)
            c[j] = np.clip(new_j, -0.499, 0.499)
        
        c = np.clip(c, -0.5, 0.5)
    
    return c


def _simulated_annealing(centers, deadline, n_iters=2000):
    """
    Simulated annealing: random perturbations with acceptance based on energy.
    Energy = -radius (we want to maximize radius, so minimize -radius).
    """
    c = centers.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    T = 0.01  # Initial temperature
    T_min = 0.0001
    cooling = 0.999
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Random perturbation
        c_new = c.copy()
        idx = rng_int(0, N - 1)
        c_new[idx] += rng.normal(0, 0.01, 3)
        c_new = np.clip(c_new, -0.5, 0.5)
        
        r_new = _compute_radius(c_new)
        r_old = _compute_radius(c)
        
        delta = r_new - r_old
        
        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, T_min)):
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        
        T = max(T * cooling, T_min)
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    global rng_int
    rng_int = lambda a, b: int(rng.integers(a, b + 1))
    
    best_centers = None
    best_radius = -np.inf
    
    # Generate diverse initializations
    inits = []
    inits.append(_init_fcc_distorted(rng))
    inits.append(_init_bcc_like(rng))
    inits.append(_init_grid_perturbed(rng, 5, 5, 4, 0.01))
    inits.append(_init_grid_perturbed(rng, 4, 5, 5, 0.01))
    inits.append(_init_grid_perturbed(rng, 5, 4, 5, 0.01))
    inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        c = init_centers.copy()
        
        # Local search refinement
        c = _local_search_push_apart(c, n_iters=50, deadline=deadline - 10.0)
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
        
        # Soft-min optimization with multiple betas
        if time.time() > deadline - 12.0:
            break
        
        x_flat = c.flatten()
        beta_schedule = [30.0, 80.0, 200.0, 500.0, 1000.0]
        opt_x, opt_r = _optimize_with_soft_min(x_flat, deadline - 8.0, beta_schedule)
        opt_centers = np.clip(opt_x.reshape(-1, 3), -0.5, 0.5)
        
        # Final local search polish
        opt_centers = _local_search_push_apart(opt_centers, n_iters=30, deadline=deadline - 5.0)
        r_opt = _compute_radius(opt_centers)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = opt_centers.copy()
    
    # Final refinement if time permits
    if best_centers is not None and time.time() < deadline - 10.0:
        c = best_centers.copy()
        c = _local_search_push_apart(c, n_iters=80, deadline=deadline - 5.0)
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
        
        if time.time() < deadline - 5.0:
            x_flat = best_centers.flatten()
            beta_schedule = [500.0, 1500.0, 3000.0]
            opt_x, opt_r = _optimize_with_soft_min(x_flat, deadline - 3.0, beta_schedule)
            opt_centers = np.clip(opt_x.reshape(-1, 3), -0.5, 0.5)
            r_opt = _compute_radius(opt_centers)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = opt_centers
    
    # Fallback
    if best_centers is None:
        best_centers = _init_grid_perturbed(rng, 5, 5, 4, 0.01)
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
