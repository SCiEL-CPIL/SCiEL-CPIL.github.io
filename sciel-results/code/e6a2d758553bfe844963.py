import numpy as np
import time

N = 100

# Precompute upper-triangle indices for pairwise distances
_iu_i, _iu_j = np.triu_indices(N, k=1)

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)


def _energy_and_min_clearance(centers):
    """
    Compute energy (sum of squared violations) and minimum clearance.
    Energy = sum of max(0, target_r - clearance)^2 for all constraints.
    This gives a smooth, differentiable objective for local optimization.
    """
    c = centers
    # Wall clearances: 0.5 - |c_ik|
    wall_clear = 0.5 - np.abs(c)  # shape (N, 3)
    
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    pair_clear = dists[_iu_i, _iu_j] / 2.0  # shape (N*(N-1)/2,)
    
    all_clear = np.concatenate([wall_clear.ravel(), pair_clear])
    min_clear = np.min(all_clear)
    
    return min_clear, all_clear


def _gradient_ascent_step(centers, target_r, step_size):
    """
    One gradient ascent step to maximize the minimum clearance.
    Uses subgradient of the min function: push points away from the tightest constraints.
    """
    c = centers.copy()
    wall_clear = 0.5 - np.abs(c)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    pair_clear = dists[_iu_i, _iu_j] / 2.0
    
    all_clear = np.concatenate([wall_clear.ravel(), pair_clear])
    min_clear = np.min(all_clear)
    
    # Subgradient: average over all constraints within epsilon of the minimum
    eps = 1e-4
    mask = all_clear <= min_clear + eps
    n_active = np.sum(mask)
    if n_active == 0:
        return c
    
    grad = np.zeros_like(c)
    
    # Wall contributions
    wall_mask = mask[:3*N]
    if np.any(wall_mask):
        wall_g = np.zeros((N, 3))
        signs = np.sign(c)
        # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
        # We want to maximize clearance, so move in direction of +grad(clearance) = -sign
        # Actually we want to increase clearance, so move inward: direction = -sign
        wall_contrib = wall_mask.reshape(N, 3).astype(float) * (-signs)
        grad += wall_contrib / n_active
    
    # Pair contributions
    pair_mask = mask[3*N:]
    if np.any(pair_mask):
        # For each active pair (i,j), gradient w.r.t. c_i is (c_i - c_j)/|c_i-c_j|
        # and w.r.t. c_j is (c_j - c_i)/|c_i-c_j|
        active_pairs = np.where(pair_mask)[0]
        ai = _iu_i[active_pairs]
        aj = _iu_j[active_pairs]
        pair_diff = c[ai] - c[aj]
        pair_dists = dists[ai, aj]
        safe_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        unit_vecs = pair_diff / safe_dists[:, np.newaxis]
        
        for k in range(len(ai)):
            grad[ai[k]] += unit_vecs[k] / n_active
            grad[aj[k]] -= unit_vecs[k] / n_active
    
    c += step_size * grad
    c = np.clip(c, -0.5, 0.5)
    return c


def _annealing(centers, deadline, rng, n_phases=3):
    """
    Multi-phase simulated annealing with adaptive step sizes.
    Phase 1: large steps, high temperature (explore)
    Phase 2: medium steps, medium temperature (refine)
    Phase 3: small steps, low temperature (polish)
    """
    c = centers.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    phases = [
        {'T': 0.05, 'T_min': 0.005, 'step': 0.05, 'cooling': 0.9995, 'iters': 3000},
        {'T': 0.02, 'T_min': 0.002, 'step': 0.02, 'cooling': 0.9997, 'iters': 5000},
        {'T': 0.005, 'T_min': 0.0005, 'step': 0.005, 'cooling': 0.9998, 'iters': 8000},
    ]
    
    for phase in phases:
        if time.time() > deadline:
            break
        
        T = phase['T']
        T_min = phase['T_min']
        step_scale = phase['step']
        cooling = phase['cooling']
        
        for it in range(phase['iters']):
            if time.time() > deadline:
                break
            
            # Perturb 1-3 random points
            n_perturb = rng.integers(1, 4)
            idxs = rng.choice(N, size=n_perturb, replace=False)
            c_new = c.copy()
            c_new[idxs] += rng.normal(0, step_scale, size=(n_perturb, 3))
            c_new = np.clip(c_new, -0.5, 0.5)
            
            r_new = _compute_radius(c_new)
            r_old = best_r  # Compare against best for acceptance
            
            delta = r_new - r_old
            
            # Accept if better, or with probability exp(delta/T)
            if delta > 0 or (T > T_min and rng.random() < np.exp(delta / T)):
                c = c_new
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new.copy()
            
            T = max(T * cooling, T_min)
    
    return best_c, best_r


def _gradient_polish(centers, deadline, rng, n_iters=2000):
    """
    Gradient-based local search: iteratively move points to increase minimum clearance.
    Uses subgradient of the min function with adaptive step size.
    """
    c = centers.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    step_size = 0.01
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        c_new = _gradient_ascent_step(c, best_r, step_size)
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new
            step_size = min(step_size * 1.05, 0.02)  # Slightly increase step
        else:
            # Try smaller step
            step_size *= 0.8
            if step_size < 1e-5:
                step_size = 0.005  # Reset to moderate
                # Add small random perturbation to escape local minimum
                idx = rng.integers(0, N)
                c = c.copy()
                c[idx] += rng.normal(0, 0.01, 3)
                c = np.clip(c, -0.5, 0.5)
    
    return best_c, best_r


def _init_grid(rng, nx, ny, nz, margin=0.02):
    """Simple grid initialization."""
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_fcc(rng):
    """FCC-inspired initialization with two interleaved sublattices."""
    nx, ny, nz = 5, 5, 4
    centers = np.zeros((N, 3))
    idx = 0
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = -0.5 + (ix + 0.5) / nx
                y = -0.5 + (iy + 0.5) / ny
                z = -0.5 + (iz + 0.5) / nz
                
                sublattice = (ix + iy + iz) % 2
                if sublattice == 0:
                    offset = np.array([0.015, 0.015, 0.015])
                else:
                    offset = np.array([-0.015, -0.015, -0.015])
                
                centers[idx] = [x + offset[0], y + offset[1], z + offset[2]]
                idx += 1
    
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_random(rng):
    """Random initialization with minimum separation enforced."""
    centers = np.zeros((N, 3))
    # Place first few points randomly, then use rejection sampling
    for i in range(N):
        for attempt in range(1000):
            p = rng.uniform(-0.48, 0.48, size=3)
            if i == 0:
                centers[i] = p
                break
            # Check distance to existing points
            dists = np.linalg.norm(centers[:i] - p, axis=1)
            if np.min(dists) > 0.08:  # Minimum separation
                centers[i] = p
                break
        else:
            # Fallback: place at random position
            centers[i] = rng.uniform(-0.48, 0.48, size=3)
    
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: 120s limit, 5s buffer
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Generate diverse initializations
    inits = [
        _init_grid(rng, 5, 5, 4, 0.02),
        _init_grid(rng, 4, 5, 5, 0.02),
        _init_grid(rng, 5, 4, 5, 0.02),
        _init_fcc(rng),
        _init_random(rng),
    ]
    
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        c = init_centers.copy()
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
        
        # Annealing phase
        if time.time() > deadline - 8.0:
            break
        c_annealed, r_annealed = _annealing(c, deadline - 5.0, rng)
        if r_annealed > best_radius:
            best_radius = r_annealed
            best_centers = c_annealed.copy()
        
        # Gradient polish phase
        if time.time() > deadline - 5.0:
            break
        c_polished, r_polished = _gradient_polish(c_annealed, deadline - 3.0, rng, n_iters=1500)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = c_polished.copy()
    
    # Final refinement on the best solution
    if best_centers is not None and time.time() < deadline - 8.0:
        c = best_centers.copy()
        # More annealing with lower temperature
        c2, r2 = _annealing(c, deadline - 5.0, rng, n_phases=2)
        if r2 > best_radius:
            best_radius = r2
            best_centers = c2.copy()
        
        # Final gradient polish
        if time.time() < deadline - 4.0:
            c3, r3 = _gradient_polish(best_centers, deadline - 2.0, rng, n_iters=3000)
            if r3 > best_radius:
                best_radius = r3
                best_centers = c3.copy()
    
    # Fallback
    if best_centers is None:
        best_centers = _init_grid(rng, 5, 5, 4, 0.02)
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
