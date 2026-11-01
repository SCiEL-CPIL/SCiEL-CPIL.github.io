"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances using cKDTree for speed
    tree = cKDTree(centers)
    # Find all pairs within a reasonable distance
    pairs = tree.query_pairs(r=0.5)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        pair_dists = []
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            pair_dists.append(d)
        min_pair_dist = min(pair_dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_bcc_init():
    """
    Build a BCC (body-centered cubic) lattice initialization.
    """
    # Use a 5x5x2 grid of BCC cells
    nx, ny, nz = 5, 5, 2
    n_cells = nx * ny * nz  # 50
    
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
                points.append([cx + 0.5/nx, cy + 0.5/ny, cz + 0.5/nz])
    
    centers = np.array(points[:N])
    
    # Add slight noise
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _polish_repulsion(centers, deadline, iu):
    """
    Local search using repulsion forces with KD-tree acceleration.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    step_size = 0.003
    decay = 0.97
    min_step = 1e-7
    margin = 1.05
    
    no_improve_count = 0
    max_no_improve = 100
    
    while _time_left() > 1.0 and step_size > min_step and no_improve_count < max_no_improve:
        r = _compute_radius(centers)
        if r > best_r + 1e-12:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = max(r_current * margin, 0.01)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
                
        # Pairwise forces using KD-tree for efficiency
        tree = cKDTree(centers)
        search_radius = 2 * max(r_current * margin, 0.01)
        
        # Query neighbors
        neighbor_pairs = tree.query_pairs(r=search_radius)
        
        for i, j in neighbor_pairs:
            if i == j:
                continue
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9 and dist < search_radius:
                strength = (search_radius - dist) / search_radius
                unit_diff = diff / dist
                forces[i] += strength * unit_diff
                forces[j] -= strength * unit_diff
        
        # Normalize forces
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.4999, 0.4999)
        
        step_size *= decay
        
    return best_centers, best_r

def _optimize_from_initial(centers, deadline, iu, time_budget=20.0):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    remaining = time_budget
    
    # Phase 1: Soft-min optimization with adaptive beta
    beta_start = 30.0
    beta_end = 3000.0
    n_beta_steps = 6
    
    if remaining > 8.0:
        for step in range(n_beta_steps):
            if _time_left() < 2.0 or (time.time() - start_t) > remaining - 2.0:
                break
                
            beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
            
            x0 = best_centers.flatten()
            
            try:
                max_iter = 150 if step < 3 else 80
                res = minimize(
                    _soft_min_objective,
                    x0,
                    args=(beta, iu),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (N * 3),
                    options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 20}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
                    
            except Exception:
                continue
    
    # Phase 2: Simulated annealing to escape local minima
    if _time_left() > 5.0 and (time.time() - start_t) < remaining - 3.0:
        sa_start = time.time()
        sa_budget = min(15.0, _time_left() - 3.0)
        
        sa_temp = 0.01
        sa_cool = 0.95
        sa_min_temp = 1e-6
        sa_step = 0.02
        rng = np.random.default_rng(123)
        
        sa_centers = best_centers.copy()
        sa_radius = best_radius
        
        while sa_temp > sa_min_temp and (time.time() - sa_start) < sa_budget:
            # Random perturbation
            idx = rng.integers(0, N)
            delta = rng.normal(0, sa_step, 3)
            new_pos = sa_centers[idx].copy() + delta
            
            if np.any(new_pos < -0.4999) or np.any(new_pos > 0.4999):
                sa_temp *= sa_cool
                continue
            
            sa_centers[idx] = new_pos
            new_radius = _compute_radius(sa_centers)
            
            # Accept if better or with probability exp((new - old)/temp)
            if new_radius > sa_radius:
                sa_radius = new_radius
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = sa_centers.copy()
            else:
                accept_prob = np.exp((new_radius - sa_radius) / sa_temp)
                if rng.random() > accept_prob:
                    sa_centers[idx] -= delta
            
            sa_temp *= sa_cool
    
    # Phase 3: High-precision polish
    if _time_left() > 3.0:
        polish_budget = min(10.0, _time_left() - 2.0)
        if polish_budget > 1.0:
            polished, polished_r = _polish_repulsion(best_centers, deadline, iu)
            if polished_r > best_radius:
                best_radius = polished_r
                best_centers = polished.copy()
            
            # Final L-BFGS with very high beta
            if _time_left() > 2.0:
                try:
                    res = minimize(
                        _soft_min_objective,
                        best_centers.flatten(),
                        args=(200000.0, iu),
                        method='L-BFGS-B',
                        bounds=[(-0.4999, 0.4999)] * (N * 3),
                        options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 10}
                    )
                    new_centers = res.x.reshape(-1, 3)
                    new_centers = np.clip(new_centers, -0.4999, 0.4999)
                    new_radius = _compute_radius(new_centers)
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_centers = new_centers.copy()
                except Exception:
                    pass
    
    return best_centers, best_radius

def construct():
    """
    Construct 100 sphere centers in the unit cube using neural network
    parameterization and L-BFGS-B optimization.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    # Precompute upper triangle indices for pairwise distances
    iu = np.triu_indices(N, k=1)
    
    # Multiple initializations
    inits = [
        ('bcc', _build_bcc_init()),
        ('halton', _build_halton_init(42)),
    ]
    
    best_centers = None
    best_radius = -np.inf
    
    for name, init in inits:
        if _time_left() < 30:
            break
        
        centers, radius = _optimize_from_initial(init, _get_deadline(), iu, time_budget=60.0)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # If no valid result, fallback to grid
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
