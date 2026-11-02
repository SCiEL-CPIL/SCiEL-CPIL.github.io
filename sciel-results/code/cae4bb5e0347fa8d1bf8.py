import numpy as np
import time
from scipy.optimize import minimize

N = 100
_IU = np.triu_indices(N, k=1)

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)

    # Pairwise distances (only upper triangle)
    i, j = _IU
    diff = centers[i] - centers[j]
    dists_sq = np.einsum('ij,ij->i', diff, diff)
    min_pair_dist = np.sqrt(np.min(dists_sq))

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
    
    # 2. Pairwise distances / 2 (upper triangle only)
    i, j = iu
    diff = centers[i] - centers[j]
    dists_sq = np.einsum('ij,ij->i', diff, diff)
    pair_vals = np.sqrt(dists_sq) / 2.0
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_initial_grid(n=100, noise_scale=0.003, seed=42, shape=(5, 5, 4)):
    """
    Build a well-distributed grid initialization.
    Uses the given shape grid (default 5x5x4 = 100 points).
    """
    nx, ny, nz = shape
    # Use full extent with small margin to allow optimization to push to boundaries
    margin = 0.01
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers[:n]

def _build_bcc_init(n=100, noise_scale=0.003, seed=42):
    """
    Build a body-centered cubic initialization.
    BCC has better packing density than simple cubic.
    For a unit cube, place points at corners and face/body centers.
    """
    rng = np.random.default_rng(seed)
    centers = []
    
    # Simple cubic grid with spacing ~1/5
    s = 1.0 / 5.0
    for ix in range(6):
        for iy in range(6):
            for iz in range(6):
                x = -0.5 + ix * s
                y = -0.5 + iy * s
                z = -0.5 + iz * s
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    centers.append([x, y, z])
    
    # Add body centers of each sub-cell
    for ix in range(5):
        for iy in range(5):
            for iz in range(5):
                x = -0.5 + (ix + 0.5) * s
                y = -0.5 + (iy + 0.5) * s
                z = -0.5 + (iz + 0.5) * s
                centers.append([x, y, z])
    
    centers = np.array(centers[:n])
    if len(centers) < n:
        # Pad with random points if needed
        extra = rng.uniform(-0.49, 0.49, (n - len(centers), 3))
        centers = np.vstack([centers, extra])
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(n=100, seed=42):
    """
    Build a Halton sequence based initialization for good space-filling properties.
    """
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 3))
    for i in range(1, n+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _polish(centers, iu, deadline):
    """
    Final high-precision polish pass using very high beta soft-min
    with tight tolerances and many iterations.
    Extended range up to 50000 for finer radius improvements.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try several high-beta values with increasing precision
    betas = [3000.0, 8000.0, 15000.0, 30000.0, 50000.0]
    
    for beta in betas:
        if time.time() > deadline - 1.5:
            break
        
        try:
            res = minimize(
                _soft_min_objective,
                best_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 300}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    # Final micro-adjustment: nudge each sphere away from its nearest neighbor
    if time.time() < deadline - 0.5:
        try:
            for _ in range(3):
                new_centers = best_centers.copy()
                for i in range(N):
                    # Find nearest neighbor
                    dists_sq = np.sum((best_centers - best_centers[i])**2, axis=1)
                    dists_sq[i] = np.inf
                    nn = np.argmin(dists_sq)
                    dist = np.sqrt(dists_sq[nn])
                    if dist < 2 * best_radius * 1.001:
                        # Nudge away from nearest neighbor
                        direction = (best_centers[i] - best_centers[nn]) / (dist + 1e-15)
                        new_centers[i] += direction * 1e-6
                        # Also nudge away from walls
                        for k in range(3):
                            if abs(new_centers[i, k]) > 0.5 - best_radius * 1.001:
                                new_centers[i, k] = np.sign(new_centers[i, k]) * (0.5 - best_radius * 1.002)
                
                new_centers = np.clip(new_centers, -0.5, 0.5)
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        except Exception:
            pass
    
    return best_centers, best_radius

def _anneal(centers, iu, deadline, initial_temp=0.005, cooling_rate=0.9998):
    """
    Targeted simulated annealing that focuses on improving the minimum constraint.
    Uses bottleneck identification to guide perturbations.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    temp = initial_temp
    
    rng = np.random.default_rng(123)
    stuck_count = 0
    max_stuck = 800
    
    while time.time() < deadline and temp > 0.00005:
        # Compute all constraints to identify bottlenecks
        dists_to_walls = 0.5 - np.abs(best_centers)
        wall_min = np.min(dists_to_walls)
        
        i, j = iu
        diff = best_centers[i] - best_centers[j]
        dists_sq = np.einsum('ij,ij->i', diff, diff)
        pair_dists = np.sqrt(dists_sq) / 2.0
        pair_min = np.min(pair_dists)
        
        # Determine which constraint is the bottleneck
        # Find indices of spheres involved in tight constraints
        candidates = set()
        
        # Wall-bottleneck spheres
        wall_thresh = wall_min + 0.015
        wall_close = np.where(dists_to_walls < wall_thresh)[0]
        candidates.update(wall_close.tolist())
        
        # Pair-bottleneck spheres
        pair_thresh = pair_min + 0.015
        tight_pairs = np.where(pair_dists < pair_thresh)[0]
        for tp in tight_pairs[:10]:  # Limit to avoid too many
            candidates.add(int(i[tp]))
            candidates.add(int(j[tp]))
        
        if len(candidates) == 0:
            candidates = set(rng.choice(N, size=5, replace=False).tolist())
        
        # Perturb 1-3 spheres from the bottleneck set
        n_perturb = int(rng.integers(1, 4))
        cand_list = list(candidates)
        idx = rng.choice(cand_list, size=min(n_perturb, len(cand_list)), replace=False)
        
        new_centers = best_centers.copy()
        # Adaptive step size based on temperature
        step_size = temp * 0.25
        new_centers[idx] += rng.normal(0, step_size, (len(idx), 3))
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        delta = new_radius - current_radius
        if delta > 0:
            best_centers = new_centers
            best_radius = new_radius
            current_radius = new_radius
            stuck_count = 0
        else:
            prob = np.exp(delta / temp)
            if rng.random() < prob:
                best_centers = new_centers
                current_radius = new_radius
                stuck_count += 1
            else:
                stuck_count += 1
        
        temp *= cooling_rate
        
        if stuck_count > max_stuck:
            # Restart with slightly perturbed best
            stuck_count = 0
            temp = max(temp * 0.5, 0.001)
            best_centers = best_centers + rng.normal(0, 0.002, best_centers.shape)
            best_centers = np.clip(best_centers, -0.5, 0.5)
            current_radius = _compute_radius(best_centers)
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute upper triangle indices
    iu = _IU
    
    # Initial guess: Multiple diverse initializations
    best_centers = None
    best_radius = 0.0
    
    inits = []
    
    # Primary grid init (5x5x4 = 100 points)
    inits.append(_build_initial_grid(seed=42))
    inits.append(_build_initial_grid(seed=123))
    inits.append(_build_initial_grid(seed=456))
    
    # Additional diverse grid shapes for more basin coverage
    inits.append(_build_initial_grid(seed=42, shape=(5, 4, 5)))
    inits.append(_build_initial_grid(seed=123, shape=(5, 4, 5)))
    inits.append(_build_initial_grid(seed=42, shape=(6, 5, 4)))
    inits.append(_build_initial_grid(seed=123, shape=(6, 5, 4)))
    
    # BCC init
    inits.append(_build_bcc_init(seed=42))
    inits.append(_build_bcc_init(seed=789))
    
    # Halton init
    inits.append(_build_halton_init(seed=42))
    inits.append(_build_halton_init(seed=789))
    
    # Second random start for diversity
    rng = np.random.default_rng(999)
    random_init = rng.uniform(-0.48, 0.48, (N, 3))
    inits.append(random_init)
    
    for centers in inits:
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    if best_centers is None:
        best_centers = _build_initial_grid()
        best_radius = _compute_radius(best_centers)
        
    # Phase 1: Soft-min optimization with increasing beta
    beta = 15.0
    max_beta = 30000.0
    
    while time.time() < deadline - 40.0 and beta < max_beta:
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 150}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
            
        # Increase beta
        beta *= 1.4
        
    # Phase 2: Annealing to escape local minima
    if time.time() < deadline - 25.0:
        try:
            sa_centers, sa_radius = _anneal(
                best_centers, 
                iu,
                deadline - 25.0,
                initial_temp=0.005,
                cooling_rate=0.9998
            )
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers
        except Exception:
            pass
            
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 3.0:
        try:
            polished, p_radius = _polish(
                best_centers,
                iu,
                deadline - 3.0
            )
            if p_radius > best_radius:
                best_radius = p_radius
                best_centers = polished
        except Exception:
            pass
            
    return best_centers

# EVOLVE-BLOCK-END
