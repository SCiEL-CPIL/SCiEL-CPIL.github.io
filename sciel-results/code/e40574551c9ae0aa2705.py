import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective_and_grad(centers_flat, beta, iu):
    """
    Objective function and gradient for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    Returns (obj, grad).
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
    shifted = neg_beta_x - max_val
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = np.log(sum_exp)
    obj = (max_val + log_sum) / beta
    
    # Gradient of soft-min: -softmax(-beta * x)
    # d/dx [1/beta * ln(sum(exp(-beta*x)))] = -exp(-beta*x) / sum(exp(-beta*x))
    weights = exp_shifted / sum_exp  # softmax(-beta * x)
    
    grad_all = -weights
    
    # Build gradient for centers
    grad = np.zeros_like(centers_flat)
    
    # Wall part: indices 0 to N*3 - 1
    # wall_vals[i] = 0.5 - |center[i//3, i%3]|
    # d(wall_vals[i])/d(centers_flat[i]) = -sign(center[i//3, i%3])
    wall_grad = grad_all[:N*3].reshape(-1, 3)
    # For each coordinate, derivative of (0.5 - |x|) w.r.t x is -sign(x)
    # If x == 0, use 0
    signs = np.sign(centers)
    signs[np.abs(centers) < 1e-12] = 0.0
    grad += (wall_grad * signs).flatten()
    
    # Pair part: indices N*3 to end
    # pair_vals_unique[i] = dists[iu[0][i], iu[1][i]] / 2
    # d(dist)/d(x_i) = (x_i - x_j) / dist
    # d(dist/2)/d(x_i) = (x_i - x_j) / (2 * dist)
    # d(dist/2)/d(x_j) = (x_j - x_i) / (2 * dist)
    
    pair_grad = grad_all[N*3:]
    iu_i, iu_j = iu
    
    for k in range(len(iu_i)):
        i = iu_i[k]
        j = iu_j[k]
        d = dists[i, j]
        if d < 1e-12:
            continue
        w = pair_grad[k]
        # grad contribution to center i
        diff = (centers[i] - centers[j]) / (2.0 * d)
        grad[3*i:3*i+3] += w * diff
        # grad contribution to center j
        grad[3*j:3*j+3] += w * (-diff)
    
    return obj, grad

def _build_grid_init(nx, ny, nz, noise_scale=0.0):
    """Build a grid initialization."""
    if nx * ny * nz != N:
        return None
        
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if noise_scale > 0:
        rng = np.random.default_rng(int(np.sum(np.round(centers*1000))) % 10000)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_bcc_init():
    """
    Build a BCC lattice initialization.
    BCC unit cell has 2 points: (0,0,0) and (0.5, 0.5, 0.5) in lattice coords.
    We generate a lattice that covers the cube, then select 100 points.
    """
    # Target density: 100 points in volume 1.
    # BCC density: 2 atoms per cell of side 'a'.
    # 2/a^3 = 100 => a = (2/100)^(1/3) = 0.2714
    a = (2.0 / N) ** (1/3.0)
    
    # Number of cells to cover [-0.5, 0.5]
    n_cells = int(np.ceil(1.0 / a)) + 2
    
    points = []
    for i in range(-1, n_cells):
        for j in range(-1, n_cells):
            for k in range(-1, n_cells):
                # Corner point
                points.append([i*a, j*a, k*a])
                # Body center point
                points.append([i*a + a/2, j*a + a/2, k*a + a/2])
    
    points = np.array(points)
    
    # Filter points inside the cube (with some margin to allow optimization)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    # If we have more than N, select N closest to center? Or random?
    # Selecting closest to center creates a spherical cluster, bad for cube.
    # We should select points that cover the volume.
    # Let's just take the first N? No, that's biased to origin.
    # Let's use a random selection but seeded for reproducibility?
    # Or better: use a k-center style selection?
    
    # Simplest robust: If > N, take random subset.
    if len(points) > N:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # Fallback to grid if BCC doesn't have enough
        # This is unlikely with n_cells margin
        return _build_grid_init(5, 5, 4, 0.01)
        
    return points

def _build_halton_init(seed=42):
    """
    Build a Halton sequence initialization.
    Halton sequences are low-discrepancy sequences that provide good uniform coverage.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    rng = np.random.default_rng(seed)
    # Start indices to avoid 0,0,0
    start = rng.integers(1, 1000, size=3)
    
    centers = np.zeros((N, 3))
    for i in range(N):
        centers[i, 0] = halton(start[0] + i, 2) - 0.5
        centers[i, 1] = halton(start[1] + i, 3) - 0.5
        centers[i, 2] = halton(start[2] + i, 5) - 0.5
    
    # Clip to be safely inside
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _optimize_from_initial(centers, deadline, iu):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Beta schedule
    beta_start = 50.0
    beta_end = 2000.0
    n_beta_steps = 8
    
    # If not enough time, do fewer steps
    if time.time() > deadline - 5.0:
        return best_centers, best_radius

    for step in range(n_beta_steps):
        if time.time() > deadline - 1.0:
            break
            
        # Geometric progression
        if n_beta_steps == 1:
            beta = beta_start
        else:
            beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective_and_grad,
                x0,
                args=(beta, iu),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 10}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
            else:
                # If no improvement, maybe we are stuck. Add a tiny noise?
                # For now, just continue with the next beta (steeper)
                pass
                
        except Exception:
            pass
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    # Deadline 140s to be safe against 150s kill
    deadline = start_time + 140.0
    
    # Pre-compute unique indices for pairwise distances
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    # 1. Grid Initializations
    # 4x5x5, 5x4x5, 5x5x4 are the main candidates for 100.
    grid_configs = [
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 4),
        (10, 10, 1), # Flat sheet, might be suboptimal but checks boundary
        (10, 5, 2),
    ]
    
    noise_scales = [0.0, 0.01, 0.02]
    
    for nx, ny, nz in grid_configs:
        if time.time() > deadline - 10.0:
            break
        
        for noise in noise_scales:
            if time.time() > deadline - 5.0:
                break
            
            centers = _build_grid_init(nx, ny, nz, noise)
            if centers is None:
                continue
                
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            # Optimize
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
    
    # 2. BCC Initialization
    if time.time() < deadline - 15.0:
        centers = _build_bcc_init()
        if centers is not None and len(centers) == N:
            r_init = _compute_radius(centers)
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # 3. Halton Initialization
    if time.time() < deadline - 15.0:
        centers = _build_halton_init()
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
        
        opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers

    # 4. Random Initializations (if time permits)
    # Use a few random starts to escape deep local minima
    rng = np.random.default_rng(123)
    n_random_starts = 3
    
    if time.time() < deadline - 10.0:
        for i in range(n_random_starts):
            if time.time() > deadline - 5.0:
                break
            
            centers = rng.uniform(-0.45, 0.45, (N, 3))
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            # Shorter optimization for random starts to save time
            # We can just call the same function, it will check deadline
            opt_centers, opt_radius = _optimize_from_initial(centers, deadline, iu)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # Fallback
    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
    
    return best_centers
