"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE_SECONDS = 140.0  # safety margin before the 150s kill

def _exact_radius(x):
    """
    x: (N, 3) array of centres.
    Returns the minimum of:
      - half the minimum pairwise distance
      - the minimum distance from any centre to the cube boundary
    """
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    # Distance to walls
    wall_dists = np.minimum(0.5 - x, 0.5 + x)
    min_wall = np.min(wall_dists)

    return min(min_pair, min_wall)

def _soft_min_radius_and_grad(x, lam):
    """
    Surrogate objective: -soft_min_radius.
    soft_min uses log-sum-exp for the minimum of pairwise distances and wall distances.
    We maximize the radius, so we minimize the negative.
    
    x: (N, 3)
    lam: sharpness parameter (higher = closer to true min)
    """
    x = x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = np.minimum(0.5 - x, 0.5 + x)
    flat_wall = wall_dists.flatten()
    
    # Combine all distances into one array
    iu = np.triu_indices(N, k=1)
    pair_flat = dists[iu]
    
    all_dists = np.concatenate([pair_flat, flat_wall])
    
    # soft min = -1/lam * log(sum(exp(-lam * d)))
    min_d = np.min(all_dists)
    scaled = -lam * (all_dists - min_d)
    scaled = np.clip(scaled, -700, 700)
    logsumexp = np.log(np.sum(np.exp(scaled)))
    soft_min = min_d - logsumexp / lam
    
    # Gradient w.r.t. x
    weights = np.exp(scaled)
    weights /= np.sum(weights)
    
    grad = np.zeros((N, 3))
    
    # Pairwise contributions
    W = np.zeros((N, N))
    W[iu] = weights[:len(iu[0])]
    W = W + W.T
    np.fill_diagonal(W, 0.0)
    
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    factor = W / safe_dists
    sum_factor = np.sum(factor, axis=1)
    weighted_x = factor @ x
    pair_grad = x * sum_factor[:, np.newaxis] - weighted_x
    
    # Wall contributions
    wall_weights = weights[len(iu[0]):].reshape(N, 3)
    sign_x = np.sign(x)
    wall_grad = -wall_weights * sign_x
    
    grad = pair_grad + wall_grad
    
    return -soft_min, -grad

def _bcc_lattice_init():
    """Body Centered Cubic lattice initialization."""
    # BCC: points at integer lattice + body centers
    # For 50 points, we can use a 3x3x3 grid (27) + 27 body centers = 54, take 50
    # Or scale appropriately
    m = 3
    # Grid points
    grid_pts = []
    for k in range(m + 1):
        for j in range(m + 1):
            for i in range(m + 1):
                grid_pts.append([i / m * 1.0 - 0.5, j / m * 1.0 - 0.5, k / m * 1.0 - 0.5])
    
    # Body centers
    body_pts = []
    for k in range(m):
        for j in range(m):
            for i in range(m):
                body_pts.append([(i + 0.5) / m * 1.0 - 0.5, 
                                 (j + 0.5) / m * 1.0 - 0.5, 
                                 (k + 0.5) / m * 1.0 - 0.5])
    
    all_pts = np.array(grid_pts + body_pts)
    # Shuffle and take N
    np.random.seed(42)
    idx = np.random.choice(len(all_pts), N, replace=False)
    x = all_pts[idx]
    # Jitter
    x += np.random.randn(N, 3) * 0.02
    x = np.clip(x, -0.49, 0.49)
    return x

def _fcc_lattice_init():
    """Face Centered Cubic lattice initialization."""
    # FCC: points at corners + face centers
    m = 3
    pts = []
    for k in range(m + 1):
        for j in range(m + 1):
            for i in range(m + 1):
                # Corner
                pts.append([i / m * 1.0 - 0.5, j / m * 1.0 - 0.5, k / m * 1.0 - 0.5])
                # Face centers (only for interior faces to avoid duplicates)
                if i < m:
                    pts.append([(i + 0.5) / m * 1.0 - 0.5, j / m * 1.0 - 0.5, k / m * 1.0 - 0.5])
                if j < m:
                    pts.append([i / m * 1.0 - 0.5, (j + 0.5) / m * 1.0 - 0.5, k / m * 1.0 - 0.5])
                if k < m:
                    pts.append([i / m * 1.0 - 0.5, j / m * 1.0 - 0.5, (k + 0.5) / m * 1.0 - 0.5])
    
    all_pts = np.array(pts)
    np.random.seed(43)
    idx = np.random.choice(len(all_pts), N, replace=False)
    x = all_pts[idx]
    x += np.random.randn(N, 3) * 0.02
    x = np.clip(x, -0.49, 0.49)
    return x

def _cubic_grid_init():
    """Simple cubic grid initialization."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.seed(44)
    idx = np.random.choice(len(grid), N, replace=False)
    x = grid[idx]
    x += np.random.randn(N, 3) * 0.01
    x = np.clip(x, -0.49, 0.49)
    return x

def _random_init():
    """Random initialization."""
    np.random.seed(123)
    x = np.random.uniform(-0.49, 0.49, (N, 3))
    return x

def _jittered_grid_init():
    """Jittered grid with more structure."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.seed(45)
    idx = np.random.choice(len(grid), N, replace=False)
    x = grid[idx]
    # Larger jitter
    x += np.random.randn(N, 3) * 0.05
    x = np.clip(x, -0.49, 0.49)
    return x

def _optimize_from_init(x0, start_time):
    """Optimize from a given initial position."""
    x_curr = x0.copy()
    
    # Annealing-like schedule for lambda
    lambdas = [5, 20, 100, 500, 1000]
    
    for lam in lambdas:
        if time.time() - start_time > DEADLINE_SECONDS:
            break
            
        try:
            res = minimize(
                _soft_min_radius_and_grad,
                x_curr,
                method='L-BFGS-B',
                args=(lam,),
                jac=True,
                bounds=[(-0.499, 0.499)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 30}
            )
            x_curr = res.x
        except Exception:
            pass
        
        x_curr = np.clip(x_curr, -0.499, 0.499)
    
    return x_curr

def construct():
    start_time = time.time()
    best_x = None
    best_r = -1.0
    
    initializers = [
        _bcc_lattice_init,
        _fcc_lattice_init,
        _cubic_grid_init,
        _jittered_grid_init,
        _random_init
    ]
    
    for init_fn in initializers:
        if time.time() - start_time > DEADLINE_SECONDS:
            break
            
        x0 = init_fn()
        x_opt = _optimize_from_init(x0, start_time)
        
        r = _exact_radius(x_opt)
        
        if r > best_r:
            best_r = r
            best_x = x_opt.copy()
    
    if best_x is None:
        best_x = _random_init()
    
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x
# EVOLVE-BLOCK-END
