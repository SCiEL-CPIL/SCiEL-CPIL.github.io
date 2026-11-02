"""Packing 50 equal circles in a unit square via smooth optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
SEED_SCORE = 0.071377103865  # approximate seed (grid) radius
DEADLINE = 115.0  # safety margin before 120s kill


def _compute_radius(centers):
    """Compute the max common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    # Distance to boundaries
    # Square is [-0.5, 0.5]^2
    x_min = 0.5 - np.min(centers[:, 0])
    x_max = 0.5 - np.max(centers[:, 0])
    y_min = 0.5 - np.min(centers[:, 1])
    y_max = 0.5 - np.max(centers[:, 1])
    min_boundary = min(x_min, x_max, y_min, y_max)
    return min(min_dist / 2.0, min_boundary)


def _smooth_objective(centers_flat, softness=50.0):
    """
    Smooth surrogate for maximizing the minimum separation.
    We minimize the soft-min of pairwise distances and boundary distances.
    """
    centers = centers_flat.reshape(-1, 2)
    n = centers.shape[0]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Soft-min of pairwise distances (excluding diagonal)
    # Use log-sum-exp trick for soft-min: -logsumexp(-dists/softness) * softness
    # But we need to exclude the diagonal (inf values)
    mask = ~np.eye(n, dtype=bool)
    dists_masked = np.where(mask, dists, np.inf)
    
    # Soft min of pairwise distances
    log_sum = np.log(np.sum(np.exp(-dists_masked / softness)))
    soft_min_pair = -softness * log_sum
    
    # Boundary distances
    x_min = 0.5 - np.min(centers[:, 0])
    x_max = 0.5 - np.max(centers[:, 0])
    y_min = 0.5 - np.min(centers[:, 1])
    y_max = 0.5 - np.max(centers[:, 1])
    boundary_dists = np.array([x_min, x_max, y_min, y_max])
    
    # Soft min of boundary distances
    log_sum_b = np.log(np.sum(np.exp(-boundary_dists / softness)))
    soft_min_boundary = -softness * log_sum_b
    
    # Overall soft min
    soft_min = min(soft_min_pair, soft_min_boundary)
    
    return -soft_min  # We minimize, so negate


def _gradient_objective(centers_flat, softness=50.0):
    """Compute gradient of the smooth objective."""
    centers = centers_flat.reshape(-1, 2)
    n = centers.shape[0]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    mask = ~np.eye(n, dtype=bool)
    dists_masked = np.where(mask, dists, np.inf)
    
    # Soft min weights
    exp_neg = np.exp(-dists_masked / softness)
    exp_neg_sum = np.sum(exp_neg)
    weights = exp_neg / exp_neg_sum  # These are the soft-min weights
    
    # Gradient of soft_min_pair w.r.t. centers
    # d(soft_min_pair)/d(centers_i) = -sum_j weights_ij * (centers_i - centers_j) / dists_ij
    grad_pair = np.zeros_like(centers)
    for i in range(n):
        for j in range(n):
            if i != j:
                if dists[i, j] > 1e-15:
                    grad_pair[i] += weights[i, j] * (centers[i] - centers[j]) / dists[i, j]
    
    # Boundary gradient
    x_min = 0.5 - np.min(centers[:, 0])
    x_max = 0.5 - np.max(centers[:, 0])
    y_min = 0.5 - np.min(centers[:, 1])
    y_max = 0.5 - np.max(centers[:, 1])
    boundary_dists = np.array([x_min, x_max, y_min, y_max])
    
    exp_neg_b = np.exp(-boundary_dists / softness)
    exp_neg_b_sum = np.sum(exp_neg_b)
    weights_b = exp_neg_b / exp_neg_b_sum
    
    grad_boundary = np.zeros_like(centers)
    # x_min = 0.5 - min(x) -> d/dx_i = -weight_b[0] if i is argmin, else 0
    # Actually for soft min, we need to be more careful
    # For x_min: contribution is -weight_b[0] for the point with min x
    # But with soft min, it's distributed
    # d(x_min)/d(centers_i[0]) = -1 if centers_i[0] is min, 0 otherwise
    # With soft min: d(soft_min_boundary)/d(centers_i[0]) = -weights_b[0] * I(i is argmin x) ... 
    # Actually this is getting complex. Let's use numerical gradient for boundaries or simplify.
    
    # For simplicity, let's just use the pairwise gradient and handle boundaries separately
    # The boundary constraint can be handled by clamping
    
    grad_flat = np.concatenate([grad_pair.ravel(), grad_boundary.ravel()])
    return -grad_flat  # Negate because we're minimizing -soft_min


def _construct_grid_init():
    """Create a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _construct_hex_init():
    """Create a hexagonal initialization."""
    centers = []
    spacing = 1.0 / 8
    row = 0
    while len(centers) < N:
        offset = spacing / 2 if row % 2 else 0
        x = -0.5 + spacing / 2 + offset
        while x <= 0.5 - spacing / 2 + 1e-9:
            y = -0.5 + spacing / 2 + row * spacing * np.sqrt(3) / 2
            if y <= 0.5 - spacing / 2 + 1e-9:
                centers.append([x, y])
            x += spacing
        row += 1
    return np.array(centers[:N])


def _construct_random_init(seed):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _optimize_from(centers, max_iter=500, softness=50.0):
    """Optimize from given centers using L-BFGS-B."""
    start = time.time()
    result = minimize(
        _smooth_objective,
        centers.ravel(),
        args=(softness,),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10},
        bounds=[(-0.5, 0.5)] * 2 * N
    )
    return result.x.reshape(-1, 2)


def _polish(centers, max_iter=2000):
    """High-precision polish using smaller softness."""
    start = time.time()
    result = minimize(
        _smooth_objective,
        centers.ravel(),
        args=(10.0,),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12},
        bounds=[(-0.5, 0.5)] * 2 * N
    )
    return result.x.reshape(-1, 2)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Try multiple initializations
    inits = [
        ("grid", _construct_grid_init()),
        ("hex", _construct_hex_init()),
    ]
    
    # Add some random inits
    for seed in range(20):
        inits.append((f"rand{seed}", _construct_random_init(seed)))
    
    best_centers = None
    best_radius = -np.inf
    
    for name, init_centers in inits:
        if time.time() > deadline - 30:
            break
        
        # Optimize with moderate softness
        opt_centers = _optimize_from(init_centers, max_iter=300, softness=50.0)
        
        # Polish with smaller softness
        if time.time() > deadline - 15:
            polished = opt_centers
        else:
            polished = _polish(opt_centers, max_iter=1000)
        
        radius = _compute_radius(polished)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = polished.copy()
    
    # Final check and return
    if best_centers is None:
        return _construct_grid_init()
    
    # Ensure centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
