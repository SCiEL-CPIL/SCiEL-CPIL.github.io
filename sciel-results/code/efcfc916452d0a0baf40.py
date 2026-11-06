"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100


# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the largest common radius for given centers in [-0.5, 0.5]^3."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distance
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = float(np.min(wall))
    if min_wall < 0.0:
        return 0.0
    # Pairwise distances
    # Use a vectorized approach but chunked to avoid memory issues for N=100 (fine)
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = float(np.min(dists)) / 2.0
    return min(min_wall, min_pair)


def _smooth_min(dists, walls, t):
    """Soft-min of pairwise distances/2 and wall distances using log-sum-exp."""
    # dists is (N, N) with inf on diagonal, walls is (N,)
    # We want min over all i<j of dists[i,j]/2 and min over i of walls[i]
    # Flatten relevant terms
    iu = np.triu_indices(N, k=1)
    pair_terms = dists[iu] / 2.0
    all_terms = np.concatenate([pair_terms, walls])
    # log-sum-exp for soft min: -log(sum(exp(-x/t)))
    # To avoid overflow, shift by min
    m = np.min(all_terms)
    if not np.isfinite(m):
        return 0.0
    shifted = -all_terms / t
    shifted -= np.max(shifted)  # for numerical stability
    s = np.sum(np.exp(shifted))
    return m + t * np.log(s)


def _gradient_smooth_min(centers, t):
    """Compute gradient of soft-min objective with respect to centers."""
    walls = 0.5 - np.max(np.abs(centers), axis=1)
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_terms = dists[iu] / 2.0
    all_terms = np.concatenate([pair_terms, walls])
    
    m = np.min(all_terms)
    if not np.isfinite(m):
        return np.zeros_like(centers)
    
    shifted = -all_terms / t
    shifted -= np.max(shifted)
    exp_terms = np.exp(shifted)
    s = np.sum(exp_terms)
    weights = exp_terms / s
    
    grad = np.zeros_like(centers)
    
    # Gradient of pair terms
    n_pairs = len(iu)
    pair_weights = weights[:n_pairs]
    # For each pair (i,j), term = dists[i,j]/2
    # d(term)/d(centers[i]) = (centers[i] - centers[j]) / dists[i,j]
    # d(term)/d(centers[j]) = (centers[j] - centers[i]) / dists[i,j]
    # Weighted sum
    for k, (i, j) in enumerate(zip(iu[0], iu[1])):
        d = dists[i, j]
        if d < 1e-12:
            continue
        w = pair_weights[k]
        diff_ij = centers[i] - centers[j]
        grad[i] += w * diff_ij / d
        grad[j] += w * (centers[j] - centers[i]) / d
    
    # Gradient of wall terms
    wall_weights = weights[n_pairs:]
    # walls[i] = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
    # d(walls[i])/d(centers[i,k]) = -sign(centers[i,k]) if |c_ik| is the max
    for i in range(N):
        abs_c = np.abs(centers[i])
        max_abs = np.max(abs_c)
        if max_abs < 1e-12:
            continue
        # Find which coordinate(s) are at the max
        mask = np.abs(abs_c - max_abs) < 1e-12
        # Distribute weight among max coordinates
        w = wall_weights[i] / np.sum(mask)
        for k in range(3):
            if mask[k]:
                grad[i, k] -= w * np.sign(centers[i, k])
    
    return grad


def _bake_centers():
    """Generate a BCC-like starting configuration."""
    # Simple cubic grid of 5x5x5 = 125, take first 100
    # But BCC is denser. Let's use a 4x4x4 grid = 64, plus offsets.
    # Actually, let's just use a good grid. 5x5x4 = 100.
    # x: 5, y: 5, z: 4
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return centers[:N]


def _fcc_centers():
    """Generate an FCC-like starting configuration."""
    # FCC lattice points in a cube.
    # Use a 4x4x4 grid of unit cells, each with 4 FCC points.
    # That's 64*4 = 256 points, too many.
    # Let's use a 3x3x3 grid of unit cells. 27*4 = 108.
    # We'll take 100.
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # FCC points in unit cell [i, i+1] x [j, j+1] x [k, k+1]
                # Relative positions: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                # But we need to center them.
                # Let's just use a simple perturbation of a grid.
                pass
    # Fallback to a dense grid
    m = 5
    ticks = np.linspace(-0.45, 0.45, m)
    centers = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return centers[:N]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    init_bake = _bake_centers()
    init_fcc = _fcc_centers()
    
    best_centers = init_bake.copy()
    best_radius = _compute_radius(best_centers)
    
    # Annealing parameters
    t_start = 0.05
    t_end = 0.001
    n_iter = 3000
    
    # Try both initializations
    for init_idx, init_centers in enumerate([init_bake, init_fcc]):
        if time.time() > deadline:
            break
        
        centers = init_centers.copy()
        current_radius = _compute_radius(centers)
        
        for it in range(n_iter):
            if time.time() > deadline:
                break
            
            # Annealing schedule
            progress = it / n_iter
            t = t_start * (t_end / t_start) ** progress
            
            # Compute gradient
            grad = _gradient_smooth_min(centers, t)
            
            # Step
            lr = 0.01 * (1.0 - progress) + 0.001
            centers = centers + lr * grad
            
            # Project back into cube
            centers = np.clip(centers, -0.49, 0.49)
            
            # Kick to escape local minima
            if it % 100 == 0:
                kick_strength = 0.05 * (1.0 - progress)
                centers += kick_strength * np.random.randn(*centers.shape)
                centers = np.clip(centers, -0.49, 0.49)
            
            # Evaluate
            current_radius = _compute_radius(centers)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        
        # Polish phase
        if time.time() > deadline:
            break
        
        # Final polish with small steps
        for polish_iter in range(500):
            if time.time() > deadline:
                break
            t = 0.001
            grad = _gradient_smooth_min(centers, t)
            lr = 0.001
            centers = centers + lr * grad
            centers = np.clip(centers, -0.49, 0.49)
            current_radius = _compute_radius(centers)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
    
    # Final projection to ensure validity
    best_centers = np.clip(best_centers, -0.49, 0.49)
    return best_centers
# EVOLVE-BLOCK-END
