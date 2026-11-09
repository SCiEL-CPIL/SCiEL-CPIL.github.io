"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

DEADLINE = 140.0  # Safety margin before 150s limit
_start_time = time.time()


def _time_left():
    return DEADLINE - (time.time() - _start_time)


def _compute_radius(centers):
    """Compute the score: largest common radius for given centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are within the square
    if np.any(np.abs(centers) > 0.5 + 1e-10):
        return 0.0
    # Min distance between pairs / 2
    dists = np.zeros((N, N))
    for i in range(N):
        diff = centers[i+1:] - centers[i]
        d = np.sqrt(np.sum(diff**2, axis=1))
        dists[i, i+1:] = d
        dists[i+1:, i] = d
    # Only consider i < j
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    if len(pair_dists) == 0:
        return 0.0
    min_pair = np.min(pair_dists)
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _neg_radius(centers):
    """Negative radius for minimization."""
    return -_compute_radius(centers)


def _smooth_neg_radius(centers, beta=50.0):
    """
    Smooth surrogate for negative radius using log-sum-exp.
    r = min(min_pair/2, min_wall)
    We want to maximize r, so minimize -r.
    
    Use soft-min: -softmin(a, b) ≈ -log(exp(-a/beta) + exp(-b/beta))*beta
    where softmin approximates min.
    """
    # Pair distances
    dists = np.zeros((N, N))
    for i in range(N):
        diff = centers[i+1:] - centers[i]
        d = np.sqrt(np.sum(diff**2, axis=1))
        dists[i, i+1:] = d
        dists[i+1:, i] = d
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Wall distances (distance from center to nearest wall)
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    # r = min(min_pair/2, min_wall)
    # -r = max(-min_pair/2, -min_wall)
    # Use soft-max of the two terms: softmax(a,b) ≈ log(exp(a/beta)+exp(b/beta))*beta
    
    a = -np.min(pair_dists) / 2.0 if len(pair_dists) > 0 else 0.0
    b = -np.min(wall_dists)
    
    # Soft max to make it smooth
    max_val = np.maximum(a, b)
    log_sum = max_val + np.log(np.exp((a - max_val) / beta) + np.exp((b - max_val) / beta)) * beta
    return log_sum


def _gradient_neg_radius(centers, beta=50.0):
    """
    Numerical gradient of the smooth negative radius.
    """
    eps = 1e-5
    grad = np.zeros_like(centers)
    f0 = _smooth_neg_radius(centers, beta)
    
    for i in range(N):
        for j in range(2):
            orig = centers[i, j]
            centers[i, j] = orig + eps
            f_plus = _smooth_neg_radius(centers, beta)
            centers[i, j] = orig - eps
            f_minus = _smooth_neg_radius(centers, beta)
            centers[i, j] = orig
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    
    return grad


def _local_optimize(centers, max_iter=500, tol=1e-6):
    """
    Local optimization to improve the packing.
    Uses gradient-based method with line search.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    
    # Try multiple beta values
    for beta in [100.0, 50.0, 20.0, 10.0]:
        if _time_left() < 5.0:
            break
            
        # Compute gradient
        try:
            grad = _gradient_neg_radius(centers, beta)
        except:
            continue
        
        # Line search
        lr = 0.01
        for it in range(max_iter):
            if _time_left() < 5.0:
                break
            
            new_centers = centers - lr * grad
            
            # Check boundaries
            if np.any(np.abs(new_centers) > 0.5):
                new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_r = _compute_radius(new_centers)
            
            if new_r > current_r + 1e-8:
                centers = new_centers
                current_r = new_r
                # Recompute gradient
                try:
                    grad = _gradient_neg_radius(centers, beta)
                except:
                    break
                if np.max(np.abs(grad)) < tol:
                    break
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break
    
    return centers, current_r


def _hex_initial():
    """Hexagonal packing initialization."""
    centers = np.zeros((N, 2))
    # Hexagonal lattice
    # Spacing s, rows offset
    # For a unit square, try to fit as many as possible
    # Approximate: sqrt(N) rows
    rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    spacing = 1.0 / (rows * 0.866)
    
    idx = 0
    for row in range(rows):
        if idx >= N:
            break
        y = -0.5 + spacing * (row + 0.5) * 0.866
        if abs(y) > 0.5:
            continue
        offset = (row % 2) * spacing * 0.5
        max_cols = int((1.0 - 2 * spacing * 0.25) / spacing) + 1
        for col in range(max_cols):
            if idx >= N:
                break
            x = -0.5 + spacing * (col + 0.5) + offset
            if abs(x) <= 0.5:
                centers[idx] = [x, y]
                idx += 1
        if idx >= N:
            break
    
    # If we didn't fill, add more
    if idx < N:
        # Fill remaining with random points
        for i in range(idx, N):
            centers[i] = [np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)]
    
    return centers[:N]


def _grid_initial(m):
    """Grid initialization with m columns."""
    centers = []
    n_rows = int(np.ceil(N / m))
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(n_rows) + 0.5) / n_rows - 0.5
    for y in ys:
        for x in xs:
            centers.append([x, y])
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    # Pad if needed
    while len(centers) < N:
        centers.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
    return np.array(centers[:N])


def _random_initial():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 2))


def construct():
    """
    Construct the centers of 200 equal circles inside [-1/2, 1/2]^2
    to maximize the common radius.
    """
    global _start_time
    _start_time = time.time()
    
    best_centers = None
    best_r = -1.0
    
    # Try multiple initializations
    initializations = []
    
    # Grid with various configurations
    for m in [10, 11, 12, 13, 14]:
        initializations.append(_grid_initial(m))
    
    # Hexagonal
    initializations.append(_hex_initial())
    
    # Random
    for _ in range(5):
        initializations.append(_random_initial())
    
    # Process each initialization
    for init_idx, init_centers in enumerate(initializations):
        if _time_left() < 10.0:
            break
        
        # Evaluate initial
        r = _compute_radius(init_centers)
        if r > best_r:
            best_r = r
            best_centers = init_centers.copy()
        
        # Local optimization
        try:
            opt_centers, opt_r = _local_optimize(init_centers)
            if opt_r > best_r:
                best_r = opt_r
                best_centers = opt_centers.copy()
        except Exception:
            pass
    
    # Final polish: more aggressive local optimization on best
    if best_centers is not None and _time_left() > 15.0:
        try:
            final_centers, final_r = _local_optimize(best_centers, max_iter=1000, tol=1e-8)
            if final_r > best_r:
                best_r = final_r
                best_centers = final_centers.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    # Final safety check
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
