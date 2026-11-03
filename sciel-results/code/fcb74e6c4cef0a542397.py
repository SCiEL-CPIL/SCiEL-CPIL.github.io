"""Pack 200 equal circles in a unit square using smooth soft-min objective + L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Distance to walls
    x = centers[:, 0]
    y = centers[:, 1]
    min_wall = np.min(np.concatenate([
        0.5 - np.abs(x),
        0.5 - np.abs(y)
    ]))
    
    return min(min_pair, min_wall)


def _soft_min_objective(centers, beta):
    """
    Objective to minimize: -log(sum(exp(-beta * d_ij))) for pairs,
    plus wall constraints.
    
    We want to maximize the minimum distance, so we minimize a soft approximation
    of -min_distance.
    
    Actually, let's use: minimize sum of (1/d_ij) for pairs + wall penalties,
    or better: use the log-sum-exp trick on distances.
    
    Let's use: f = -logsumexp(-beta * d) which approximates -min(d) for large beta.
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # Soft min of distances: -logsumexp(-beta * dist)
    # For large beta, this approximates -min(dist)
    neg_beta_dist = -beta * dist
    # Numerically stable logsumexp
    max_val = np.max(neg_beta_dist)
    logsumexp = max_val + np.log(np.sum(np.exp(neg_beta_dist - max_val)))
    pair_term = -logsumexp
    
    # Wall distances
    x = centers[:, 0]
    y = centers[:, 1]
    wall_dists = np.concatenate([
        0.5 - x,  # distance to right wall
        0.5 + x,  # distance to left wall
        0.5 - y,  # distance to top wall
        0.5 + y   # distance to bottom wall
    ])
    
    # Soft min of wall distances
    neg_beta_wall = -beta * wall_dists
    max_wall = np.max(neg_beta_wall)
    logsumexp_wall = max_wall + np.log(np.sum(np.exp(neg_beta_wall - max_wall)))
    wall_term = -logsumexp_wall
    
    return pair_term + wall_term


def _soft_min_gradient(centers, beta):
    """Gradient of the soft-min objective."""
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise part
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    neg_beta_dist = -beta * dist
    max_val = np.max(neg_beta_dist)
    exp_vals = np.exp(neg_beta_dist - max_val)
    sum_exp = np.sum(exp_vals)
    
    # d(-logsumexp)/d(centers_i) = -sum_j [exp(-beta*d_ij)/sum_exp] * (-beta * d(dist)/d(centers_i))
    # = sum_j [beta * exp(-beta*d_ij)/sum_exp] * d(dist)/d(centers_i)
    # d(dist)/d(centers_i) = (centers_i - centers_j) / dist_ij
    
    weights = beta * exp_vals / sum_exp
    weights[np.diag_indices(n)] = 0
    
    for i in range(n):
        # grad_i = sum_j weights[i,j] * (centers_i - centers_j) / dist[i,j]
        for j in range(n):
            if i != j:
                d = dist[i, j]
                if d > 1e-10:
                    grad[i] += weights[i, j] * (centers[i] - centers[j]) / d
    
    # Wall part
    x = centers[:, 0]
    y = centers[:, 1]
    wall_dists = np.concatenate([
        0.5 - x,
        0.5 + x,
        0.5 - y,
        0.5 + y
    ])
    
    neg_beta_wall = -beta * wall_dists
    max_wall = np.max(neg_beta_wall)
    exp_wall = np.exp(neg_beta_wall - max_wall)
    sum_exp_wall = np.sum(exp_wall)
    
    # d(-logsumexp_wall)/d(centers)
    # Wall distances: 0.5-x, 0.5+x, 0.5-y, 0.5+y
    # d(0.5-x)/dx = -1, d(0.5+x)/dx = 1, d(0.5-y)/dy = -1, d(0.5+y)/dy = 1
    
    w0 = exp_wall[:n] / sum_exp_wall      # for 0.5-x
    w1 = exp_wall[n:2*n] / sum_exp_wall   # for 0.5+x
    w2 = exp_wall[2*n:3*n] / sum_exp_wall # for 0.5-y
    w3 = exp_wall[3*n:] / sum_exp_wall    # for 0.5+y
    
    for i in range(n):
        # d/dx: -beta*w0[i]*(-1) + beta*w1[i]*(1) = beta*(w0[i] + w1[i])... wait
        # Let me recompute. f = -logsumexp(-beta * w)
        # df/dw_k = -(-beta * exp(-beta*w_k)/sum_exp) = beta * exp(-beta*w_k)/sum_exp
        # dw_0k/dx = -1, dw_1k/dx = 1, dw_2k/dy = -1, dw_3k/dy = 1
        # df/dx = df/dw_0 * dw_0/dx + df/dw_1 * dw_1/dx
        #       = beta*w0[i]*(-1) + beta*w1[i]*(1)
        #       = beta*(w1[i] - w0[i])
        # df/dy = beta*(w3[i] - w2[i])
        grad[i, 0] += beta * (w1[i] - w0[i])
        grad[i, 1] += beta * (w3[i] - w2[i])
    
    return grad


def _random_init(seed=42):
    """Random initial positions inside the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _grid_init():
    """Grid initial positions."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _hex_init():
    """Hexagonal lattice initial positions."""
    # Try to fit a hexagonal pattern
    points = []
    # Hexagonal spacing
    # For N points, try various spacings
    best_spacing = 0.04
    for spacing in np.linspace(0.03, 0.05, 20):
        pts = []
        row = 0
        while len(pts) < N:
            offset = (row % 2) * spacing * 0.5
            col = 0
            while len(pts) < N:
                x = -0.5 + spacing/2 + col * spacing + offset
                y = -0.5 + spacing * np.sqrt(3)/2 * row
                if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                    pts.append([x, y])
                col += 1
            row += 1
        if len(pts) >= N:
            pts = pts[:N]
            # Check if this is better
            arr = np.array(pts)
            # Quick check
            break
    return np.array(pts)


def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    best_centers = _grid_init()
    best_radius = _compute_radius(best_centers)
    
    # Try multiple initializations and beta values
    inits = [
        _grid_init(),
        _random_init(42),
        _random_init(123),
        _random_init(999),
    ]
    
    betas = [50, 100, 200, 500]
    
    for init in inits:
        if time.time() - start_time > deadline:
            break
        
        for beta in betas:
            if time.time() - start_time > deadline:
                break
            
            try:
                # Optimize with L-BFGS-B
                result = minimize(
                    _soft_min_objective,
                    init.copy(),
                    jac=_soft_min_gradient,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 500,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20
                    }
                )
                
                if result.success or result.fun < 0:
                    centers = result.x
                    # Clip to valid range
                    centers[:, 0] = np.clip(centers[:, 0], -0.499, 0.499)
                    centers[:, 1] = np.clip(centers[:, 1], -0.499, 0.499)
                    
                    radius = _compute_radius(centers)
                    if radius > best_radius:
                        best_radius = radius
                        best_centers = centers.copy()
                        
            except Exception:
                pass
    
    # Final local refinement: maximize minimum distance using a simpler approach
    # Use projected gradient on the actual min-distance objective
    if time.time() - start_time < deadline - 5:
        centers = best_centers.copy()
        lr = 0.001
        for it in range(2000):
            if time.time() - start_time > deadline - 2:
                break
            
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
            dist_sq = np.sum(diff * diff, axis=2)
            dist = np.sqrt(dist_sq + 1e-12)
            np.fill_diagonal(dist, np.inf)
            
            # Find the closest pair
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            i, j = min_idx
            
            # Gradient: push i and j apart
            direction = centers[i] - centers[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            direction = direction / norm
            
            grad = np.zeros_like(centers)
            grad[i] = -direction
            grad[j] = direction
            
            # Also check wall constraints
            x = centers[:, 0]
            y = centers[:, 1]
            wall_dists = np.stack([
                0.5 - x,
                0.5 + x,
                0.5 - y,
                0.5 + y
            ], axis=1)
            min_wall_idx = np.argmin(wall_dists, axis=1)
            for i in range(N):
                if wall_dists[i, min_wall_idx[i]] < dist[min_idx]:
                    # Push away from wall
                    if min_wall_idx[i] == 0:  # right wall
                        grad[i, 0] -= 1.0
                    elif min_wall_idx[i] == 1:  # left wall
                        grad[i, 0] += 1.0
                    elif min_wall_idx[i] == 2:  # top wall
                        grad[i, 1] -= 1.0
                    else:  # bottom wall
                        grad[i, 1] += 1.0
            
            centers += lr * grad
            centers[:, 0] = np.clip(centers[:, 0], -0.499, 0.499)
            centers[:, 1] = np.clip(centers[:, 1], -0.499, 0.499)
            
            radius = _compute_radius(centers)
            if radius > best_radius:
                best_radius = radius
                best_centers = centers.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
