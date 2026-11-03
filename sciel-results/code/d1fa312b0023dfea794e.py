"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
CUBE_HALF = 0.5

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import logsumexp

def _radius(centers):
    """Calculate the maximum possible radius for the given centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    dist_to_walls = np.min(np.stack([
        CUBE_HALF - np.abs(centers[:, 0]),
        CUBE_HALF - np.abs(centers[:, 1]),
        CUBE_HALF - np.abs(centers[:, 2])
    ], axis=1), axis=1)
    min_wall_dist = np.min(dist_to_walls)
    return min(min_pair_dist / 2.0, min_wall_dist)

def _local_search(centers, max_iters=50000, time_limit=None):
    """
    Vectorized gradient local search to maximize the minimum radius.
    Uses a soft-min surrogate objective with annealing.
    """
    centers = centers.copy()
    current_r = _radius(centers)
    best_centers = centers.copy()
    best_r = current_r

    if time_limit is None:
        time_limit = time.time() + 5

    # Precompute triangle indices
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def soft_min_value_and_grad(c, beta):
        """
        Vectorized soft-min surrogate for the packing objective.
        Returns (value, gradient) for the soft-min of all constraints.
        """
        # Pairwise distances
        diff_full = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_full = np.sqrt(np.sum(diff_full ** 2, axis=2))
        d_pairs = d_full[tri_i, tri_j] / 2.0

        # Wall distances
        wall_dists = np.concatenate([
            CUBE_HALF - c[:, 0],
            c[:, 0] + CUBE_HALF,
            CUBE_HALF - c[:, 1],
            c[:, 1] + CUBE_HALF,
            CUBE_HALF - c[:, 2],
            c[:, 2] + CUBE_HALF
        ])
        g_all = np.concatenate([d_pairs, wall_dists])

        # Soft min value
        neg_beta_g = -beta * g_all
        log_sum_exp = logsumexp(neg_beta_g)
        val = -log_sum_exp / beta

        # Weights
        weights = np.exp(neg_beta_g - log_sum_exp)

        # Gradient: vectorized pairwise contribution
        grad_c = np.zeros_like(c)

        # Pairwise gradient
        diff_ij = c[tri_i] - c[tri_j]
        d_ij = d_full[tri_i, tri_j]
        safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
        unit_vecs = diff_ij / safe_d[:, np.newaxis]
        w_pair = weights[:n_pairs] * 0.5 / safe_d
        contrib = w_pair[:, np.newaxis] * unit_vecs

        np.add.at(grad_c, tri_i, contrib)
        np.add.at(grad_c, tri_j, -contrib)

        # Wall gradients
        ws = n_pairs
        grad_c[:, 0] -= weights[ws:ws+N]
        grad_c[:, 0] += weights[ws+N:ws+2*N]
        grad_c[:, 1] -= weights[ws+2*N:ws+3*N]
        grad_c[:, 1] += weights[ws+3*N:ws+4*N]
        grad_c[:, 2] -= weights[ws+4*N:ws+5*N]
        grad_c[:, 2] += weights[ws+5*N:ws+6*N]

        return val, grad_c

    def exact_grad(c):
        """
        Exact subgradient for the nonsmooth min objective.
        """
        grad = np.zeros_like(c)
        
        # Pairwise
        diff_full = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_full = np.sqrt(np.sum(diff_full ** 2, axis=2))
        np.fill_diagonal(d_full, np.inf)
        min_pair = np.min(d_full)
        eps = 1e-8
        active_pairs = d_full < (min_pair + eps)
        rows, cols = np.where(active_pairs)
        
        for r_idx, c_idx in zip(rows, cols):
            if r_idx == c_idx: continue
            vec = c[r_idx] - c[c_idx]
            dist = np.linalg.norm(vec)
            if dist > 1e-12:
                grad[r_idx] += vec / dist
                grad[c_idx] -= vec / dist
        
        # Wall
        dist_to_walls = np.min(np.stack([
            CUBE_HALF - np.abs(c[:, 0]),
            CUBE_HALF - np.abs(c[:, 1]),
            CUBE_HALF - np.abs(c[:, 2])
        ], axis=1), axis=1)
        min_wall = np.min(dist_to_walls)
        current_limit = min(min_pair / 2.0, min_wall)
        
        for i in range(N):
            if dist_to_walls[i] < current_limit + eps:
                x, y, z = c[i]
                dx = CUBE_HALF - abs(x)
                dy = CUBE_HALF - abs(y)
                dz = CUBE_HALF - abs(z)
                
                if dx <= dy and dx <= dz:
                    grad[i, 0] += 1.0 if x >= 0 else -1.0
                elif dy <= dx and dy <= dz:
                    grad[i, 1] += 1.0 if y >= 0 else -1.0
                else:
                    grad[i, 2] += 1.0 if z >= 0 else -1.0
        
        return grad

    # Annealing schedule
    betas = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 100000, 200000]
    step_size = 0.005
    min_step_size = 1e-6
    stagnation = 0

    for beta in betas:
        if time.time() > time_limit:
            break
        
        for i in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Use soft-min gradient for smoothness
            val, grad = soft_min_value_and_grad(centers, beta)
            
            # Normalize gradient
            norm = np.linalg.norm(grad)
            if norm > 1e-12:
                grad /= norm
            
            # Propose new centers
            new_centers = centers + step_size * grad
            new_centers = np.clip(new_centers, -CUBE_HALF, CUBE_HALF)
            
            # Calculate new radius
            new_r = _radius(new_centers)
            
            if new_r > current_r + 1e-12:
                centers = new_centers
                current_r = new_r
                stagnation = 0
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            else:
                stagnation += 1
                if stagnation > 100:
                    # Random perturbation
                    centers = centers + np.random.normal(0, step_size, size=centers.shape)
                    centers = np.clip(centers, -CUBE_HALF, CUBE_HALF)
                    current_r = _radius(centers)
                    if current_r < best_r - 1e-3:
                        centers = best_centers.copy()
                        current_r = best_r
                    stagnation = 0
                    step_size = max(min_step_size, step_size * 0.8)

    return best_centers, best_r

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s limit

    best_centers = None
    best_r = 0.0

    # Strategy: Try a few different initializations
    # 1. Jittered Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]

    np.random.seed(42)
    init_configs = []
    # Jittered grid
    init_configs.append(grid + np.random.normal(0, 0.02, size=grid.shape))

    # Random configs
    for _ in range(9):
        rand_c = np.random.uniform(-CUBE_HALF * 0.8, CUBE_HALF * 0.8, size=(N, 3))
        init_configs.append(rand_c)

    # Run local search on each
    for idx, config in enumerate(init_configs):
        if time.time() > time_limit:
            break

        remaining = time_limit - time.time()
        allocated = remaining / (len(init_configs) - idx)

        c, r = _local_search(config, max_iters=50000, time_limit=time.time() + allocated)

        if r > best_r:
            best_r = r
            best_centers = c

    if best_centers is None:
        return grid

    return best_centers
# EVOLVE-BLOCK-END
