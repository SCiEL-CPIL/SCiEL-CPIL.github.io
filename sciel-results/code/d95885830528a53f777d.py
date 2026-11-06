"""Program for packing 50 equal spheres in a unit cube.
Uses BCC-inspired initialization with repulsive potential energy minimization."""
import numpy as np
from scipy.optimize import minimize

N = 50


def _repulsive_potential(flat_x, power=4):
    """Compute a repulsive potential energy that we want to MINIMIZE.
    
    We use an inverse-power repulsion between all pairs and walls.
    Minimizing this maximizes the minimum distance.
    """
    x = flat_x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, np.inf)
    
    # Wall distances: distance from each center to the nearest wall
    # Cube is [-0.5, 0.5]^3
    wall_dists = np.minimum(
        np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        ),
        np.minimum(0.5 - x[:, 2], x[:, 2] + 0.5)
    )
    
    # Clip to avoid division by zero
    dist_sq_safe = np.maximum(dist_sq, 1e-12)
    wall_dists_safe = np.maximum(wall_dists, 1e-8)
    
    # Calculate potentials
    # 1 / d^power = 1 / (d_sq^(power/2))
    pair_energy = np.sum(1.0 / (dist_sq_safe ** (power / 2.0)))
    wall_energy = np.sum(1.0 / (wall_dists_safe ** power))
    
    return pair_energy + wall_energy


def _score(x):
    """Compute the actual score: minimum radius."""
    x = x.reshape(N, 3)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        ),
        np.minimum(0.5 - x[:, 2], x[:, 2] + 0.5)
    )
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _bcc_initialization(scale=1.0, seed=42):
    """Initialize 50 centers on a distorted BCC superlattice."""
    np.random.seed(seed)
    
    n_cells = 3
    cell_size = 1.0 / n_cells
    
    centers = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                cx = -0.5 + (i + 0.5) * cell_size
                cy = -0.5 + (j + 0.5) * cell_size
                cz = -0.5 + (k + 0.5) * cell_size
                
                offset = 0.25 * cell_size
                centers.append([cx - offset, cy - offset, cz - offset])
                centers.append([cx + offset, cy + offset, cz + offset])
    
    centers = np.array(centers[:N])
    
    perturbation = 0.02 * scale
    centers += np.random.randn(N, 3) * perturbation
    
    margin = 0.48
    centers = np.clip(centers, -margin, margin)
    
    return centers

def _random_initialization(seed=42):
    """Initialize 50 centers randomly."""
    np.random.seed(seed)
    centers = np.random.uniform(-0.45, 0.45, (N, 3))
    return centers

def _optimize_from(x0, bounds, max_iter=1500):
    """Run L-BFGS-B with increasing power."""
    x_current = x0.copy()
    
    powers = [4, 8, 16]
    
    for p in powers:
        def potential_wrapper(flat_x):
            return _repulsive_potential(flat_x, power=p)
        
        result = minimize(
            potential_wrapper,
            x_current.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 100000}
        )
        
        x_current = result.x.reshape(N, 3)
        x_current = np.clip(x_current, -0.5, 0.5)
        
    return x_current


def construct():
    """Construct centers for 50 equal spheres in a unit cube."""
    best_x = None
    best_score = 0.0
    
    seeds = [42, 123, 456, 789, 1011, 2023, 333, 444]
    
    for seed in seeds:
        # BCC Init
        x0_bcc = _bcc_initialization(scale=1.0, seed=seed)
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        x_opt_bcc = _optimize_from(x0_bcc, bounds, max_iter=1500)
        
        s_bcc = _score(x_opt_bcc)
        if s_bcc > best_score:
            best_score = s_bcc
            best_x = x_opt_bcc
        
        # Random Init
        x0_rand = _random_initialization(seed=seed)
        
        x_opt_rand = _optimize_from(x0_rand, bounds, max_iter=1500)
        
        s_rand = _score(x_opt_rand)
        if s_rand > best_score:
            best_score = s_rand
            best_x = x_opt_rand
            
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x
