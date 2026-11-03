"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # 15s safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are inside the square [-0.5, 0.5]^2
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Distance to walls
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances using pdist for exact min
    from scipy.spatial.distance import pdist
    min_pair = np.min(pdist(centers))
    
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    # For hexagonal packing, the density is pi/(2*sqrt(3)) ≈ 0.9069
    # We need to fit N circles. Let's try different orientations.
    
    # Hexagonal lattice: spacing s between centers
    # For a hex grid, we can use rows with alternating offsets
    # Let's parameterize by the number of rows
    
    best_centers = None
    best_r = 0.0
    
    # Try different row counts
    for n_rows in range(10, 25):
        n_cols = int(np.ceil(N / n_rows))
        
        # Hexagonal spacing: horizontal = s, vertical = s * sqrt(3)/2
        # With alternating offset of s/2
        # We need to fit in [-0.5, 0.5]^2
        
        # Let's compute the required spacing
        # Width: (n_cols - 1) * s + 2*r <= 1  =>  s + 2*r <= 1/(n_cols-1) if n_cols > 1
        # Actually, for hex: the width is (n_cols - 1) * s + 2*r
        # The height is (n_rows - 1) * s * sqrt(3)/2 + 2*r
        
        # We want to maximize r subject to:
        # (n_cols - 1) * s + 2*r <= 1
        # (n_rows - 1) * s * sqrt(3)/2 + 2*r <= 1
        # s >= 2*r (non-overlapping in hex: nearest neighbor distance = s)
        
        # From hex geometry, nearest neighbor distance = s
        # So we need s >= 2*r
        
        # Let's solve: maximize r
        # s = 2*r (tight packing)
        # (n_cols - 1) * 2*r + 2*r <= 1  =>  2*r * n_cols <= 1  =>  r <= 1/(2*n_cols)
        # (n_rows - 1) * 2*r * sqrt(3)/2 + 2*r <= 1  =>  r * ((n_rows-1)*sqrt(3) + 2) <= 1  =>  r <= 1/((n_rows-1)*sqrt(3) + 2)
        
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        
        # Build the grid
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = -0.5 + r + j * s + (i % 2) * s / 2
                y = -0.5 + r + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        centers = np.array(centers[:N])
        
        # Check if all are in bounds
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            # Adjust: try to center the grid
            # Recompute with centering
            xs = centers[:, 0]
            ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            
            # Now check if it fits
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                continue
        
        r_actual = _compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

def _soft_min(x, beta=50.0):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    x_min = np.min(x)
    shifted = -(x - x_min) * beta
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return x_min - log_sum / beta

def _smooth_objective(centers, t, beta=50.0):
    """
    Smooth surrogate: maximize t subject to all constraints.
    Returns a scalar penalty that is 0 when all constraints are satisfied with margin t.
    Constraints:
      - Pairwise: |c_i - c_j| >= 2*t for all i < j
      - Wall: min(0.5 - |c_ik|) >= t for all i,k
    """
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(centers)
    # Soft min of (d - 2*t)
    pair_violation = _soft_min(d - 2.0 * t, beta)
    
    # Wall distances
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    wall_violation = _soft_min(wall_dists - t, beta)
    
    # Combined penalty: want both >= 0, so penalty is max(-pair_violation, -wall_violation)
    # Use soft max for smoothness
    v1 = -pair_violation
    v2 = -wall_violation
    v_max = np.max([v1, v2])
    if v_max > 0:
        # Soft max of the two violations
        beta_max = 20.0
        shifted = np.array([v1, v2]) - v_max
        return v_max + np.log(np.sum(np.exp(shifted * beta_max))) / beta_max
    else:
        return 0.0

def _gradient_refine(centers, max_iter=2000, lr=0.001):
    """
    Gradient-based refinement using smooth surrogate with epigraph formulation.
    We optimize over centers and t jointly to maximize t.
    """
    centers = centers.copy()
    t = _compute_radius(centers)
    best_t = t
    best_centers = centers.copy()
    
    no_improve_count = 0
    current_lr = lr
    beta = 50.0
    
    # Use more neighbors for better gradient estimation
    k_neighbors = 8
    
    for it in range(max_iter):
        if _time_left() < 10.0:
            break
        
        # Compute current constraint values
        from scipy.spatial.distance import pdist
        d = pdist(centers)
        min_pair_dist = np.min(d)
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        min_wall = np.min(wall_dists)
        
        current_r = min(min_pair_dist / 2.0, min_wall)
        
        # If we improved t, update
        if current_r > best_t + 1e-12:
            best_t = current_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Build KDTree for neighbor queries
        tree = cKDTree(centers)
        dists, idx = tree.query(centers, k=k_neighbors)
        
        # Gradient: for each point, sum of repulsion from nearby neighbors + wall repulsion
        grad = np.zeros_like(centers)
        
        # Determine the bottleneck radius to weight gradients appropriately
        r_target = current_r
        
        for i in range(N):
            # Repulsion from nearby neighbors
            for k in range(1, k_neighbors):
                j = idx[i, k]
                diff = centers[i] - centers[j]
                dist = dists[i, k]
                if dist > 1e-10:
                    # Weight by inverse distance, stronger for closer pairs
                    # Use a sharper falloff to focus on the most constrained pairs
                    margin = dist - 2.0 * r_target
                    if margin < 0.05:  # Only consider pairs that are close to violating
                        weight = 1.0 / (dist * dist + 1e-8)
                        grad[i] += weight * diff
            
            # Wall repulsion
            for k in range(2):
                if centers[i, k] < 0:
                    wall_dist = 0.5 + centers[i, k]
                    if wall_dist < r_target + 0.05:
                        grad[i, k] += 1.0 / (wall_dist + 1e-6)
                else:
                    wall_dist = 0.5 - centers[i, k]
                    if wall_dist < r_target + 0.05:
                        grad[i, k] -= 1.0 / (wall_dist + 1e-6)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update
        centers += current_lr * grad
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Adaptive learning rate with slower decay
        if no_improve_count > 50:
            current_lr *= 0.95
            no_improve_count = 0
            if current_lr < 1e-9:
                break
    
    return best_centers, best_t

def _local_search(centers, max_iter=5000, step_scale=0.01):
    """Local search using gradient-free perturbation with gradient refinement."""
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    tree = cKDTree(centers)
    
    # Phase 1: Random perturbation search
    for it in range(max_iter):
        if _time_left() < 30.0:
            break
        
        # Find the most constrained pairs
        dists, idx = tree.query(centers, k=3)
        nn_dists = dists[:, 1]
        nn_idxs = idx[:, 1]
        
        # Also check wall constraints
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        
        # Perturb the most constrained points
        n_perturb = 2 + (it % 2)
        
        candidates = np.concatenate([
            np.argsort(nn_dists)[:n_perturb],
            np.argsort(wall_dists)[:n_perturb]
        ])
        candidates = np.unique(candidates[:n_perturb])
        
        # Generate random perturbations
        improvements = 0
        for _ in range(15):  # More trial perturbations
            trial = centers.copy()
            for idx in candidates:
                angle = np.random.uniform(0, 2 * np.pi)
                step = step_scale * (0.5 + np.random.rand())
                trial[idx, 0] += step * np.cos(angle)
                trial[idx, 1] += step * np.sin(angle)
            
            trial[:, 0] = np.clip(trial[:, 0], -0.5, 0.5)
            trial[:, 1] = np.clip(trial[:, 1], -0.5, 0.5)
            
            r_trial = _compute_radius(trial)
            if r_trial > best_r + 1e-12:
                best_r = r_trial
                best_centers = trial.copy()
                centers = trial.copy()
                tree = cKDTree(centers)
                step_scale *= 0.97  # Slower decay
                improvements += 1
        
        if improvements == 0:
            step_scale *= 0.995  # Slower decay
            if step_scale < 1e-7:
                break
    
    # Phase 2: Gradient refinement
    if _time_left() > 20.0:
        centers, r = _gradient_refine(centers, max_iter=4000, lr=0.0015)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers, best_r

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Start with hexagonal grid
    centers = _generate_hex_grid()
    r = _compute_radius(centers)
    
    # Local search with gradient refinement
    if _time_left() > 10.0:
        centers, r = _local_search(centers, max_iter=10000)
    
    # Final high-precision polish pass with multiple configurations
    if _time_left() > 15.0:
        polish_configs = [
            (2000, 0.0002),
            (1500, 0.0001),
            (1000, 0.00005),
        ]
        for max_iter, lr in polish_configs:
            if _time_left() < 10.0:
                break
            centers, r_new = _gradient_refine(centers, max_iter=max_iter, lr=lr)
            if r_new > r:
                r = r_new
    
    # Final verification and clipping
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
