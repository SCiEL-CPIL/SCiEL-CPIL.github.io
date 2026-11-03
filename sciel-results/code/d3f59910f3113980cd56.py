"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s kill

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
    
    # Pairwise distances using KDTree for speed
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    best_centers = None
    best_r = 0.0
    
    # Try different row counts
    for n_rows in range(10, 25):
        n_cols = int(np.ceil(N / n_rows))
        
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        
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
        
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 - 1e-9):
            xs = centers[:, 0]
            ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                continue
        
        r_actual = _compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

def _gradient_refine(centers, max_iter=2000, lr=0.002):
    """
    Gradient-based refinement using repulsion forces.
    """
    centers = centers.copy()
    best_t = _compute_radius(centers)
    best_centers = centers.copy()
    
    no_improve_count = 0
    current_lr = lr
    
    for it in range(max_iter):
        if _time_left() < 15.0:
            break
        
        # Compute current radius
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        min_wall = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall)
        
        if current_r > best_t + 1e-12:
            best_t = current_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Build KDTree for neighbor queries
        tree = cKDTree(centers)
        k_neighbors = min(10, N)
        dists, idx = tree.query(centers, k=k_neighbors)
        
        # Gradient: for each point, sum of repulsion from nearby neighbors + wall repulsion
        grad = np.zeros_like(centers)
        
        for i in range(N):
            # Repulsion from nearby neighbors with distance-based weighting
            for k in range(1, k_neighbors):
                j = idx[i, k]
                diff = centers[i] - centers[j]
                dist = dists[i, k]
                if dist > 1e-10:
                    # Stronger repulsion for closer pairs, with smooth falloff
                    weight = 1.0 / (dist * dist + 1e-8)
                    grad[i] += weight * diff
            
            # Wall repulsion with smooth activation
            for k in range(2):
                if centers[i, k] < 0:
                    wall_dist = 0.5 + centers[i, k]
                    if wall_dist < 0.2:
                        grad[i, k] += 1.0 / (wall_dist + 1e-6)
                else:
                    wall_dist = 0.5 - centers[i, k]
                    if wall_dist < 0.2:
                        grad[i, k] -= 1.0 / (wall_dist + 1e-6)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update with adaptive learning rate
        centers += current_lr * grad
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Adaptive learning rate: reduce when no improvement
        if no_improve_count > 50:
            current_lr *= 0.85
            no_improve_count = 0
            if current_lr < 1e-9:
                break
        elif no_improve_count == 20:
            current_lr *= 0.95
    
    return best_centers, best_t

def _nlp_refine(centers, max_iter=500):
    """
    NLP-based refinement using scipy.optimize.minimize with SLSQP.
    Maximizes the minimum radius by minimizing -r subject to constraints.
    """
    if _time_left() < 30.0:
        return centers, _compute_radius(centers)
    
    # Flatten centers for optimization
    x0 = centers.flatten()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # Precompute pairs that are close to each other (within 3*r of current best)
    # to limit the number of constraints
    tree = cKDTree(centers)
    search_r = 3.0 * best_r
    # Get all pairs within search_r
    pairs = tree.query_pairs(search_r, output_type='ndarray')
    
    # Also get wall-constrained points
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    wall_constrained = np.where(wall_dists < 1.5 * best_r)[0]
    
    # Limit number of constraints for speed
    max_pairs = 500
    if len(pairs) > max_pairs:
        # Keep the closest pairs
        pair_dists = []
        for p in pairs:
            d = np.linalg.norm(centers[p[0]] - centers[p[1]])
            pair_dists.append((d, p))
        pair_dists.sort(key=lambda x: x[0])
        pairs = np.array([p for _, p in pair_dists[:max_pairs]])
    
    max_wall = 100
    if len(wall_constrained) > max_wall:
        wall_constrained = wall_constrained[:max_wall]
    
    def objective(x):
        """Minimize -r (i.e., maximize r)."""
        c = x.reshape(N, 2)
        # Check bounds
        if np.any(c < -0.5) or np.any(c > 0.5):
            return 1.0  # Large penalty
        
        # Compute radius
        wall_d = np.min(np.minimum(0.5 - c, c + 0.5), axis=1)
        min_wall = np.min(wall_d)
        
        # Pair distances for constrained pairs
        if len(pairs) > 0:
            pair_diffs = c[pairs[:, 0]] - c[pairs[:, 1]]
            pair_dists = np.sqrt(np.sum(pair_diffs ** 2, axis=1))
            min_pair = np.min(pair_dists)
        else:
            min_pair = 1.0
        
        r = min(min_wall, min_pair / 2.0)
        return -r
    
    def constraints_func(x):
        """List of constraint functions: each must be >= 0."""
        c = x.reshape(N, 2)
        cons = []
        
        # Wall constraints: distance to each wall >= r (approximated by current best_r)
        # Use best_r as the target radius for constraints
        for i in wall_constrained:
            # Distance to left wall
            cons.append(c[i, 0] + 0.5 - best_r)
            # Distance to right wall
            cons.append(0.5 - c[i, 0] - best_r)
            # Distance to bottom wall
            cons.append(c[i, 1] + 0.5 - best_r)
            # Distance to top wall
            cons.append(0.5 - c[i, 1] - best_r)
        
        # Pair constraints: distance between pairs >= 2*best_r
        for p in pairs:
            i, j = p[0], p[1]
            diff = c[i] - c[j]
            dist_sq = np.sum(diff ** 2)
            cons.append(dist_sq - (2 * best_r) ** 2)
        
        return cons
    
    # Use SLSQP with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Create constraint list
    constraint_list = []
    for i in range(len(constraints_func(np.zeros(2*N)))):
        def make_con(i):
            def con(x):
                return constraints_func(x)[i]
            return con
        constraint_list.append({'type': 'ineq', 'fun': make_con(i)})
    
    try:
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < -best_r:
            new_centers = result.x.reshape(N, 2)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            new_r = _compute_radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers
    except Exception:
        pass
    
    return best_centers, best_r

def _local_search(centers, max_iter=3000, step_scale=0.01):
    """Local search using gradient-free perturbation on constrained points."""
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    tree = cKDTree(centers)
    
    for it in range(max_iter):
        if _time_left() < 10.0:
            break
        
        # Find the most constrained pairs
        dists, idx = tree.query(centers, k=3)
        nn_dists = dists[:, 1]
        nn_idxs = idx[:, 1]
        
        # The minimum distance pair
        min_idx = np.argmin(nn_dists)
        min_dist = nn_dists[min_idx]
        neighbor = nn_idxs[min_idx]
        
        # Also check wall constraints
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_dist = wall_dists[min_wall_idx]
        
        # Select points to perturb
        n_perturb = 2 + (it % 2)
        
        candidates = np.concatenate([
            np.argsort(nn_dists)[:n_perturb],
            np.argsort(wall_dists)[:n_perturb]
        ])
        candidates = np.unique(candidates[:n_perturb])
        
        # Generate random perturbations
        improvements = 0
        rng = np.random.RandomState(42 + it)
        for _ in range(10):
            trial = centers.copy()
            for idx in candidates:
                angle = rng.uniform(0, 2 * np.pi)
                step = step_scale * (0.5 + rng.rand())
                trial[idx, 0] += step * np.cos(angle)
                trial[idx, 1] += step * np.sin(angle)
            
            # Keep in bounds
            trial[:, 0] = np.clip(trial[:, 0], -0.5, 0.5)
            trial[:, 1] = np.clip(trial[:, 1], -0.5, 0.5)
            
            r_trial = _compute_radius(trial)
            if r_trial > best_r + 1e-12:
                best_r = r_trial
                best_centers = trial.copy()
                centers = trial.copy()
                tree = cKDTree(centers)
                step_scale *= 0.95
                improvements += 1
        
        if improvements == 0:
            step_scale *= 0.99
            if step_scale < 1e-6:
                break
    
    return best_centers, best_r

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Generate multiple diverse initial configurations
    base_centers = _generate_hex_grid()
    
    # Create variants with small perturbations
    centers_list = [base_centers.copy()]
    
    # Add perturbed versions for diversity
    rng = np.random.RandomState(42)
    for _ in range(2):
        perturbed = base_centers.copy()
        noise = rng.normal(0, 0.005, size=perturbed.shape)
        perturbed += noise
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.45, 0.45)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.45, 0.45)
        centers_list.append(perturbed)
    
    # Multi-start gradient optimization
    best_centers = base_centers.copy()
    best_r = _compute_radius(base_centers)
    
    for centers in centers_list:
        if _time_left() < 30.0:
            break
        centers_opt, r = _gradient_refine(centers, max_iter=1500, lr=0.002)
        if r > best_r:
            best_r = r
            best_centers = centers_opt.copy()
    
    # NLP refinement using SLSQP - this is the structural change
    if _time_left() > 40.0:
        centers_nlp, r_nlp = _nlp_refine(best_centers, max_iter=300)
        if r_nlp > best_r:
            best_r = r_nlp
            best_centers = centers_nlp.copy()
    
    # Final local search polish
    if _time_left() > 20.0:
        centers_ls, r_ls = _local_search(best_centers, max_iter=2000, step_scale=0.005)
        if r_ls > best_r:
            best_r = r_ls
            best_centers = centers_ls.copy()
    
    # Final high-precision polish pass with decreasing learning rates
    if _time_left() > 15.0:
        for lr in [0.0005, 0.0002, 0.0001]:
            if _time_left() < 10.0:
                break
            centers_polish, r_new = _gradient_refine(best_centers, max_iter=500, lr=lr)
            if r_new > best_r:
                best_r = r_new
                best_centers = centers_polish.copy()
    
    # Final verification and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
