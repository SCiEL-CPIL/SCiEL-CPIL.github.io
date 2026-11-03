"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize, NonlinearConstraint, Bounds

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # 10s safety margin before 150s kill

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
    
    # Use KDTree for faster nearest-neighbor distance
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
        
        # Add small jitter to break symmetry
        rng = np.random.default_rng(42)
        centers += rng.normal(0, 0.001, centers.shape)
        
        # Check if all are in bounds
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 - 1e-9):
            # Adjust: try to center the grid
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

def _gradient_refine(centers, max_iter=2000, lr=0.001):
    """
    Gradient-based refinement using smooth surrogate with epigraph formulation.
    """
    centers = centers.copy()
    t = _compute_radius(centers)
    best_t = t
    best_centers = centers.copy()
    
    no_improve_count = 0
    current_lr = lr
    
    k_neighbors = 10
    
    for it in range(max_iter):
        if _time_left() < 10.0:
            break
        
        # Compute current constraint values using KDTree (faster)
        tree = cKDTree(centers)
        dists, idx = tree.query(centers, k=k_neighbors)
        min_pair_dist = np.min(dists[:, 1])
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        min_wall = np.min(wall_dists)
        
        current_r = min(min_pair_dist / 2.0, min_wall)
        
        if current_r > best_t + 1e-14:
            best_t = current_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        grad = np.zeros_like(centers)
        r_target = current_r
        margin_threshold = max(0.01, 0.05 * (1.0 - it / max(1, max_iter)))
        
        for i in range(N):
            for k in range(1, k_neighbors):
                j = idx[i, k]
                diff = centers[i] - centers[j]
                dist = dists[i, k]
                if dist > 1e-10:
                    margin = dist - 2.0 * r_target
                    if margin < margin_threshold:
                        weight = 1.0 / (dist * dist + 1e-8)
                        if margin < 0:
                            weight *= 2.0
                        grad[i] += weight * diff
            
            for k in range(2):
                if centers[i, k] < 0:
                    wall_dist = 0.5 + centers[i, k]
                    if wall_dist < r_target + margin_threshold:
                        grad[i, k] += 1.0 / (wall_dist + 1e-6)
                else:
                    wall_dist = 0.5 - centers[i, k]
                    if wall_dist < r_target + margin_threshold:
                        grad[i, k] -= 1.0 / (wall_dist + 1e-6)
        
        min_margin = min(min_pair_dist - 2.0 * r_target, min_wall - r_target)
        margin_scale = 1.0
        if min_margin < 0.01:
            margin_scale = max(0.1, 1.0 + min_margin / 0.01 * 0.9)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
            effective_lr = current_lr * margin_scale
            centers += effective_lr * grad
        
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        if no_improve_count > 30:
            current_lr *= 0.97
            no_improve_count = 0
            if current_lr < 1e-10:
                break
    
    return best_centers, best_t

def _local_search(centers, max_iter=5000, step_scale=0.01):
    """Local search using gradient-free perturbation with gradient refinement."""
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    tree = cKDTree(centers)
    
    for it in range(max_iter):
        if _time_left() < 30.0:
            break
        
        dists, idx = tree.query(centers, k=3)
        nn_dists = dists[:, 1]
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        
        n_perturb = 3 + (it % 3)
        nn_candidates = np.argsort(nn_dists)[:n_perturb]
        wall_candidates = np.argsort(wall_dists)[:n_perturb]
        candidates = np.unique(np.concatenate([nn_candidates, wall_candidates]))
        
        improvements = 0
        for _ in range(20):
            trial = centers.copy()
            for idx in candidates:
                angle = np.random.uniform(0, 2 * np.pi)
                step = step_scale * (0.5 + np.random.rand())
                trial[idx, 0] += step * np.cos(angle)
                trial[idx, 1] += step * np.sin(angle)
            
            trial[:, 0] = np.clip(trial[:, 0], -0.5, 0.5)
            trial[:, 1] = np.clip(trial[:, 1], -0.5, 0.5)
            
            r_trial = _compute_radius(trial)
            if r_trial > best_r + 1e-14:
                best_r = r_trial
                best_centers = trial.copy()
                centers = trial.copy()
                tree = cKDTree(centers)
                step_scale *= 0.98
                improvements += 1
        
        if improvements == 0:
            step_scale *= 0.997
            if step_scale < 1e-8:
                break
    
    if _time_left() > 20.0:
        centers, r = _gradient_refine(centers, max_iter=3000, lr=0.0015)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers, best_r

def _nlp_refine(centers, max_iter=1000):
    """
    NLP-based refinement using scipy.optimize.minimize with SLSQP.
    Maximizes radius r subject to pairwise distance and wall constraints.
    """
    if _time_left() < 15.0:
        return centers.copy(), _compute_radius(centers)
    
    centers = centers.copy()
    x0 = centers.flatten()
    n = len(x0) // 2
    
    # Objective: minimize -r (maximize r)
    # Variables: x[0:2N] are centers, x[2N] is r
    
    def objective(x):
        r = x[2*N]
        return -r
    
    # Constraints:
    # 1. Pairwise distances: |c_i - c_j|^2 >= (2r)^2 for all i < j
    # 2. Wall constraints: 0.5 - |c_ik| >= r for all i, k
    
    # We'll use a simpler approach: fix r and check feasibility, then binary search
    # Or use SLSQP with explicit constraints
    
    # Let's try SLSQP with constraints
    # This might be slow for 200 points, so we limit the number of constraints checked
    
    # Actually, let's use a different approach: 
    # Use the current best radius as a lower bound and try to increase it
    current_r = _compute_radius(centers)
    
    # Define constraints for SLSQP
    # We'll only enforce the most critical constraints (nearest neighbors)
    
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=5)
    
    # Collect critical pairs (distance < 2.5 * current_r)
    critical_pairs = []
    for i in range(N):
        for k in range(1, 5):
            j = idx[i, k]
            if i < j and dists[i, k] < 2.5 * current_r:
                critical_pairs.append((i, j))
    
    # Also add wall constraints for points close to walls
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    critical_walls = [i for i in range(N) if wall_dists[i] < 1.5 * current_r]
    
    def constraints(x):
        c_centers = x[:2*N].reshape(N, 2)
        r = x[2*N]
        
        cons = []
        
        # Pairwise constraints
        for i, j in critical_pairs:
            diff = c_centers[i] - c_centers[j]
            dist_sq = np.dot(diff, diff)
            cons.append(dist_sq - (2*r)**2)
        
        # Wall constraints
        for i in critical_walls:
            for k in range(2):
                if c_centers[i, k] < 0:
                    cons.append(0.5 + c_centers[i, k] - r)
                else:
                    cons.append(0.5 - c_centers[i, k] - r)
        
        return cons
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2*N) + [(current_r, 0.1)]
    
    # Try SLSQP
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraints}],
            options={'maxiter': 50, 'ftol': 1e-8}
        )
        
        if result.success and result.x[2*N] > current_r:
            new_centers = result.x[:2*N].reshape(N, 2)
            new_r = result.x[2*N]
            
            # Verify
            new_r = _compute_radius(new_centers)
            if new_r > current_r:
                return new_centers, new_r
    except Exception:
        pass
    
    return centers, current_r

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Start with hexagonal grid
    centers = _generate_hex_grid()
    r = _compute_radius(centers)
    
    # Local search with gradient refinement
    if _time_left() > 10.0:
        centers, r = _local_search(centers, max_iter=8000)
    
    # NLP refinement
    if _time_left() > 20.0:
        centers_nlp, r_nlp = _nlp_refine(centers)
        if r_nlp > r:
            r = r_nlp
            centers = centers_nlp
    
    # Final high-precision polish pass with multiple configurations
    if _time_left() > 15.0:
        polish_configs = [
            (1500, 0.0002),
            (1000, 0.0001),
            (500, 0.00005),
        ]
        for max_iter, lr in polish_configs:
            if _time_left() < 10.0:
                break
            centers, r_new = _gradient_refine(centers, max_iter=max_iter, lr=lr)
            if r_new > r:
                r = r_new
    
    # One more round of local search if time permits
    if _time_left() > 20.0:
        centers, r_new = _local_search(centers, max_iter=3000, step_scale=0.001)
        if r_new > r:
            r = r_new
    
    # Final gradient polish
    if _time_left() > 12.0:
        centers, r_new = _gradient_refine(centers, max_iter=500, lr=0.00002)
        if r_new > r:
            r = r_new
    
    # Final NLP polish
    if _time_left() > 15.0:
        centers_nlp, r_nlp = _nlp_refine(centers)
        if r_nlp > r:
            r = r_nlp
            centers = centers_nlp
    
    # Final verification and clipping
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
