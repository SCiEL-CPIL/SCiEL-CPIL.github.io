"""
Voronoi-based circle packing optimizer.

Approach: Represent the packing as a set of circle centers. Use a smooth
"soft-min" objective based on the Voronoi inradius (distance to nearest
neighbor and walls) to optimize the centers. Combine multiple optimization
strategies:
1. Initial configurations from structured grids (hex, square) and random.
2. Local search (hill climbing) with adaptive step size.
3. L-BFGS-B gradient-based polish using a smooth approximation of the
   minimum distance (soft-min / log-sum-exp).
4. Simulated annealing to escape local minima.

The key insight is that the soft-min objective is smooth and differentiable,
allowing gradient-based methods to work effectively on the packing problem.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay
import time
import sys

N = 50
TIME_LIMIT_TOTAL = 110.0  # Leave margin for the 120s limit


def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    # N=50 is small enough for O(N^2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)


def _soft_min_objective(params, k=200.0):
    """
    Smooth approximation of the negative minimum radius.
    Uses log-sum-exp to create a smooth surrogate for min of all distances.
    
    The objective is -soft_min, so minimizing it maximizes the soft minimum.
    """
    centers = params.reshape(N, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Combine all terms
    all_terms = np.concatenate([pair_terms.flatten(), wall_dists.flatten()])
    all_terms = all_terms[np.isfinite(all_terms)]
    
    if len(all_terms) == 0:
        return 0.0
    
    m = np.min(all_terms)
    shifted = all_terms - m
    exp_terms = np.exp(-k * shifted)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - (1.0 / k) * log_sum
    
    return -soft_min_val


def _gradient_objective(params, k=500.0):
    """
    Alternative smooth objective: sum of exp(-k * d_i) where d_i are all
    constraint distances. Minimizing this pushes all distances up.
    """
    centers = params.reshape(N, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    wall_dists = 0.5 - np.abs(centers)
    
    all_terms = np.concatenate([pair_terms.flatten(), wall_dists.flatten()])
    all_terms = all_terms[np.isfinite(all_terms)]
    
    if len(all_terms) == 0:
        return 0.0
    
    # Sum of exponentials - this is smooth and differentiable
    # Larger k makes it more sensitive to the minimum
    return np.sum(np.exp(-k * all_terms))


def _local_search(centers, max_iter=3000, time_limit=15.0, initial_step=0.01):
    """
    Local search (hill climbing) to improve the packing.
    Tries to move each circle to increase its minimum clearance.
    """
    centers = centers.copy()
    best_r = _calculate_true_radius(centers)
    step_size = initial_step
    no_improve_count = 0
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        indices = np.random.permutation(N)
        
        for idx in indices:
            orig_center = centers[idx].copy()
            
            # Compute current local constraints for this circle
            wall_dist = 0.5 - np.abs(orig_center)
            current_min_wall = np.min(wall_dist)
            
            diffs = centers - orig_center
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists[idx] = np.inf
            current_min_pair = np.min(dists) / 2.0
            
            current_local_min = min(current_min_wall, current_min_pair)
            
            # Determine primary direction to move
            closest_idx = np.argmin(dists)
            if dists[closest_idx] / 2.0 < current_min_wall:
                # Constrained by neighbor
                direction = orig_center - centers[closest_idx]
            else:
                # Constrained by wall - move toward center
                direction = -orig_center
            
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                direction = np.random.normal(0, 1, 2)
                norm = np.linalg.norm(direction)
            direction /= norm
            
            # Try several candidate moves
            best_local_min = current_local_min
            best_new_center = orig_center
            
            n_trials = 30
            for _ in range(n_trials):
                # Mix of directed and random moves
                if np.random.random() < 0.7:
                    noise = np.random.normal(0, 1, 2) * step_size * 0.3
                    move = direction * step_size + noise
                else:
                    move = np.random.normal(0, step_size, 2)
                
                new_center = orig_center + move
                new_center = np.clip(new_center, -0.499, 0.499)
                
                # Evaluate new local min
                new_wall_dist = 0.5 - np.abs(new_center)
                new_min_wall = np.min(new_wall_dist)
                
                new_diffs = centers - new_center
                new_dists = np.sqrt(np.sum(new_diffs**2, axis=1))
                new_dists[idx] = np.inf
                new_min_pair = np.min(new_dists) / 2.0
                
                new_local_min = min(new_min_wall, new_min_pair)
                
                if new_local_min > best_local_min + 1e-13:
                    best_local_min = new_local_min
                    best_new_center = new_center
            
            if best_local_min > current_local_min + 1e-12:
                centers[idx] = best_new_center
                improved = True
        
        if not improved:
            no_improve_count += 1
            step_size *= 0.92
            if no_improve_count > 40:
                break
        else:
            no_improve_count = 0
            # Slightly increase step if improving well
            step_size = min(step_size * 1.02, 0.05)
        
        current_r = _calculate_true_radius(centers)
        if current_r > best_r:
            best_r = current_r
    
    return centers, best_r


def _simulated_annealing(centers, time_limit=15.0):
    """
    Simulated annealing to escape local minima.
    Accepts worse moves with decreasing probability.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _calculate_true_radius(centers)
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.995
    step_size = 0.02
    
    start_time = time.time()
    n_iter = 0
    
    while time.time() - start_time < time_limit and T > T_min:
        n_iter += 1
        idx = np.random.randint(N)
        orig_center = centers[idx].copy()
        
        # Random perturbation
        move = np.random.normal(0, step_size, 2)
        new_center = orig_center + move
        new_center = np.clip(new_center, -0.499, 0.499)
        
        centers[idx] = new_center
        new_r = _calculate_true_radius(centers)
        
        delta = new_r - best_r  # Using best_r as reference for acceptance
        
        # Accept if better, or with probability exp(delta/T) if worse
        if new_r > best_r or np.random.random() < np.exp(delta / max(T, 1e-10)):
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            centers[idx] = orig_center
        
        T *= cooling
        # Adapt step size
        if n_iter % 100 == 0:
            step_size *= 0.98
    
    return best_centers, best_r


def _optimize_from_start(start_centers, bounds, time_budget=25.0):
    """
    Multi-phase optimization from a starting configuration.
    Phase 1: Local search
    Phase 2: Simulated annealing  
    Phase 3: L-BFGS-B gradient polish
    Phase 4: Final local search
    """
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    elapsed = 0.0
    t0 = time.time()
    
    # Phase 1: Local search
    if time.time() - t0 < time_budget:
        ls_centers, ls_score = _local_search(
            best_centers, max_iter=2000, 
            time_limit=min(12.0, time_budget - (time.time() - t0)),
            initial_step=0.012
        )
        elapsed = time.time() - t0
        if ls_score > best_score:
            best_score = ls_score
            best_centers = ls_centers
    
    # Phase 2: Simulated annealing
    if time.time() - t0 < time_budget - 5:
        sa_centers, sa_score = _simulated_annealing(
            best_centers, 
            time_limit=min(10.0, time_budget - (time.time() - t0) - 5)
        )
        elapsed = time.time() - t0
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers
    
    # Phase 3: L-BFGS-B gradient polish with soft-min objective
    if time.time() - t0 < time_budget - 3:
        try:
            # Use soft-min objective for smoothness
            def obj(params):
                return _soft_min_objective(params, k=300.0)
            
            result = minimize(
                obj,
                best_centers.reshape(-1),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12,
                         'maxls': 50}
            )
            centers = result.x.reshape(N, 2)
            centers = np.clip(centers, -0.4999, 0.4999)
            true_r = _calculate_true_radius(centers)
            
            if true_r > best_score:
                best_score = true_r
                best_centers = centers
        except Exception:
            pass
    
    # Phase 4: Final local search with small steps
    if time.time() - t0 < time_budget:
        final_centers, final_score = _local_search(
            best_centers, max_iter=1000,
            time_limit=min(5.0, time_budget - (time.time() - t0)),
            initial_step=0.005
        )
        if final_score > best_score:
            best_score = final_score
            best_centers = final_centers
    
    return best_centers, best_score


def _generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal lattice (best known dense packing pattern)
    s = 0.138  # Spacing for hexagonal
    points = []
    n_rows = 8
    for i in range(n_rows):
        x_base = -0.45 + i * s * np.sqrt(3) / 2
        for j in range(8):
            y = -0.45 + j * s
            x = x_base + (s / 2) if i % 2 == 1 else x_base
            points.append([x, y])
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.48) & (np.abs(points[:, 1]) <= 0.48)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices].copy())
    
    # 2. Square grid 7x7 (pick 50)
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices].copy())
    
    # 3. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered hexagonal
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.015, (N, 2))
        jittered = np.clip(jittered, -0.48, 0.48)
        inits.append(jittered)
    
    # 5. Another random with different seed
    np.random.seed(1000)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Structured: 5 rows x 10 cols with slight jitter
    np.random.seed(777)
    xs = np.linspace(-0.42, 0.42, 10)
    ys = np.linspace(-0.42, 0.42, 5)
    structured = np.array([[x, y] for y in ys for x in xs])
    structured += np.random.normal(0, 0.01, structured.shape)
    structured = np.clip(structured, -0.48, 0.48)
    inits.append(structured)
    
    # 7. Concentric rings
    np.random.seed(888)
    ring_centers = []
    ring_radii = [0.0, 0.15, 0.30, 0.42]
    ring_counts = [1, 6, 12, 17]  # Total ~36, add more
    for r, count in zip(ring_radii, ring_counts):
        if r == 0:
            ring_centers.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for a in angles:
                ring_centers.append([r*np.cos(a), r*np.sin(a)])
    # Add more to reach 50
    while len(ring_centers) < N:
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.05, 0.45)
        ring_centers.append([r*np.cos(angle), r*np.sin(angle)])
    ring_centers = np.array(ring_centers[:N])
    ring_centers += np.random.normal(0, 0.005, ring_centers.shape)
    ring_centers = np.clip(ring_centers, -0.48, 0.48)
    inits.append(ring_centers)
    
    # 8. Random with clustering tendency
    np.random.seed(555)
    cluster_centers = np.random.uniform(-0.3, 0.3, (5, 2))
    clustered = []
    for i in range(N):
        c = cluster_centers[i % 5]
        pt = c + np.random.normal(0, 0.08, 2)
        pt = np.clip(pt, -0.48, 0.48)
        clustered.append(pt)
    inits.append(np.array(clustered))
    
    return inits


def construct():
    """
    Main entry point. Returns array of shape (N, 2) with circle centers
    in the square [-1/2, 1/2]^2.
    """
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = _generate_initializations()
    
    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    # Allocate time budget across initializations
    # Give more time to promising starts
    time_per_init = min(20.0, (TIME_LIMIT_TOTAL - 5.0) / len(inits))
    
    for i, init in enumerate(inits):
        remaining = TIME_LIMIT_TOTAL - 5.0 - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        # Adaptive time allocation: later inits get less time
        budget = min(time_per_init, remaining)
        
        centers, score = _optimize_from_start(init, bounds, time_budget=budget)
        
        if score > best_score:
            best_score = score
            best_centers = centers
        
        # Early termination if we're close to known best
        if best_score > 0.0713:
            # Still try remaining inits but with reduced budget
            time_per_init *= 0.7
    
    # Final safety check and clip
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(best_centers)):
        # Fallback to simple grid
        grid_x = np.linspace(-0.45, 0.45, 7)
        grid_y = np.linspace(-0.45, 0.45, 7)
        best_centers = np.array([[x, y] for x in grid_x for y in grid_y])[:N]
    
    return best_centers
