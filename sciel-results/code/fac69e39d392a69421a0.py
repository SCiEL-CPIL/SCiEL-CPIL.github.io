"""
Improved packing of 200 equal circles in a unit square.
Strategy:
1. Generate diverse initial configurations (Hex, Square, Random).
2. Refine using a hybrid local search:
   a. Projected Gradient Descent (PGD) with momentum to escape local minima
      by maximizing the sum of squared pairwise distances (energy).
   b. Simulated Annealing (SA) on the exact non-smooth objective (min distance)
      for final polishing.
"""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are inside the square [-0.5, 0.5]^2
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Distance to walls
    # min(0.5 - x, 0.5 - y, 0.5 - (-x), 0.5 - (-y)) -> min(0.5 - |x|, 0.5 - |y|)
    # Actually: min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Using cKDTree for efficiency
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding self)
    # k=2 gives self and nearest neighbor
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
        
        # Hexagonal spacing: horizontal = s, vertical = s * sqrt(3)/2
        # With alternating offset of s/2
        # We need to fit in [-0.5, 0.5]^2
        
        # Let's compute the required spacing
        # Width: (n_cols - 1) * s + 2*r <= 1
        # Height: (n_rows - 1) * s * sqrt(3)/2 + 2*r <= 1
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

def _generate_random_grid():
    """Generate a random configuration with minimum distance constraint."""
    centers = np.random.uniform(-0.5, 0.5, (N, 2))
    # Sort by x to help with some structures? No, just random.
    # Maybe add a small grid structure to random to avoid complete chaos
    # Let's just return random for now, or a perturbed square grid.
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    noise = np.random.normal(0, 0.01, grid.shape)
    grid += noise
    grid = np.clip(grid, -0.5, 0.5)
    return grid

def _pgd_step(centers, step_size, momentum=0.9):
    """
    Perform a step of Projected Gradient Descent.
    Objective: Maximize sum of squared pairwise distances.
    Gradient of sum_{i<j} ||x_i - x_j||^2 w.r.t x_i is sum_{j!=i} 2(x_i - x_j).
    """
    # Compute gradient
    # For each i, grad_i = 2 * sum_{j!=i} (x_i - x_j)
    # This is equivalent to 2 * (N * x_i - sum_all x)
    # Wait, sum_{j!=i} (x_i - x_j) = sum_{j!=i} x_i - sum_{j!=i} x_j
    # = (N-1) x_i - (sum_all x - x_i)
    # = N x_i - sum_all x
    
    sum_all = np.sum(centers, axis=0)
    grad = 2.0 * (N * centers - sum_all)
    
    # Update with momentum
    # We need to store velocity. Since this is a stateless function call in this context,
    # we'll implement a simple SGD step without persistent momentum state across calls 
    # unless we wrap it. Let's do a simple step for now, or implement a loop inside.
    
    # Actually, let's do a mini-batch of steps inside the function or return new centers.
    # For simplicity in this structure, let's do a single step with a fixed step size.
    # To make it effective, we should probably do multiple steps.
    
    new_centers = centers + step_size * grad
    # Project to bounds
    new_centers = np.clip(new_centers, -0.5, 0.5)
    
    return new_centers

def _local_search_pgd(centers, max_iter=500, base_step=0.001):
    """
    Local search using Projected Gradient Descent on the energy function.
    This helps escape local minima of the min-distance function by pushing points apart uniformly.
    """
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # We'll run PGD for a while, then check radius.
    # If radius improves, keep it.
    # If not, maybe revert or keep going.
    # Since PGD maximizes sum of squares, it might increase min distance.
    
    step_size = base_step
    velocity = np.zeros_like(centers)
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
            
        # Compute gradient
        sum_all = np.sum(centers, axis=0)
        grad = 2.0 * (N * centers - sum_all)
        
        # Momentum
        velocity = momentum * velocity + grad
        
        # Update
        centers = centers + step_size * velocity
        
        # Project to bounds
        centers = np.clip(centers, -0.5, 0.5)
        
        # Check improvement
        if it % 10 == 0:
            r_current = _compute_radius(centers)
            if r_current > best_r:
                best_r = r_current
                best_centers = centers.copy()
                # Maybe reduce step size on improvement
                step_size *= 0.99
            else:
                # If no improvement for a while, maybe revert to best and try smaller step?
                # Or just continue. PGD is smooth, so it should converge.
                pass
        
        # Adaptive step size: reduce if oscillating?
        # Simple decay
        step_size *= 0.999
        
    return best_centers, best_r

def _simulated_annealing(centers, max_iter=5000, initial_temp=0.01, cooling_rate=0.999):
    """
    Simulated Annealing on the exact min-distance objective.
    """
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    current_r = best_r
    temp = initial_temp
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
            
        # Generate a neighbor
        trial = centers.copy()
        # Pick a random point to move
        idx = np.random.randint(N)
        # Random perturbation
        delta = np.random.normal(0, temp, 2)
        trial[idx] += delta
        # Project to bounds
        trial[idx] = np.clip(trial[idx], -0.5, 0.5)
        
        # Evaluate
        r_trial = _compute_radius(trial)
        
        # Acceptance criterion
        if r_trial > current_r:
            current_r = r_trial
            centers = trial
            if r_trial > best_r:
                best_r = r_trial
                best_centers = trial.copy()
        else:
            # Metropolis criterion: accept if exp((r_trial - current_r) / temp) > random
            # Since we maximize, if r_trial < current_r, delta is negative.
            # P = exp(delta / temp)
            delta = r_trial - current_r
            if np.random.rand() < np.exp(delta / temp):
                current_r = r_trial
                centers = trial
        
        # Cool down
        temp *= cooling_rate
        
    return best_centers, best_r

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    best_overall = None
    best_r_overall = 0.0
    
    # 1. Generate initial configurations
    inits = []
    
    # Hex grid
    hex_centers = _generate_hex_grid()
    inits.append(hex_centers)
    
    # Random grid (perturbed square)
    rand_centers = _generate_random_grid()
    inits.append(rand_centers)
    
    # Another hex with different orientation?
    # Or a square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 2. Refine each initial configuration
    for centers in inits:
        if _time_left() < 10.0:
            break
            
        r_init = _compute_radius(centers)
        if r_init > best_r_overall:
            best_r_overall = r_init
            best_overall = centers.copy()
            
        # Phase 1: PGD to spread out
        if _time_left() > 20.0:
            centers_pgd, r_pgd = _local_search_pgd(centers, max_iter=200, base_step=0.005)
            if r_pgd > best_r_overall:
                best_r_overall = r_pgd
                best_overall = centers_pgd.copy()
            centers = centers_pgd
            
        # Phase 2: SA to polish
        if _time_left() > 10.0:
            centers_sa, r_sa = _simulated_annealing(centers, max_iter=2000, initial_temp=0.005)
            if r_sa > best_r_overall:
                best_r_overall = r_sa
                best_overall = centers_sa.copy()
                
    # 3. Final verification and clipping
    if best_overall is None:
        best_overall = _generate_hex_grid()
        
    best_overall = np.clip(best_overall, -0.5, 0.5)
    
    return best_overall

# EVOLVE-BLOCK-END
