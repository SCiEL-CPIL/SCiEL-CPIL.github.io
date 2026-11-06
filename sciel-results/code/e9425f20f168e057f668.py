"""
Physics-based packing of 200 circles in a unit square.
Strategy: Simulated Annealing with repulsive forces.
- Initialization: Hexagonal grid.
- Dynamics: Particles repel each other (inverse square) and are attracted to a "virtual" target spacing.
- Cooling: Temperature decreases over time, allowing large moves initially and fine-tuning later.
- Constraint Handling: Penalty for exceeding bounds and overlapping.
"""
import numpy as np
import time

N = 200

def _compute_score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) == 0 or centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    # Centers must be in [-0.5, 0.5]
    # Distance from center to wall is 0.5 - |coord|
    wall_dist = np.min(0.5 - np.abs(centers))
    
    if wall_dist < 0:
        return 0.0 # Invalid, outside square

    # Distance to neighbors
    # Vectorized pairwise distance
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(wall_dist, min_dist / 2.0)
    return r

def _generate_hex_grid(N):
    """Generate a hexagonal grid of points."""
    # Estimate d (spacing)
    # Area per circle ~ N * d^2 * (sqrt(3)/2) approx?
    # Hex packing density is sqrt(3)/2.
    # N * pi * r^2 = Area * density
    # d = 2r. N * pi * (d/2)^2 = 1 * sqrt(3)/2
    # N * pi * d^2 / 4 = 0.866
    # d^2 = 4 * 0.866 / (N * pi)
    d = np.sqrt(4.0 * (np.sqrt(3.0)/2.0) / (N * np.pi))
    
    points = []
    row_spacing = d * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_spacing)) + 2
    
    for row in range(num_rows):
        y = -0.5 + row * row_spacing
        if y > 0.5:
            break
        offset = d / 2.0 if row % 2 else 0.0
        num_cols = int(np.ceil(1.0 / d)) + 2
        for col in range(num_cols):
            x = -0.5 + col * d + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        # Fallback: If hex grid doesn't have enough, just pad with random or use square grid
        # For now, return what we have and let the caller handle? 
        # Better: Generate a square grid if hex fails to provide N points.
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    return points[:N]

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    # Initial position
    centers = _generate_hex_grid(N).astype(np.float64)
    
    # Physics parameters
    # We want to maximize min_dist. 
    # Force model: F_ij = k * (1/d_ij^2) * unit_vec_ij for d_ij < d_target
    # But we don't know d_target. 
    # Alternative: Minimize potential energy U = sum_{i<j} (1/d_ij^2) + wall_penalty
    # This is what the L-BFGS approach tried to do. 
    # Simulated Annealing approach:
    # 1. Propose random displacement.
    # 2. Calculate change in "Energy" (inverse of score proxy).
    # 3. Accept if better, or worse with probability exp(-dE/T).
    
    # To make this efficient, we need a fast score evaluation.
    # Full N^2 is 40000 ops. 150s / 40000 = 3750 evals/s. 
    # If we do 1000 iterations, we have ~150ms per iteration? No, 150s total.
    # 150s / 1000 iters = 0.15s per iter.
    # 40000 ops in numpy is ~0.1ms. So we can do ~1000 evals in 0.1s.
    # So we can do ~150,000 iterations in 150s if we are efficient?
    # Let's target 50,000 iterations to be safe.
    
    max_iterations = 50000
    current_centers = centers.copy()
    best_centers = current_centers.copy()
    best_score = _compute_score(current_centers)
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.0001
    cooling_rate = (T_end / T_start) ** (1.0 / max_iterations)
    T = T_start
    
    # Precompute indices for pairs? No, vectorized full matrix is faster in numpy for N=200.
    
    for i in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Propose move
        # Random displacement for a random subset or all?
        # Moving all points with small noise is good for annealing.
        noise_scale = T * 0.1
        proposed_centers = current_centers + np.random.normal(0, noise_scale, (N, 2))
        
        # Clip to bounds? Or allow and penalize?
        # Clipping prevents "outside" states which score 0.
        # Better to clip to valid range [-0.5, 0.5] to keep score defined, 
        # but this creates "sticking" to walls. 
        # Let's clip to [-0.499, 0.499] to be strictly inside, 
        # but the score function handles walls via min_dist.
        # Actually, if a point goes outside, wall_dist becomes negative, score 0.
        # It's better to clip to [-0.5, 0.5] and let the score function handle it,
        # but if score is 0, we always reject. 
        # So we should clip to slightly inside to avoid 0 score unless overlapping.
        proposed_centers = np.clip(proposed_centers, -0.5, 0.5)
        
        # Calculate scores
        # Optimization: We only need to know if it's better or worse.
        # But we need the exact score for the acceptance criterion if we want to be precise.
        # Actually, for SA, we need Delta E.
        # Let's define Energy E = -log(r) or 1/r? 
        # r is the score. We want to maximize r.
        # Let's use E = -r. Minimize E.
        
        current_score = _compute_score(current_centers)
        proposed_score = _compute_score(proposed_centers)
        
        if proposed_score > current_score:
            current_centers = proposed_centers
            if proposed_score > best_score:
                best_score = proposed_score
                best_centers = proposed_centers.copy()
        else:
            # Accept with probability exp(-dE/T) where dE = E_new - E_old = (-r_new) - (-r_old) = r_old - r_new
            # Since r_new <= r_old, dE >= 0.
            dE = current_score - proposed_score
            if dE == 0:
                current_centers = proposed_centers
            else:
                # Avoid division by zero or overflow
                if T > 1e-10:
                    prob = np.exp(-dE / T)
                    if np.random.random() < prob:
                        current_centers = proposed_centers
                # else reject
                
        # Cool down
        T *= cooling_rate
        
    return best_centers

if __name__ == "__main__":
    # Quick test
    c = construct()
    print(_compute_score(c))
