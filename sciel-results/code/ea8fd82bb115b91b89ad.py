import numpy as np
import time

N = 50

def _energy(centers):
    """
    Calculate the smooth repulsive energy.
    Lower energy is better.
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Repulsive potential: 1/d^4 is very steep, helps separate close pairs
    # We only care about pairs that are somewhat close, but summing all is fine for N=50
    # To avoid numerical issues with very large values if d is tiny, we clip dist_sq
    dist_sq_clipped = np.maximum(dist_sq, 1e-6)
    
    # Sum over upper triangle to avoid double counting and self interaction
    # Using a mask for i < j
    i, j = np.triu_indices(N, k=1)
    pair_dist_sq = dist_sq_clipped[i, j]
    
    # Energy is sum of 1/d^4
    repulsion = np.sum(1.0 / (pair_dist_sq**2))
    
    # Wall penalty: penalize points close to walls
    # Wall distance is 0.5 - |c_k|
    abs_centers = np.abs(centers)
    wall_dist = 0.5 - abs_centers
    # Penalize if wall_dist < 0.1
    # Use a smooth penalty: 1/wall_dist^2 if wall_dist < 0.1 else 0
    # Or simply sum of (max(0, 0.1 - wall_dist))^2
    # Let's use a strong penalty for being close to the wall
    wall_penalty = 0.0
    # If any point is outside, return huge penalty
    if np.any(wall_dist < -1e-6):
        return 1e10
        
    # Smooth wall penalty: sum of (1/wall_dist^2) for wall_dist < 0.2
    # This pushes points away from walls
    mask = wall_dist < 0.2
    if np.any(mask):
        # Avoid div by zero
        safe_wall_dist = np.maximum(wall_dist[mask], 1e-6)
        wall_penalty = np.sum(1.0 / (safe_wall_dist**2))
        
    return repulsion + 10.0 * wall_penalty

def _compute_score(centers):
    """Compute the minimum radius for the given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _simulated_annealing(centers, rng, steps=2000, t_start=0.1, t_end=0.001):
    """
    Perform Simulated Annealing.
    """
    current_centers = centers.copy()
    current_energy = _energy(current_centers)
    
    # Geometric cooling schedule
    # T_k = T_start * (T_end/T_start)^(k/steps)
    cool_factor = (t_end / t_start) ** (1.0 / steps)
    
    for step in range(steps):
        T = t_start * (cool_factor ** step)
        
        # Propose move: add Gaussian noise
        # Scale noise with temperature? Or fixed?
        # Fixed scale is often better for SA, let's use a small step
        noise_scale = 0.01
        perturbation = rng.normal(0, noise_scale, size=current_centers.shape)
        proposed_centers = current_centers + perturbation
        
        # Clip to bounds to ensure validity, though penalty handles it
        proposed_centers = np.clip(proposed_centers, -0.5, 0.5)
        
        proposed_energy = _energy(proposed_centers)
        
        delta_e = proposed_energy - current_energy
        
        if delta_e < 0:
            # Accept
            current_centers = proposed_centers
            current_energy = proposed_energy
        else:
            # Accept with probability exp(-delta_e / T)
            if T > 0:
                prob = np.exp(-delta_e / T)
                if rng.random() < prob:
                    current_centers = proposed_centers
                    current_energy = proposed_energy
                    
    return current_centers

def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin
    
    best_score = 0.0
    best_centers = None
    
    # Use a few different initialization strategies
    # 1. Jittered Grid
    # 2. Random
    
    # Strategy 1: Jittered Grid
    # Create a 4x4x4 grid (64 points), select 50, add jitter
    # This provides a good spread
    
    m = 4
    spacing = 1.0 / m
    grid_points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                grid_points.append([
                    -0.5 + (i + 0.5) * spacing,
                    -0.5 + (j + 0.5) * spacing,
                    -0.5 + (k + 0.5) * spacing
                ])
    grid_points = np.array(grid_points)
    
    # Select 50 points from the 64 grid points
    # We can just take the first 50, or shuffle and take 50
    # Taking first 50 might introduce bias. Let's shuffle.
    
    num_restarts = 20
    rng_master = np.random.default_rng(42)
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        rng = np.random.default_rng(42 + i)
        
        if i % 2 == 0:
            # Grid-based init
            # Shuffle grid points and take 50
            indices = rng.permutation(len(grid_points))
            centers = grid_points[indices[:N]].copy()
            # Add small jitter
            centers += rng.normal(0, 0.01, size=centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
        else:
            # Random init
            centers = rng.uniform(-0.45, 0.45, (N, 3))
        
        # Run SA
        # Adjust steps based on remaining time?
        # For now, fixed steps.
        relaxed_centers = _simulated_annealing(centers, rng, steps=1000, t_start=0.05, t_end=0.001)
        
        # Calculate score
        score = _compute_score(relaxed_centers)
        
        if score > best_score:
            best_score = score
            best_centers = relaxed_centers.copy()
            
    # Final Polish: Local Search on the best candidate
    if best_centers is not None and time.time() - start_time < time_limit - 5:
        # Try to improve the best candidate with a finer SA or local search
        # Use a smaller noise scale and more steps
        rng_polish = np.random.default_rng(999)
        polished = _simulated_annealing(best_centers, rng_polish, steps=2000, t_start=0.01, t_end=0.0001)
        polish_score = _compute_score(polished)
        
        if polish_score > best_score:
            best_score = polish_score
            best_centers = polished
            
    if best_centers is None:
        # Fallback
        centers = np.zeros((N, 3))
        grid = np.array(np.meshgrid(np.linspace(-0.4, 0.4, 4), 
                                    np.linspace(-0.4, 0.4, 4), 
                                    np.linspace(-0.4, 0.4, 4))).T.reshape(-1, 3)
        centers = grid[:50]
        return centers

    return best_centers
