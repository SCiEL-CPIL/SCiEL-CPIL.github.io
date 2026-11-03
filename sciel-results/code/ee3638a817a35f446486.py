import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 110.0  # Keep a safety margin
    
    best_sum = 0.0
    best_circles = None
    
    # Helper to check validity and calculate score
    def evaluate(circles):
        # circles: (N, 2) for centers, (N,) for radii
        # Check bounds
        if np.any(circles[:, 0] < 0) or np.any(circles[:, 1] < 0):
            return -1
        if np.any(circles[:, 0] > 1) or np.any(circles[:, 1] > 1):
            return -1
        
        # Check overlaps
        # Distance matrix
        diff = circles[:, np.newaxis, :] - circles[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        min_dists = np.min(np.where(np.eye(N, dtype=bool), np.inf, dists))
        
        # For each circle, the minimum distance to any other circle must be >= 2 * radius
        # Actually, for circle i, dist(i, j) >= r_i + r_j for all j != i
        # A simpler check: for all pairs i < j, dist(i, j) >= r_i + r_j
        # Vectorized:
        radii = circles[:, 2]
        # Create distance matrix
        d = np.sqrt(np.sum((circles[:, np.newaxis, :2] - circles[np.newaxis, :, :2])**2, axis=2))
        # Create sum of radii matrix
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        # Check if any pair violates constraint
        # Exclude diagonal
        np.fill_diagonal(d, np.inf)
        np.fill_diagonal(r_sum, 0)
        
        if np.any(d < r_sum - 1e-9):
            return -1
            
        return np.sum(radii)

    def run_sa(seed):
        np.random.seed(seed)
        # Initialize
        # Start with a reasonable guess: random positions, small radii
        centers = np.random.rand(N, 2) * 0.8 + 0.1
        radii = np.random.rand(N) * 0.05 + 0.02
        circles = np.column_stack((centers, radii))
        
        current_sum = evaluate(circles)
        if current_sum < 0:
             # If initial is invalid, shrink radii until valid
             while current_sum < 0:
                 radii *= 0.9
                 circles = np.column_stack((centers, radii))
                 current_sum = evaluate(circles)
        
        best_sum_local = current_sum
        best_circles_local = circles.copy()
        
        T = 0.1
        T_min = 1e-5
        alpha = 0.999
        steps = 50000
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
                
            # Perturb
            new_circles = circles.copy()
            
            # Move a random subset of centers
            mask = np.random.rand(N) < 0.2
            step_size = 0.05 * T
            new_circles[mask, :2] += np.random.randn(np.sum(mask), 2) * step_size
            
            # Adjust a random subset of radii
            mask_r = np.random.rand(N) < 0.2
            new_circles[mask_r, 2] += np.random.randn(np.sum(mask_r)) * 0.01 * T
            
            # Clamp radii to be positive and not too large
            new_circles[:, 2] = np.clip(new_circles[:, 2], 0.001, 0.5)
            # Clamp centers to be inside square (with margin for radius)
            # This is a soft clamp, validity check will handle the rest
            new_circles[:, 0] = np.clip(new_circles[:, 0], 0, 1)
            new_circles[:, 1] = np.clip(new_circles[:, 1], 0, 1)
            
            new_sum = evaluate(new_circles)
            
            if new_sum < 0:
                # Invalid move, reject
                continue
                
            if new_sum > current_sum:
                circles = new_circles
                current_sum = new_sum
                if current_sum > best_sum_local:
                    best_sum_local = current_sum
                    best_circles_local = circles.copy()
            else:
                # Metropolis criterion
                delta = new_sum - current_sum
                if np.random.rand() < np.exp(delta / T):
                    circles = new_circles
                    current_sum = new_sum
                    
            T *= alpha
            if T < T_min:
                break
                
        return best_sum_local, best_circles_local

    # Run multiple restarts
    seed = 42
    while time.time() - start_time < time_limit:
        s, c = run_sa(seed)
        if s > best_sum:
            best_sum = s
            best_circles = c
        seed += 1
        
    # If no valid circle found, fallback to seed
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_circles
# EVOLVE-BLOCK-END
