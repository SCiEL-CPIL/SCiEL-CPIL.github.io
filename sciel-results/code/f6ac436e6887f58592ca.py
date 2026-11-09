import numpy as np
import time
from scipy.spatial.distance import cdist

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    def score_centres(c):
        """Compute the max common radius for given centres."""
        if len(c) <= 1:
            return 0.5
        # Pairwise distances
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        # Boundary distances
        min_boundary = np.min(0.5 - np.abs(c))
        return min(min_pair, min_boundary)
    
    # 1. Initialization: Multiple random starts from jittered grids
    np.random.seed(42)
    
    def make_initial_grid():
        # Try different grid configurations
        configs = [
            (4, 4, 4),  # 64 points
            (5, 5, 4),  # 100 points  
            (5, 5, 5),  # 125 points
            (3, 4, 5),  # 60 points
            (4, 5, 5),  # 100 points
        ]
        nx, ny, nz = configs[np.random.randint(len(configs))]
        tx = (np.arange(nx) + 0.5) / nx - 0.5
        ty = (np.arange(ny) + 0.5) / ny - 0.5
        tz = (np.arange(nz) + 0.5) / nz - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        # Shuffle and pick N
        indices = np.random.permutation(len(grid))[:N]
        centres = grid[indices].copy()
        # Add jitter
        centres += np.random.normal(0, 0.02, centres.shape)
        centres = np.clip(centres, -0.49, 0.49)
        return centres
    
    # Try a few initial configurations and keep the best
    best_centres = None
    best_radius = -np.inf
    for trial in range(5):
        c = make_initial_grid()
        r = score_centres(c)
        if r > best_radius:
            best_radius = r
            best_centres = c.copy()
    
    centres = best_centres.copy()
    current_radius = best_radius
    
    # 2. Simulated Annealing with targeted moves
    temp = 0.01
    cooling = 0.995
    min_temp = 1e-6
    
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Pick a random point to move
        i = np.random.randint(N)
        
        # Compute current distances from point i
        d_i = cdist(centres[i:i+1], centres).flatten()
        d_i[i] = np.inf
        
        # Boundary distances for point i
        b_i = 0.5 - np.abs(centres[i])
        
        # Find the binding constraint: closest pair or boundary
        min_pair_dist = np.min(d_i)
        min_boundary_dist = np.min(b_i)
        
        # Determine move direction: away from closest constraint
        direction = np.zeros(3)
        
        if min_pair_dist < min_boundary_dist:
            # Move away from closest neighbor
            j = np.argmin(d_i)
            vec = centres[i] - centres[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                direction = vec / norm
        else:
            # Move away from closest boundary face
            axis = np.argmin(b_i)
            sign = np.sign(centres[i][axis])
            if sign == 0:
                sign = 1
            direction[axis] = -sign
        
        # Add random component
        direction += np.random.normal(0, 0.5, 3)
        
        # Step size proportional to temperature
        step = temp * np.linalg.norm(direction)
        if step < 1e-8:
            step = 1e-8
        direction = direction / np.linalg.norm(direction) * step
        
        # Propose new position
        old_pos = centres[i].copy()
        new_pos = centres[i] + direction
        new_pos = np.clip(new_pos, -0.49, 0.49)
        
        # Evaluate new radius
        centres[i] = new_pos
        new_radius = score_centres(centres)
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        delta = current_radius - new_radius
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-9)):
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centres = centres.copy()
        else:
            centres[i] = old_pos
        
        # Cool down
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
        
        # Periodic restart with small perturbation if stuck
        if iteration % 500 == 0 and time.time() - start_time < time_limit - 10:
            # Add small random perturbation to escape local minima
            perturbation = np.random.normal(0, 0.01, (N, 3))
            centres = np.clip(centres + perturbation, -0.49, 0.49)
            current_radius = score_centres(centres)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centres = centres.copy()
    
    return best_centres
