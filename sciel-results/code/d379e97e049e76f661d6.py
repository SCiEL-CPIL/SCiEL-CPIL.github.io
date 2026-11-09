import numpy as np
import time

N = 200

def _calc_radius(pos):
    """
    Calculate the maximum radius for the given centers.
    pos: (N, 2) array of centers.
    """
    # Pairwise distances
    # To save memory/time, we can compute upper triangle only, but full matrix is fine for N=200
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    # Distance to boundary for each circle
    # x-boundary: min(0.5 - x, 0.5 + x)
    # y-boundary: min(0.5 - y, 0.5 + y)
    margin_x = np.min(np.minimum(0.5 - pos[:, 0], 0.5 + pos[:, 0]))
    margin_y = np.min(np.minimum(0.5 - pos[:, 1], 0.5 + pos[:, 1]))
    min_bound_dist = np.min([margin_x, margin_y])
    
    return min(min_pair_dist / 2, min_bound_dist)

def _init_hex_grid(n):
    """
    Initialize centers in a hexagonal pattern.
    """
    # Estimate number of rows
    # Hexagonal packing density is higher.
    # Let's just use a square grid for simplicity of init, then optimize.
    # Or a slightly distorted grid.
    
    # Square grid init is robust.
    m = int(np.ceil(np.sqrt(n)))
    # Center the grid
    x = np.linspace(-0.5 + 0.5/m, 0.5 - 0.5/m, m)
    y = np.linspace(-0.5 + 0.5/m, 0.5 - 0.5/m, m)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    return grid[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Initial guess
    pos = _init_hex_grid(N)
    current_r = _calc_radius(pos)
    best_pos = pos.copy()
    best_r = current_r
    
    # Simulated Annealing parameters
    T = 0.05  # Initial temperature
    T_min = 1e-7
    cooling = 0.995
    
    # We will run iterations until time runs out
    while time.time() < deadline:
        # Generate a new candidate
        # Move a random subset of points or a single point?
        # Moving a single point is slow to escape deep local minima.
        # Let's move a random number of points (1 to 5)
        n_move = np.random.randint(1, 6)
        indices = np.random.choice(N, n_move, replace=False)
        
        new_pos = pos.copy()
        # Random perturbation
        step_size = T
        new_pos[indices] += np.random.normal(0, step_size, (n_move, 2))
        
        # Clip to square
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        new_r = _calc_radius(new_pos)
        
        # Acceptance criterion
        delta = new_r - current_r
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            pos = new_pos
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_pos = pos.copy()
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Occasionally reheat to escape local minima
        if np.random.rand() < 0.005:
            T = 0.05
            # Also maybe restart from a slightly perturbed best solution?
            # No, just reheat.
            
    return best_pos
