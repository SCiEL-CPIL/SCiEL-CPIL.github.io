import numpy as np
import time
from scipy.optimize import minimize

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_positions = None
    best_value = -np.inf
    
    # Soft-min parameter
    beta = 20.0

    def calc_radii_smooth(positions, beta):
        # Ensure positions is 2D (N, 2)
        if positions.ndim == 1:
            positions = positions.reshape(N, 2)
            
        x = positions[:, 0]
        y = positions[:, 1]
        
        # Boundary distances
        bdist = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        
        # Neighbor distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # Set diagonal to infinity
        np.fill_diagonal(dist, np.inf)
        
        # Neighbor distances are 0.5 * dist
        ndist = 0.5 * dist
        
        # Softmin over neighbors for each i
        m = np.min(ndist, axis=1)
        shifted = ndist - m[:, np.newaxis]
        exp_terms = np.exp(-beta * shifted)
        log_sums = np.log(np.sum(exp_terms, axis=1))
        neighbor_softmin = m - log_sums / beta
        
        # Combine boundary and neighbor softmin
        vals = np.stack([bdist, neighbor_softmin], axis=1)
        
        m2 = np.min(vals, axis=1)
        shifted2 = vals - m2[:, np.newaxis]
        exp_terms2 = np.exp(-beta * shifted2)
        log_sums2 = np.log(np.sum(exp_terms2, axis=1))
        radii = m2 - log_sums2 / beta
        
        return radii

    def objective(positions):
        # positions is 1D from optimizer
        radii = calc_radii_smooth(positions, beta)
        return -np.sum(radii)

    def run_optimization(initial_pos):
        x0 = initial_pos.flatten()
        bounds = [(0.0, 1.0) for _ in range(2 * N)]
        
        # Reduced maxiter to speed up and fit in time
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
        
        # Always evaluate the result from minimize, even if 'success' is False,
        # as it often finds a good local optimum.
        pos = res.x.reshape(N, 2)
        
        # Calculate true radii for score
        x = pos[:, 0]
        y = pos[:, 1]
        bdist = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        ndist = 0.5 * dist
        neighbor_min = np.min(ndist, axis=1)
        true_radii = np.minimum(bdist, neighbor_min)
        score = np.sum(true_radii)
        
        return pos, score

    # Generate initial configurations
    side = int(np.ceil(np.sqrt(N)))
    grid_x, grid_y = np.meshgrid(np.linspace(0.5/side, 1-0.5/side, side), np.linspace(0.5/side, 1-0.5/side, side))
    grid_pos = np.column_stack((grid_x.flatten(), grid_y.flatten()))[:N]
    
    np.random.seed(42)
    jitter = np.random.uniform(-0.1, 0.1, size=grid_pos.shape)
    init_grid = grid_pos + jitter
    init_grid = np.clip(init_grid, 0.01, 0.99)
    
    init_random = np.random.uniform(0.05, 0.95, size=(N, 2))
    init_random2 = np.random.uniform(0.05, 0.95, size=(N, 2))
    
    initializations = [init_grid, init_random, init_random2]
    
    for init in initializations:
        if time.time() > deadline:
            break
            
        pos, score = run_optimization(init)
        if pos is not None and score > best_value:
            best_value = score
            best_positions = pos
        
        # Additional random restarts if time permits
        if time.time() < deadline - 15:
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                init_rand = np.random.uniform(0.05, 0.95, size=(N, 2))
                pos, score = run_optimization(init_rand)
                if pos is not None and score > best_value:
                    best_value = score
                    best_positions = pos

    if best_positions is None:
        # Fallback to grid
        best_positions = init_grid
        
    # Final polish: Calculate radii based on true min distances
    x = best_positions[:, 0]
    y = best_positions[:, 1]
    bdist = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    ndist = 0.5 * dist
    neighbor_min = np.min(ndist, axis=1)
    radii = np.minimum(bdist, neighbor_min)
    
    result = np.column_stack((x, y, radii))
    
    return result
