# EVOLVE-BLOCK-START
import time
import numpy as np

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N
    t0 = time.time()
    DEADLINE = 600.0
    
    def elapsed():
        return time.time() - t0

    # Pre-compute tube indices
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten().astype(np.float64)
    j_flat = j_grid.flatten().astype(np.float64)
    
    # High-resolution grid for volume estimation
    # 64 tubes * 50^3 points = 8M checks per eval. 
    # Vectorized: 64 * 125k = 8M booleans. ~8MB memory. Fast.
    n_grid = 50
    a_grid, b_grid, c_grid = np.meshgrid(
        np.linspace(0, 1, n_grid), 
        np.linspace(0, 1, n_grid), 
        np.linspace(0, 1, n_grid), 
        indexing='ij'
    )
    a_flat = a_grid.flatten()
    b_flat = b_grid.flatten()
    c_flat = c_grid.flatten()
    
    def estimate_volume(positions):
        """
        positions: (64, 2) array of base (x, y)
        Returns: estimated volume of union in [0,1]^3
        """
        x_base = positions[:, 0]
        y_base = positions[:, 1]
        
        # Tube k occupies [x_k + c*i_k/N, x_k + w + c*i_k/N] x [y_k + c*j_k/N, y_k + w + c*j_k/N]
        # Vectorized check:
        # x_shift[k, m] = x_base[k] + c_flat[m] * i_flat[k] / N
        # y_shift[k, m] = y_base[k] + c_flat[m] * j_flat[k] / N
        
        x_shift = np.outer(x_base, np.ones_like(c_flat)) + np.outer(i_flat / N, c_flat)
        y_shift = np.outer(y_base, np.ones_like(c_flat)) + np.outer(j_flat / N, c_flat)
        
        in_x = (a_flat[None, :] >= x_shift) & (a_flat[None, :] <= x_shift + w)
        in_y = (b_flat[None, :] >= y_shift) & (b_flat[None, :] <= y_shift + w)
        
        in_tube = in_x & in_y  # (64, 125000)
        
        # Union: any tube contains the point
        covered = np.any(in_tube, axis=0)
        
        # Volume in [0,1]^3
        return np.sum(covered) / len(covered)

    # Initial positions: simple grid
    x0 = np.arange(N) * w
    y0 = np.arange(N) * w
    x_init, y_init = np.meshgrid(x0, y0, indexing='ij')
    init_pos = np.stack([x_init.flatten(), y_init.flatten()], axis=1)
    
    # Population for GA
    pop_size = 20
    # Initialize population with the grid and some random perturbations
    population = [init_pos.copy() for _ in range(pop_size)]
    # Add some random individuals
    rng = np.random.default_rng(42)
    for i in range(1, pop_size):
        population[i] = init_pos + rng.normal(0, 0.1, size=(64, 2))
        population[i] = np.clip(population[i], 0.0, 1.0 - w)
        
    # Evaluate initial population
    vols = np.array([estimate_volume(p) for p in population])
    best_idx = np.argmin(vols)
    best_pos = population[best_idx].copy()
    best_vol = vols[best_idx]
    
    # Genetic Algorithm
    # We treat the 64 positions as a chromosome.
    # Crossover: uniform crossover between two parents.
    # Mutation: Gaussian noise on random elements.
    
    T = 0.05 # Initial mutation strength
    alpha = 0.99
    n_iter = 0
    
    while elapsed() < DEADLINE - 10:
        # Selection: Tournament selection
        def select_tournament():
            idxs = rng.integers(0, pop_size, size=3)
            return population[idxs[np.argmin(vols[idxs])]].copy()
        
        # Create new population
        new_population = []
        new_vols = []
        
        # Elitism: keep the best
        best_idx_cur = np.argmin(vols)
        new_population.append(population[best_idx_cur].copy())
        new_vols.append(vols[best_idx_cur])
        
        while len(new_population) < pop_size:
            if elapsed() > DEADLINE - 5:
                break
                
            p1 = select_tournament()
            p2 = select_tournament()
            
            # Crossover: Uniform
            mask = rng.random((64, 2)) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation
            if rng.random() < 0.8:
                # Mutate a few random elements
                n_mut = rng.integers(1, 4)
                idxs = rng.choice(64, size=n_mut, replace=False)
                child[idxs] += rng.normal(0, T, size=(n_mut, 2))
                child = np.clip(child, 0.0, 1.0 - w)
            
            vol = estimate_volume(child)
            new_population.append(child)
            new_vols.append(vol)
            
            n_iter += 1
            
        population = new_population
        vols = np.array(new_vols)
        
        # Update best
        idx_min = np.argmin(vols)
        if vols[idx_min] < best_vol:
            best_vol = vols[idx_min]
            best_pos = population[idx_min].copy()
            
        # Cool down mutation
        T *= alpha
        if T < 0.01:
            T = 0.01
            
        # If stuck for a while, try a larger perturbation (restart-like)
        # Check stagnation
        if n_iter % 100 == 0:
             # If best hasn't improved in last 100 iters (approx), increase T slightly?
             # For now, just keep cooling.
             pass

    return best_pos

# EVOLVE-BLOCK-END
