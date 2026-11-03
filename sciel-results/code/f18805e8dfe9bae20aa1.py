"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for 150s limit
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Function to compute the minimum radius
    def min_radius(c):
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    current_r = min_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Phase 1: Simulated Annealing to escape local minima
    T = 0.01
    cooling_rate = 0.995
    min_T = 1e-6
    
    while time.time() - start_time < time_limit * 0.4 and T > min_T:
        # Calculate subgradient direction
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        dist_to_walls = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
        
        r = min(np.min(d) / 2.0, np.min(dist_to_walls))
        
        G = np.zeros_like(centers)
        eps = 1e-4
        
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 2 * r + eps:
                    vec = centers[i] - centers[j]
                    norm = d[i, j]
                    if norm > 1e-9:
                        G[i] += vec / norm
                        G[j] -= vec / norm
        
        for i in range(N):
            if dist_to_walls[i] < r + eps:
                for k in range(3):
                    if centers[i, k] > 0:
                        G[i, k] -= 1.0
                    else:
                        G[i, k] += 1.0
        
        norm_G = np.linalg.norm(G)
        if norm_G > 1e-9:
            G /= norm_G
        
        step_size = 0.01 * (T / 0.01) + 0.001
        noise = np.random.normal(0, 0.005, centers.shape)
        new_centers = centers + step_size * G + noise
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = min_radius(new_centers)
        
        delta = new_r - r
        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        
        T *= cooling_rate
    
    # Phase 2: Local optimization using scipy
    if time.time() - start_time < time_limit - 20:
        # Define objective function to maximize (so we minimize negative)
        def objective(x):
            c = x.reshape(N, 3)
            r = min_radius(c)
            return -r
        
        # Run local optimization from multiple starting points
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 20:
                break
            
            # Start from current best or a perturbed version
            if restart == 0:
                x0 = best_centers.flatten()
            else:
                x0 = best_centers.flatten() + np.random.normal(0, 0.01, N * 3)
                x0 = np.clip(x0, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Use L-BFGS-B for constrained optimization
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                
                if result.success or result.fun < objective(x0):
                    new_centers = result.x.reshape(N, 3)
                    new_r = min_radius(new_centers)
                    
                    if new_r > best_r:
                        best_r = new_r
                        best_centers = new_centers.copy()
                        
            except Exception:
                pass
    
    # Phase 3: Final polish with smaller steps
    if time.time() - start_time < time_limit - 5:
        polish_start = time.time()
        polish_limit = time_limit - 5.0
        step_size_polish = 0.0005
        noise_scale_polish = 0.0005
        
        centers = best_centers.copy()
        
        while time.time() - polish_start < (polish_limit - polish_start) and time.time() - start_time < time_limit:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            
            dist_to_walls = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
            r = min(np.min(d) / 2.0, np.min(dist_to_walls))
            
            G = np.zeros_like(centers)
            eps = 1e-5
            for i in range(N):
                for j in range(i + 1, N):
                    if d[i, j] < 2 * r + eps:
                        vec = centers[i] - centers[j]
                        norm = d[i, j]
                        if norm > 1e-9:
                            G[i] += vec / norm
                            G[j] -= vec / norm
            
            for i in range(N):
                if dist_to_walls[i] < r + eps:
                    for k in range(3):
                        if centers[i, k] > 0:
                            G[i, k] -= 1.0
                        else:
                            G[i, k] += 1.0
            
            norm_G = np.linalg.norm(G)
            if norm_G > 1e-9:
                G /= norm_G
            
            noise = np.random.normal(0, noise_scale_polish, centers.shape)
            new_centers = centers + step_size_polish * G + noise
            new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_r = min_radius(new_centers)
            
            if new_r > r:
                centers = new_centers
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
