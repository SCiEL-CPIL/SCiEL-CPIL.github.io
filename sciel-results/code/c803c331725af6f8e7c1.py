"""Rewritten program using Multi-Restart Simulated Annealing for circle packing."""
import numpy as np
from scipy.spatial.distance import cdist

N = 200

# EVOLVE-BLOCK-START
def construct():
    def calc_radius(centers):
        # Min distance to walls
        wall_dists = np.min([
            np.min(centers[:, 0] + 0.5),
            np.min(0.5 - centers[:, 0]),
            np.min(centers[:, 1] + 0.5),
            np.min(0.5 - centers[:, 1])
        ])
        
        # Min distance between centers
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        return min(wall_dists, min_pair_dist) / 2.0

    def run_sa(centers_init, max_steps=20000):
        centers = centers_init.copy()
        current_r = calc_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        T = 0.02
        T_min = 1e-6
        cooling = 0.998
        step_size_base = 0.05
        
        for step in range(max_steps):
            i = np.random.randint(N)
            old_pos = centers[i].copy()
            
            # Adaptive step size: larger when T is high
            step_size = step_size_base * (T / 0.02 + 0.1)
            new_pos = old_pos + np.random.normal(0, step_size, size=2)
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            centers[i] = new_pos
            new_r = calc_radius(centers)
            
            if new_r > current_r:
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            else:
                delta = current_r - new_r
                if delta > 0:
                    p = np.exp(-delta / T)
                    if np.random.random() < p:
                        current_r = new_r
                    else:
                        centers[i] = old_pos
                # else: new_r == current_r, accept (or reject, doesn't matter much)
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        return best_centers, best_r

    # Initialization: Grid
    m = int(np.ceil(np.sqrt(N)))
    x = np.linspace(-0.5, 0.5, m)
    y = np.linspace(-0.5, 0.5, m)
    grid = np.array([(xi, yi) for yi in y for xi in x])
    centers_init = grid[:N].copy()
    centers_init += np.random.normal(0, 0.01, size=centers_init.shape)
    centers_init = np.clip(centers_init, -0.49, 0.49)
    
    # Multi-restart
    best_overall_r = 0
    best_overall_centers = centers_init
    
    num_restarts = 3
    steps_per_restart = 20000
    
    for r in range(num_restarts):
        # Perturb the best found so far to create a new start
        if r == 0:
            start_centers = centers_init
        else:
            start_centers = best_overall_centers + np.random.normal(0, 0.02, size=best_overall_centers.shape)
            start_centers = np.clip(start_centers, -0.49, 0.49)
            
        res_centers, res_r = run_sa(start_centers, steps_per_restart)
        
        if res_r > best_overall_r:
            best_overall_r = res_r
            best_overall_centers = res_centers

    return best_overall_centers
# EVOLVE-BLOCK-END
