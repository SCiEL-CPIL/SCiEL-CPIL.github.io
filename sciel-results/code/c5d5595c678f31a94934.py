"""Improved packing of 50 equal circles in a unit square via simulated annealing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
import time

def _radius_score(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^2."""
    # Distance to boundaries
    dist_bound = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    
    # Pairwise distances using KDTree for efficiency
    if len(centres) < 2:
        min_pair = np.inf
    else:
        tree = cKDTree(centres)
        # Query nearest neighbor for each point
        dists, _ = tree.query(centres, k=2)
        min_pair = dists[:, 1].min()
    
    r = min(min_pair / 2.0, dist_bound.min())
    return r

def _construct_hexagonal():
    """Create a hexagonal lattice-based initial configuration for 50 circles."""
    # For 50 circles, try different row configurations
    # Hexagonal packing: rows alternate offset
    # Try 8 rows: some with 6, some with 7
    # 8*6 = 48, need 2 more -> make 2 rows have 7: 6*6 + 2*7 = 36+14=50
    
    # Better: try configurations that fit well in unit square
    # Row counts that sum to 50
    configs = [
        [6, 7, 6, 7, 6, 7, 6, 6],  # 50
        [7, 6, 7, 6, 7, 6, 7, 4],  # 50
        [5, 6, 7, 6, 7, 6, 7, 6],  # 50
        [6, 6, 7, 6, 7, 6, 7, 5],  # 50
    ]
    
    best_centres = None
    best_r = 0
    
    for config in configs:
        n_rows = len(config)
        max_cols = max(config)
        
        # Compute spacing
        # Horizontal spacing for a row with c circles in width 1
        # Vertical spacing between hex rows
        h_spacing = 1.0 / max_cols
        v_spacing = h_spacing * np.sqrt(3) / 2
        
        # Center the lattice
        total_height = (n_rows - 1) * v_spacing
        y_start = -0.5 + 0.5 - total_height / 2  # center vertically
        
        centres = []
        for row_idx, n_cols in enumerate(config):
            y = y_start + row_idx * v_spacing
            x_spacing = 1.0 / n_cols
            x_start = -0.5 + 0.5 - (n_cols - 1) * x_spacing / 2
            # Offset odd rows
            offset = x_spacing / 2 if row_idx % 2 == 1 else 0
            for col_idx in range(n_cols):
                x = x_start + col_idx * x_spacing + offset
                centres.append([x, y])
        
        centres = np.array(centres)
        
        # Clamp to square
        centres = np.clip(centres, -0.5, 0.5)
        
        r = _radius_score(centres)
        if r > best_r:
            best_r = r
            best_centres = centres
    
    return best_centres

def _local_search(centres, max_iter=5000, seed=42, time_limit=50):
    """Aggressive local search with adaptive step size."""
    np.random.seed(seed)
    random.seed(seed)
    
    best_centres = centres.copy()
    best_score = _radius_score(centres)
    current_centres = centres.copy()
    current_score = best_score
    
    start_time = time.time()
    
    step_size = 0.01
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Adaptive step size
        if no_improve > 500:
            step_size *= 0.95
            no_improve = 0
        elif no_improve > 1000:
            step_size *= 0.9
            no_improve = 0
        
        # Generate perturbation - move one or two points
        n_move = 1 if random.random() < 0.7 else 2
        indices = np.random.choice(N, size=n_move, replace=False)
        
        new_centres = current_centres.copy()
        for idx in indices:
            delta = np.random.normal(0, step_size, size=2)
            new_centres[idx] += delta
        
        # Clamp to square
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_score = _radius_score(new_centres)
        
        if new_score > current_score:
            current_centres = new_centres
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_centres = new_centres.copy()
        else:
            # Metropolis acceptance
            if step_size > 1e-6:
                prob = np.exp((new_score - current_score) / (step_size * 0.1))
                if random.random() < prob:
                    current_centres = new_centres
                    current_score = new_score
            no_improve += 1
        
        # Occasionally do a larger move
        if it % 500 == 0 and it > 0:
            idx = np.random.randint(N)
            delta = np.random.normal(0, step_size * 3, size=2)
            new_centres = current_centres.copy()
            new_centres[idx] += delta
            new_centres = np.clip(new_centres, -0.5, 0.5)
            new_score = _radius_score(new_centres)
            if new_score > current_score:
                current_centres = new_centres
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_centres = new_centres.copy()
    
    return best_centres, best_score

def construct():
    """Construct 50 circle centres in [-0.5, 0.5]^2 maximizing the common radius."""
    # Start with hexagonal configuration
    centres = _construct_hexagonal()
    
    best_centres = centres.copy()
    best_score = _radius_score(centres)
    
    # Multiple restarts with different perturbations
    for seed in range(15):
        np.random.seed(seed + 500)
        
        # Different perturbation strategies
        if seed < 5:
            # Small random perturbation
            perturbed = centres + np.random.normal(0, 0.01, size=(N, 2))
        elif seed < 10:
            # Medium perturbation
            perturbed = centres + np.random.normal(0, 0.03, size=(N, 2))
        else:
            # Random restart from hexagonal with jitter
            perturbed = _construct_hexagonal()
            perturbed = perturbed + np.random.normal(0, 0.02, size=(N, 2))
        
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        opt_centres, opt_score = _local_search(perturbed, max_iter=3000, seed=seed + 1000, time_limit=6)
        
        if opt_score > best_score:
            best_score = opt_score
            best_centres = opt_centres
    
    # Final polish on the best solution
    final_centres, final_score = _local_search(best_centres, max_iter=10000, seed=9999, time_limit=55)
    if final_score > best_score:
        best_centres = final_centres
        best_score = final_score
    
    return best_centres
# EVOLVE-BLOCK-END
