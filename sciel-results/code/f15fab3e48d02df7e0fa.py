import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initialization: Concentric rings
    # 1, 6, 12, 18, 13
    points = np.zeros((N, 2))
    
    # Ring 0
    idx = 0
    points[idx] = [0, 0]
    idx += 1
    
    # Ring 1: 6 points
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    radius1 = 0.2
    points[idx:idx+6] = np.column_stack([radius1 * np.cos(angles), radius1 * np.sin(angles)])
    idx += 6
    
    # Ring 2: 12 points
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    radius2 = 0.4
    points[idx:idx+12] = np.column_stack([radius2 * np.cos(angles), radius2 * np.sin(angles)])
    idx += 12
    
    # Ring 3: 18 points
    angles = np.linspace(0, 2*np.pi, 18, endpoint=False)
    radius3 = 0.6
    points[idx:idx+18] = np.column_stack([radius3 * np.cos(angles), radius3 * np.sin(angles)])
    idx += 18
    
    # Ring 4: 13 points
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    radius4 = 0.8
    points[idx:idx+13] = np.column_stack([radius4 * np.cos(angles), radius4 * np.sin(angles)])
    idx += 13
    
    best_points = points.copy()
    best_r = -1.0
    
    # Optimization
    # We will use a simple iterative improvement with random kicks
    # Objective: maximize min( min_ij dist_ij / 2, min_i (1 - norm_i) )
    
    def get_radius(pts):
        # Pairwise distances
        # Using broadcasting
        # pts shape (N, 2)
        # diff shape (N, N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        norms = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    current_r = get_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    step_size = 0.05
    noise_scale = 0.05
    
    # Simulated Annealing style
    T = 0.1
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # Compute forces
        # For each pair i,j, if dist < 2*r_target, push apart
        # But we don't have a fixed r_target.
        # Instead, let's just do a random search with acceptance based on improvement.
        
        new_points = points + np.random.normal(0, noise_scale, points.shape)
        
        # Ensure new points are somewhat close to original to avoid huge jumps
        # Actually, let's try a gradient step
        # Gradient of min distance is complex.
        # Let's stick to stochastic local search but make it smarter.
        
        # Calculate current radius
        r_curr = get_radius(points)
        
        # Calculate new radius
        r_new = get_radius(new_points)
        
        if r_new > r_curr:
            points = new_points
            if r_new > best_r:
                best_r = r_new
                best_points = new_points.copy()
                # Reduce noise slightly on improvement?
                # noise_scale *= 0.99
        else:
            # Accept with probability exp((r_new - r_curr) / T)
            if np.random.random() < np.exp((r_new - r_curr) / T):
                points = new_points
        
        # Cool down
        T *= cooling_rate
        if T < 0.001:
            T = 0.001
            # If stuck, kick
            # We can detect stuck by checking if best_r hasn't improved in many steps
            # But simpler: just keep going, maybe increase noise if no improvement for a while?
            # For now, let's just run.
            
        # Optional: Reduce noise scale over time to polish
        noise_scale *= 0.999
        if noise_scale < 0.001:
             noise_scale = 0.001

    return best_points
