import numpy as np

def construct():
    N = 26
    circles = []
    
    # Initial grid
    grid_size = 200
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    base_candidates = np.vstack([XX.ravel(), YY.ravel()]).T
    
    for _ in range(N):
        # Calculate radii for base candidates
        dist_bound = np.minimum(np.minimum(XX, 1-XX), np.minimum(YY, 1-YY)).ravel()
        
        min_dist_to_circles = np.inf * np.ones(len(base_candidates))
        
        if circles:
            # Vectorize over circles for speed if possible, but loop is fine for 26
            for (cx, cy, cr) in circles:
                dists = np.sqrt((base_candidates[:, 0] - cx)**2 + (base_candidates[:, 1] - cy)**2)
                avail = dists - cr
                min_dist_to_circles = np.minimum(min_dist_to_circles, avail)
        
        radii = np.minimum(dist_bound, min_dist_to_circles)
        
        # Find best base candidate
        max_idx = np.argmax(radii)
        best_x = base_candidates[max_idx, 0]
        best_y = base_candidates[max_idx, 1]
        best_r = radii[max_idx]
        
        # Local refinement
        # Create a finer grid around best_x, best_y
        # Box size should be small enough to be local, large enough to capture the peak
        # The grid step was 1/200 = 0.005. Let's use a box of 0.05 (10 steps)
        # and refine with 50 points per side.
        
        box_size = 0.05
        refine_grid_size = 50
        rx = np.linspace(max(0, best_x - box_size), min(1, best_x + box_size), refine_grid_size)
        ry = np.linspace(max(0, best_y - box_size), min(1, best_y + box_size), refine_grid_size)
        RXX, RYY = np.meshgrid(rx, ry)
        refine_candidates = np.vstack([RXX.ravel(), RYY.ravel()]).T
        
        # Calculate radii for refine candidates
        r_dist_bound = np.minimum(np.minimum(RXX, 1-RXX), np.minimum(RYY, 1-RYY)).ravel()
        
        r_min_dist_to_circles = np.inf * np.ones(len(refine_candidates))
        
        if circles:
            for (cx, cy, cr) in circles:
                dists = np.sqrt((refine_candidates[:, 0] - cx)**2 + (refine_candidates[:, 1] - cy)**2)
                avail = dists - cr
                r_min_dist_to_circles = np.minimum(r_min_dist_to_circles, avail)
        
        r_radii = np.minimum(r_dist_bound, r_min_dist_to_circles)
        
        # Find best refine candidate
        r_max_idx = np.argmax(r_radii)
        new_best_x = refine_candidates[r_max_idx, 0]
        new_best_y = refine_candidates[r_max_idx, 1]
        new_best_r = r_radii[r_max_idx]
        
        # Use the refined values
        circles.append((new_best_x, new_best_y, new_best_r))
        
    return np.array(circles)
