import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_sum_radii = 0.0
    best_config = None
    
    # Try different aspect ratios for the rectangle (w + h = 2)
    # w ranges from 0.5 to 1.5
    aspect_ratios = np.linspace(0.6, 1.4, 15)
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
            
        h = 2.0 - w
        
        # Initialize circles in a grid
        # Approximate grid dimensions
        # For N=21, a 5x5 grid has 25 spots, we use 21.
        # Or 4x6 = 24.
        # Let's use a 5x5 grid and remove 4 corners? Or just pack 21 into 5x5.
        # 5 cols, 5 rows.
        cols = 5
        rows = 5
        total_cells = cols * rows
        
        xs = []
        ys = []
        for i in range(cols):
            for j in range(rows):
                if len(xs) < N:
                    # Center of cell
                    cx = (i + 0.5) * (w / cols)
                    cy = (j + 0.5) * (h / rows)
                    xs.append(cx)
                    ys.append(cy)
        
        xs = np.array(xs)
        ys = np.array(ys)
        
        # Initial radius: slightly less than min(w/cols, h/rows) / 2
        r = min(w / cols, h / rows) * 0.45
        
        # Local search parameters
        alpha = 0.05  # Step size
        k_rep = 1.0   # Repulsion strength
        k_bound = 1.0 # Boundary strength
        growth_step = 0.005
        max_iter = 2000
        
        current_sum_radii = N * r
        
        for iter_idx in range(max_iter):
            if time.time() > deadline:
                break
                
            # Calculate forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r + r  # Assuming uniform radius for simplicity in this phase
                    if dist < min_dist:
                        if dist < 1e-8:
                            dx = 1e-8
                            dy = 0.0
                            dist = 1e-8
                        overlap = min_dist - dist
                        force = k_rep * overlap
                        fx[i] += force * (dx / dist)
                        fy[i] += force * (dy / dist)
                        fx[j] -= force * (dx / dist)
                        fy[j] -= force * (dy / dist)
            
            # Boundary forces
            for i in range(N):
                # Left
                if xs[i] < r:
                    fx[i] += k_bound * (r - xs[i])
                # Right
                if xs[i] > w - r:
                    fx[i] -= k_bound * (xs[i] - (w - r))
                # Bottom
                if ys[i] < r:
                    fy[i] += k_bound * (r - ys[i])
                # Top
                if ys[i] > h - r:
                    fy[i] -= k_bound * (ys[i] - (h - r))
            
            # Update positions
            xs += alpha * fx
            ys += alpha * fy
            
            # Check violations to decide if we can grow
            min_dist_pair = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < min_dist_pair:
                        min_dist_pair = dist
            
            # Boundary violation check
            min_bound = np.inf
            for i in range(N):
                min_bound = min(min_bound, xs[i], ys[i], w - xs[i], h - ys[i])
            
            # If no overlaps and no boundary violations, try to grow
            if min_dist_pair >= 2*r - 1e-6 and min_bound >= r - 1e-6:
                r += growth_step
                current_sum_radii = N * r
            else:
                # If stuck, maybe reduce alpha or restart? 
                # For now, just continue searching with current r
                pass
            
            # Update best if better
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_config = np.column_stack((xs.copy(), ys.copy(), np.full(N, r)))
    
    # If no config was found (should not happen), return a safe default
    if best_config is None:
        # Fallback: small circles in a grid
        w = 1.0
        h = 1.0
        cols = 5
        rows = 5
        xs = []
        ys = []
        for i in range(cols):
            for j in range(rows):
                if len(xs) < N:
                    cx = (i + 0.5) * (w / cols)
                    cy = (j + 0.5) * (h / rows)
                    xs.append(cx)
                    ys.append(cy)
        r = 0.1
        best_config = np.column_stack((np.array(xs), np.array(ys), np.full(N, r)))
        
    return best_config
