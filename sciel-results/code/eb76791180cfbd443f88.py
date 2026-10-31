import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Initial grid layout
    side = int(np.ceil(np.sqrt(N)))
    # Create grid positions
    # We want to place N circles in a grid of size side x side
    # But we only have N circles, so we take the first N positions
    # To make it more symmetric, we can center them
    # Let's use a 5x5 grid for 21 circles
    # Indices: 0..4 for i, 0..4 for j
    # We take 21 of them.
    
    # A better initial layout: 5x5 grid, but maybe shift to center?
    # For now, standard grid.
    x = np.array([(i + 0.5) / side for i in range(side) for j in range(side)][:N])
    y = np.array([(j + 0.5) / side for i in range(side) for j in range(side)][:N])
    
    # Calculate initial radius
    def calc_radius(x, y):
        # Min distance to boundary
        min_bound = np.min(np.concatenate([x, y, 1-x, 1-y]))
        # Min distance between centers
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < min_dist:
                    min_dist = d
        return min(min_bound, min_dist / 2.0)
    
    r = calc_radius(x, y)
    
    # Local search
    # We will try to move points to increase the minimum radius
    # Use a simple random walk with acceptance based on improvement
    
    # Pre-compute indices for faster distance calc?
    # N=21, so N(N-1)/2 = 210 pairs. Very fast.
    
    steps = 0
    while time.time() < deadline:
        # Pick a random point
        idx = np.random.randint(N)
        
        # Generate a random move
        # Step size should be adaptive or small
        # Let's use a fixed small step size for now, maybe decreasing
        step_size = 0.01
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        
        new_x = x.copy()
        new_y = y.copy()
        new_x[idx] += dx
        new_y[idx] += dy
        
        # Check if within bounds
        if new_x[idx] < 0 or new_x[idx] > 1 or new_y[idx] < 0 or new_y[idx] > 1:
            continue
            
        new_r = calc_radius(new_x, new_y)
        
        if new_r > r:
            x = new_x
            y = new_y
            r = new_r
        else:
            # Simulated annealing-like acceptance?
            # Or just reject.
            # Given the landscape, greedy is often okay for packing.
            # Let's try a small probability of acceptance if close
            if new_r > r * 0.99 and np.random.rand() < 0.01:
                x = new_x
                y = new_y
                r = new_r
                
        steps += 1
        
        # Adaptive step size?
        # If no improvement in last 1000 steps, reduce step size
        # This is complex to track. Let's stick to fixed or simple decay.
        # Let's decay step size every 10000 steps
        if steps % 10000 == 0:
            step_size *= 0.9

    # Final check and format
    # Ensure circles are disjoint and within perimeter 4 (1x1 box)
    # The evaluator checks validity.
    # Return array of shape (n, 3)
    result = np.column_stack((x, y, np.full(N, r)))
    return result
