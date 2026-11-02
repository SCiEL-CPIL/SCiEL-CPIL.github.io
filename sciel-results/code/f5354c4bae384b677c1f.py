import numpy as np

def construct():
    N = 32
    
    # Helper function to calculate the sum of radii for given centers
    # Uses iterative relaxation to solve for r_i
    def calc_score(centers):
        # centers: (N, 2)
        radii = np.zeros(N)
        # Precompute wall distances
        dists_wall = np.minimum(
            np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
            np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        )
        
        # Precompute pairwise distances
        # d[i, j] = distance between i and j
        d = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        
        # Iterate to find consistent radii
        # r_i <= d_ij - r_j  => r_i + r_j <= d_ij
        # We want to maximize sum(r_i)
        # This is equivalent to finding the maximum independent set in a graph? No.
        # It's a linear programming problem.
        # Simple iterative update:
        # r_i = min( d_wall_i, min_j (d_ij - r_j) )
        # This is a contraction mapping if we initialize r=0?
        # Let's try it.
        
        # Initialize radii to 0
        r = np.zeros(N)
        
        # Run relaxation
        for _ in range(50):
            # Update all radii based on current neighbors
            # For each i, r_i is limited by walls and neighbors
            # new_r_i = min(dists_wall[i], min_j (d[i, j] - r[j]))
            
            # Vectorized min over j
            # We need min_j (d[i, j] - r[j]) for each i
            # Let M[i, j] = d[i, j] - r[j]
            # We want min_j M[i, j] for j != i
            # Set diagonal to inf
            M = d.copy()
            M[np.arange(N), np.arange(N)] = np.inf
            min_neighbor_dist = np.min(M - r[np.newaxis, :], axis=1)
            
            # Combine with wall distance
            new_r = np.minimum(dists_wall, min_neighbor_dist)
            
            # Ensure non-negative
            new_r = np.maximum(0, new_r)
            
            # Check for convergence (optional, but we just run fixed steps)
            r = new_r
            
        return np.sum(r), r

    # Initialization: Hexagonal lattice
    # Approximate hexagonal packing
    # Rows of circles
    # Let's try a grid first, then perturb, or start with hex
    # Hex lattice:
    # Row 0: y = 0.5/side?
    # Let's use a standard hex grid generation
    
    # Calculate optimal hex grid parameters
    # We want to fit N circles.
    # Let's try to generate a hex grid that fits in [0,1]^2
    
    # Simple approach: Grid initialization, then SA will fix it.
    # But a good init helps.
    # Let's use a "Jittered Grid" or just a Grid.
    side = int(np.ceil(np.sqrt(N)))
    # Ensure we have N points
    # If side*side > N, we trim.
    
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Add small jitter to break symmetry
                x += np.random.uniform(-0.01, 0.01)
                y += np.random.uniform(-0.01, 0.01)
                centers.append([x, y])
                
    centers = np.array(centers)
    
    # Ensure inside bounds
    centers = np.clip(centers, 0.05, 0.95)
    
    # Simulated Annealing
    # Parameters
    T = 0.05
    cooling = 0.995
    steps = 5000
    move_size = 0.05
    
    # Initial score
    current_score, current_radii = calc_score(centers)
    
    best_score = current_score
    best_centers = centers.copy()
    best_radii = current_radii.copy()
    
    rng = np.random.default_rng(42)
    
    for step in range(steps):
        # Pick a random circle
        i = rng.integers(0, N)
        
        # Generate a move
        # Move size decreases slightly or stays constant?
        # Let's keep it constant for simplicity, T handles acceptance
        dx = rng.uniform(-move_size, move_size)
        dy = rng.uniform(-move_size, move_size)
        
        old_x = centers[i, 0]
        old_y = centers[i, 1]
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Keep inside box
        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        
        # If clipped, the move is effectively smaller
        centers[i, 0] = new_x
        centers[i, 1] = new_y
        
        # Calculate new score
        new_score, new_radii = calc_score(centers)
        
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            # Accept
            current_score = new_score
            current_radii = new_radii
            
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
                best_radii = current_radii.copy()
        else:
            # Reject, revert
            centers[i, 0] = old_x
            centers[i, 1] = old_y
            
        # Cool down
        T *= cooling
        
        # Optional: Reduce move size
        # move_size *= 0.999
        
    # Final result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_centers[i, 0], best_centers[i, 1], best_radii[i]]
        
    return result
