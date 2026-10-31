"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initialization: Hexagonal close packing
    # We want to place N points in [0,1]^2.
    # A hexagonal lattice has density sqrt(3)/2.
    # Let's generate a grid of candidate points on a hex lattice.
    
    # Generate points in a hex lattice covering [0,1]x[0,1] with some margin
    # Spacing s. Number of points approx Area / (s^2 * sqrt(3)/2)
    # For N=26, Area=1, s approx sqrt(2*1/(26*sqrt(3)/2)) = sqrt(4/(26*1.732)) approx 0.27
    # Let's try spacing 0.25 or 0.3
    
    def generate_hex_points(spacing=0.25):
        points = []
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        # Cover range [-0.5, 1.5] to ensure we have enough points to pick from
        for i in range(-2, int(1.5 / dx) + 3):
            for j in range(-2, int(1.5 / dy) + 3):
                x = i * dx + (j % 2) * (dx / 2)
                y = j * dy
                # Keep points roughly inside or near the square
                if -0.2 <= x <= 1.2 and -0.2 <= y <= 1.2:
                    points.append((x, y))
        return np.array(points)
    
    # We need exactly N points. 
    # Strategy: Generate a dense hex grid, then select N points that maximize the minimum distance (max-min dist) or just randomly sample a subset of a well-spread grid?
    # Better: Start with a grid of N points arranged in a way that approximates hex packing.
    # Let's use a simple grid first, then optimize. The optimization will move them to hex-like positions.
    
    # Initial guess: Uniform grid
    side = int(np.ceil(np.sqrt(N)))
    # Adjust grid to fit N
    # E.g. 5x5 = 25, add 1? Or 6x5=30, remove 4?
    # Let's use a 5x6 grid and pick 26.
    # Or just use a 5x5 grid and add one in the center? 
    # Actually, a better initial guess is to take a hex grid and pick the N points closest to a uniform distribution.
    
    # Let's try a specific hex-like initialization for 26 circles.
    # 26 is 2 * 13. Not a perfect square.
    # 5 rows of ~5?
    # Let's just use a standard grid initialization and let the optimizer do the work.
    # The optimizer is the key.
    
    def get_initial_points():
        # Create a grid of 5x6 = 30 points, then randomly select 26?
        # Or 5x5=25 and add one?
        # Let's use a 5x6 grid and drop the 4 corners to keep center density?
        # Simpler: Uniform grid 5x6.
        xs = np.linspace(0.1, 0.9, 6)
        ys = np.linspace(0.1, 0.9, 5)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        # We have 30 points. We need 26.
        # Remove 4 points. Which ones? 
        # Maybe the ones closest to each other? Or just random?
        # Let's keep the central ones.
        # Calculate distance from center (0.5, 0.5)
        dists = np.linalg.norm(pts - np.array([0.5, 0.5]), axis=1)
        # Sort by distance, keep closest 26?
        idx = np.argsort(dists)
        return pts[idx[:N]]
    
    x0 = get_initial_points()
    
    # Objective function: Maximize sum of radii
    # r_i = min( dist to boundary, min_{j != i} dist(i,j)/2 )
    
    def calculate_radii(points):
        # points: (N, 2)
        n = points.shape[0]
        radii = np.zeros(n)
        # Distance to boundary
        dist_boundary = np.min(np.column_stack([
            points[:, 0], 
            1 - points[:, 0], 
            points[:, 1], 
            1 - points[:, 1]
        ]), axis=1)
        
        # Distance to other points
        # Compute pairwise distances
        # dist[i, j] = ||p_i - p_j||
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf so it doesn't affect min
        np.fill_diagonal(dists, np.inf)
        
        min_dist_neighbor = np.min(dists, axis=1)
        radii_from_neighbors = min_dist_neighbor / 2.0
        
        radii = np.minimum(dist_boundary, radii_from_neighbors)
        return radii
    
    def objective(params):
        # params: flattened (N, 2)
        points = params.reshape(-1, 2)
        radii = calculate_radii(points)
        return -np.sum(radii) # Minimize negative sum
        
    # Optimization
    # Use Nelder-Mead for derivative-free optimization
    # Since N=26, dimension is 52. Nelder-Mead can be slow in high dims, but often works for packing.
    # Alternative: Gradient descent with subgradient? Or just random perturbations?
    # Given the non-smooth nature, a simple local search with random perturbations might be more robust and faster to implement correctly.
    
    def local_search(points, max_iters=2000):
        current_points = points.copy()
        current_score = -objective(current_points.ravel())
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = np.random.randint(N)
            # Random step size, decreasing over time
            step_size = 0.05 * (1 - i / max_iters) + 0.001
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            
            new_points = current_points.copy()
            new_points[idx, 0] += dx
            new_points[idx, 1] += dy
            
            # Keep inside [0,1]
            new_points = np.clip(new_points, 0, 1)
            
            new_score = -objective(new_points.ravel())
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
        
        return current_points, current_score

    # Run multiple local searches from different starts?
    # Or just one good one?
    # Let's run a few restarts.
    
    best_points = x0
    best_score = -objective(x0.ravel())
    
    # Try 5 different initializations
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Perturb initial points
        if restart == 0:
            start_pts = x0
        else:
            # Random jitter
            start_pts = x0 + np.random.uniform(-0.1, 0.1, x0.shape)
            start_pts = np.clip(start_pts, 0.05, 0.95)
            
        # Run local search
        # Increase iterations for first run, less for others
        iters = 5000 if restart == 0 else 2000
        pts, score = local_search(start_pts, max_iters=iters)
        
        if score > best_score:
            best_score = score
            best_points = pts
            
        # Check time
        if time.time() > deadline - 10:
            break

    # Final Polish: Try to expand radii slightly if possible?
    # The local search already maximizes the sum of constrained radii.
    # The radii are determined by the positions.
    # We just need to return the positions and the calculated radii.
    
    final_radii = calculate_radii(best_points)
    
    # Construct result array (x, y, r)
    result = np.column_stack((best_points, final_radii))
    
    # Validate
    # Check if any radius is negative or if circles overlap
    # (They shouldn't by construction, but floating point errors might occur)
    
    return result

# EVOLVE-BLOCK-END
