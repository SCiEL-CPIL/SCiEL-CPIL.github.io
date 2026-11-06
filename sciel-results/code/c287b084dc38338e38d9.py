"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_SECONDS = 110.0  # Safety margin before 150s limit

def compute_radius(pts):
    """
    Computes the maximum radius for the given centers.
    pts: (N, 2) array of centers.
    Returns r such that circles of radius r are disjoint and inside [-0.5, 0.5]^2.
    """
    # 1. Distance to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    min_bound = np.min(np.minimum(d_x, d_y))
    
    # 2. Distance between centers
    # Use broadcasting for N=50
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_inter = np.sqrt(np.min(dist_sq))
    
    r_inter = min_dist_inter / 2.0
    r = np.minimum(r_inter, min_bound)
    
    return r

def objective(x):
    """
    Objective function to minimize: -radius.
    """
    pts = x.reshape(-1, 2)
    r = compute_radius(pts)
    return -r

def generate_hex_initialization():
    """
    Generates a hexagonal-like initialization with alternating row lengths.
    This is often a better starting point than a regular square grid for circle packing.
    """
    # Target N=50
    # A common good layout for ~50 circles is rows of 7, 6, 7, 6, 7, 6, 7, 4
    # Total: 7+6+7+6+7+6+7+4 = 50
    row_lengths = [7, 6, 7, 6, 7, 6, 7, 4]
    
    points = []
    num_rows = len(row_lengths)
    
    # Vertical spacing
    # We want to fit 8 rows in height 1.
    # Rough spacing is 1/8 = 0.125.
    # Hexagonal spacing y = s * sqrt(3)/2. If s ~ 0.14, y ~ 0.12.
    # Let's start with a base spacing and adjust.
    s = 0.14 # Approximate diameter
    y_spacing = s * np.sqrt(3) / 2.0
    
    # Center vertically
    total_height = (num_rows - 1) * y_spacing
    y_start = -total_height / 2.0
    
    for i, length in enumerate(row_lengths):
        y = y_start + i * y_spacing
        # Horizontal spacing
        x_spacing = s
        # Center horizontally
        total_width = (length - 1) * x_spacing
        x_start = -total_width / 2.0
        
        # Offset odd rows for hexagonal packing
        offset = 0.0
        if i % 2 != 0:
            offset = x_spacing / 2.0
            
        for j in range(length):
            x = x_start + j * x_spacing + offset
            points.append([x, y])
            
    points = np.array(points)
    
    # Ensure we have exactly N points
    if len(points) > N:
        points = points[:N]
    elif len(points) < N:
        # Pad with random points near the center if needed
        num_pad = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(num_pad, 2))
        points = np.vstack([points, random_pts])
        
    # Clip to ensure inside square
    points = np.clip(points, -0.49, 0.49)
    
    return points

def generate_grid_initialization():
    """
    Generates a standard grid initialization.
    """
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    if len(grid) > N:
        grid = grid[:N]
    elif len(grid) < N:
        num_pad = N - len(grid)
        random_pts = np.random.uniform(-0.4, 0.4, size=(num_pad, 2))
        grid = np.vstack([grid, random_pts])
        
    return grid

def local_refinement(pts, max_iters=1000):
    """
    Simple local refinement: find the closest pair and push them apart.
    This helps escape local minima where two circles are slightly overlapping.
    """
    pts = pts.copy()
    for _ in range(max_iters):
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Find min distance pair
        min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        i, j = min_idx
        if i == j:
            break
            
        d = pts[j] - pts[i]
        dist = np.sqrt(np.sum(d*d))
        
        if dist < 1e-9:
            # If coincident, push randomly
            d = np.array([1.0, 0.0])
            dist = 1.0
            
        # Direction to push
        direction = d / dist
        
        # Step size: small fraction of the gap or a fixed small step
        # We want to increase the distance.
        step = 0.001
        
        new_i = pts[i] - direction * step
        new_j = pts[j] + direction * step
        
        # Check boundaries
        if np.max(np.abs(new_i)) <= 0.5 and np.max(np.abs(new_j)) <= 0.5:
            # Verify improvement in radius
            old_r = compute_radius(pts)
            pts[i] = new_i
            pts[j] = new_j
            new_r = compute_radius(pts)
            
            if new_r > old_r:
                # Improvement, keep it
                continue
            else:
                # Revert
                pts[i] = pts[i] + direction * step
                pts[j] = pts[j] - direction * step
                # If no improvement, maybe try a different pair? 
                # For simplicity, we break or continue to next iteration.
                # Breaking might be too aggressive. Let's just continue.
                pass
        else:
            # Boundary violation, try pushing in the opposite direction or skip
            pass
            
    return pts

def optimize_from_start(start_pts, start_time, deadline):
    """
    Runs local optimization starting from start_pts.
    """
    x0 = start_pts.flatten()
    
    # Use Powell method
    try:
        result = minimize(
            objective, 
            x0, 
            method='Powell', 
            options={'maxiter': 500, 'ftol': 1e-9, 'xatol': 1e-9}
        )
        best_pts = result.x.reshape(-1, 2)
        best_pts = np.clip(best_pts, -0.5, 0.5)
        
        # Local refinement
        if time.time() < deadline - 5:
            best_pts = local_refinement(best_pts, max_iters=200)
            
    except Exception:
        best_pts = start_pts
        
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # List of initializations
    inits = []
    
    # 1. Hexagonal (Strongest)
    inits.append(generate_hex_initialization())
    
    # 2. Grid
    inits.append(generate_grid_initialization())
    
    # 3. Random
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 2)))
    
    # 4. Hex with different spacing (perturbed)
    hex_pts = generate_hex_initialization()
    hex_pts += np.random.normal(0, 0.01, size=hex_pts.shape)
    hex_pts = np.clip(hex_pts, -0.49, 0.49)
    inits.append(hex_pts)
    
    best_pts = inits[0]
    best_r = compute_radius(best_pts)
    
    for i, init in enumerate(inits):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 10: # Leave 10s for final check/return
            break
            
        # Optimize
        try:
            optimized_pts = optimize_from_start(init, start_time, deadline)
            r = compute_radius(optimized_pts)
            
            if r > best_r:
                best_r = r
                best_pts = optimized_pts
                
        except Exception:
            pass
            
    # Final safety clip
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
