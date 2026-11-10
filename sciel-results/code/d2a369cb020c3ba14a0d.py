import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(positions):
    """Compute the maximum common radius for given circle centers."""
    # Boundary constraint
    min_bound = 0.5 - np.max(np.abs(positions))
    
    # Pairwise distance constraint
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    return min(min_bound, min_dist / 2.0)

def generate_hexagonal_initial(N, square_half=0.5):
    """Generate a hexagonal packing initialization."""
    positions = []
    # Hexagonal packing: rows are offset
    # Radius of circles in hex packing relative to spacing: r = spacing / 2
    # Distance between rows: spacing * sqrt(3)/2
    
    # We want to fit N circles.
    # Let's try to arrange them in a rectangular grid with hex offsets.
    # Approximate number of rows: sqrt(N)
    # Approximate number of cols: sqrt(N)
    
    # Start with a dense grid and refine
    # Let's use a simple grid first, then hexagonalize
    
    # Better: Use a known good configuration for N=50?
    # No, let's generate a hex grid that covers the square.
    
    # Spacing s such that we fit roughly sqrt(N) rows and cols
    # Let's try s = 1.0 / sqrt(N) * factor
    # For N=50, sqrt(50) ~ 7.07
    # If we use 7x7 grid, spacing ~ 1/7 = 0.142
    # Hex spacing is slightly tighter.
    
    # Let's create a hex grid covering [-0.5, 0.5] x [-0.5, 0.5]
    # and pick the N points closest to the center.
    
    max_coord = 0.5
    spacing = 0.15 # Initial guess, will be adjusted
    
    # Generate points in a hex grid
    points = []
    row = 0
    while True:
        y = -max_coord + row * spacing * np.sqrt(3) / 2
        if y > max_coord:
            break
        x_offset = (row % 2) * spacing / 2
        col = 0
        while True:
            x = -max_coord + col * spacing + x_offset
            if x > max_coord:
                break
            points.append([x, y])
            col += 1
        row += 1
        if len(points) > 200: # Safety break
            break
            
    points = np.array(points)
    
    # If we have more points than N, select the N closest to the center
    if len(points) > N:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        points = points[indices]
    elif len(points) < N:
        # If too few, increase density (decrease spacing) and retry
        # For now, just pad with random if necessary, but 0.15 spacing should be enough for 50
        # 1/0.15 = 6.6 units. Area ~ 44. Hex density is higher.
        # Let's just use what we have if < N? No, we need N.
        # Let's restart with smaller spacing
        return generate_hexagonal_initial(N, square_half, spacing * 0.9)
        
    # Scale points to fit exactly within [-0.5, 0.5] with some margin
    # Actually, we want to maximize radius, so we want them as spread out as possible?
    # No, we want to pack them tight.
    # The hex grid above is fixed to the square. 
    # To optimize, we should probably center the grid and allow the optimizer to move them.
    
    # Center the points
    center = np.mean(points, axis=0)
    points = points - center
    
    # Scale to fit within [-0.48, 0.48] to avoid boundary issues initially
    max_abs = np.max(np.abs(points))
    if max_abs > 0.48:
        points = points * (0.48 / max_abs)
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # 1. Initial placement: Hexagonal grid
    positions = generate_hexagonal_initial(N)
    
    current_radius = compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    # 2. Simulated Annealing
    T = 0.05  # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.9995
    step_size = 0.01
    
    iter_count = 0
    max_iter = 2000000 # Limit by time instead
    
    np.random.seed(42)
    
    while time.time() < deadline:
        iter_count += 1
        
        # Update temperature
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: Restart from best if stuck
            if iter_count % 10000 == 0:
                 positions = best_positions + np.random.normal(0, 0.01, (N, 2))
                 positions = np.clip(positions, -0.49, 0.49)
                 current_radius = compute_radius(positions)
        
        # Pick a random circle to move
        idx = np.random.randint(0, N)
        
        # Generate a random move
        # Move size should be related to temperature
        move_scale = step_size * (1.0 + T)
        delta = np.random.normal(0, move_scale, 2)
        
        old_pos = positions[idx].copy()
        new_pos = old_pos + delta
        
        # Check boundaries
        if np.any(np.abs(new_pos) > 0.5):
            # Reject if out of bounds
            continue
            
        # Try the move
        positions[idx] = new_pos
        new_radius = compute_radius(positions)
        
        # Calculate energy difference (we want to maximize radius, so minimize -radius)
        delta_energy = current_radius - new_radius
        
        # Acceptance criterion
        if delta_energy < 0: # Improvement
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_positions = positions.copy()
        else: # Worse
            # Accept with probability exp(-delta_energy / T)
            if np.random.rand() < np.exp(-delta_energy / T):
                current_radius = new_radius
            else:
                # Reject
                positions[idx] = old_pos
                
    return best_positions
