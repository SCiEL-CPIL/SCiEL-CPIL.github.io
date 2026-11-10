"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def compute_radius(centers):
    """Compute the largest common radius for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    # Distance to walls
    wall_dists = np.min(
        np.stack([
            centers[:, 0] + 0.5,
            0.5 - centers[:, 0],
            centers[:, 1] + 0.5,
            0.5 - centers[:, 1]
        ]), axis=0
    )
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use broadcasting for all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)


def hex_grid_init(n):
    """Generate a hexagonal grid initialization for n circles in unit square."""
    # For hexagonal packing, we want to fit as many as possible
    # Try different row counts and find the best fit
    best_centers = None
    best_radius = -1
    
    # The square is [-0.5, 0.5]^2
    # Hex grid: rows offset by half spacing
    # Let's try to place circles in a hex pattern
    
    # For n=50, we can try various configurations
    # A 7x7 hex grid would give 49 circles, close to 50
    # Let's try to create a hex grid with approximately sqrt(n) rows
    
    # Actually, let's use a simpler approach: start with a square grid
    # and then use optimization to improve it
    m = int(np.ceil(np.sqrt(n)))
    
    # Create a hex-like pattern
    # Rows with alternating offsets
    centers = []
    
    # Try to fit n circles in a hex pattern
    # For a hex grid with row spacing dy and column spacing dx:
    # Even rows: x = -0.5 + r + i*dx for i in range(cols)
    # Odd rows: x = -0.5 + r + dx/2 + i*dx for i in range(cols-1 or cols)
    
    # Let's try a 7-row hex grid (gives ~49-50 circles)
    for n_rows in range(5, 11):
        # For n_rows rows, we need to fit n circles
        # In hex packing, odd rows have one fewer circle (or same)
        # Let's say even rows have c circles, odd rows have c or c-1
        
        # Try different numbers of columns
        for c in range(5, 12):
            # Calculate positions
            centers_list = []
            for row in range(n_rows):
                if row % 2 == 0:
                    n_cols = c
                    x_offset = 0.0
                else:
                    n_cols = c - 1
                    x_offset = 0.5  # half spacing offset (relative)
                
                if n_cols <= 0:
                    continue
                    
                # x positions: spread across [-0.5, 0.5]
                if n_cols == 1:
                    xs = [0.0]
                else:
                    # Evenly spaced with margin
                    xs = np.linspace(-0.5 + 0.05, 0.5 - 0.05, n_cols).tolist()
                
                # y position
                y = -0.5 + 0.05 + row * (0.9 / max(1, n_rows - 1))
                
                for x in xs:
                    if x_offset > 0:
                        x += 0.05  # small offset
                    centers_list.append([x, y])
            
            if len(centers_list) >= n:
                centers_arr = np.array(centers_list[:n])
                r = compute_radius(centers_arr)
                if r > best_radius:
                    best_radius = r
                    best_centers = centers_arr
    
    if best_centers is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:n]
    
    return best_centers


def simulated_annealing(centers, time_limit=100.0):
    """Run simulated annealing to maximize the minimum radius."""
    start_time = time.time()
    n = len(centers)
    
    # Initial temperature
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.999
    
    best_centers = centers.copy()
    best_radius = compute_radius(centers)
    current_radius = best_radius
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Generate a random perturbation
        i = rng.integers(0, n)
        delta = rng.normal(0, T, size=2)
        
        new_centers = centers.copy()
        new_centers[i] += delta
        
        # Keep within bounds
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = compute_radius(new_centers)
        
        # Acceptance criterion
        if new_radius >= current_radius:
            accept = True
        else:
            delta_e = current_radius - new_radius
            if delta_e < 1e-12:
                accept = True
            else:
                accept = np.random.random() < np.exp(-delta_e / T)
        
        if accept:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        
        T *= cooling_rate
    
    return best_centers, best_radius


def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for safety
    
    # Start with hex grid initialization
    centers = hex_grid_init(N)
    best_radius = compute_radius(centers)
    
    # Run simulated annealing
    centers, radius = simulated_annealing(centers, time_limit=time_limit - 5.0)
    if radius > best_radius:
        best_radius = radius
        best_centers = centers
    else:
        best_centers = centers
    
    # Check time and return
    elapsed = time.time() - start_time
    if elapsed < time_limit - 2.0:
        # Try one more round with different random seed
        rng2 = np.random.default_rng(123)
        centers2 = centers + rng2.normal(0, 0.01, size=(N, 2))
        centers2 = np.clip(centers2, -0.5, 0.5)
        centers2, radius2 = simulated_annealing(centers2, time_limit=time_limit - elapsed - 2.0)
        if radius2 > best_radius:
            best_radius = radius2
            best_centers = centers2
    
    return best_centers
# EVOLVE-BLOCK-END
