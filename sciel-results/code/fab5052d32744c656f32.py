"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the maximum common radius for the given centers."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: square is [-0.5, 0.5]^2
    min_x = 0.5 - np.max(centers[:, 0])
    min_y = 0.5 - np.max(centers[:, 1])
    min_x2 = np.min(centers[:, 0]) + 0.5
    min_y2 = np.min(centers[:, 1]) + 0.5
    min_wall = min(min_x, min_y, min_x2, min_y2)
    
    return min(min_pair, min_wall)


def _negative_radius(centers_flat):
    """Objective function to minimize (negative radius)."""
    centers = centers_flat.reshape(N, 2)
    return -_compute_radius(centers)


def _initial_hex():
    """Generate an initial hexagonal lattice arrangement."""
    # Hexagonal packing: arrange in rows with offset
    # For 50 circles, use roughly 7-8 rows
    # Let's try a hexagonal pattern
    
    # Start with a grid-like hex pattern
    # Rows alternate offset
    # We want to fit in [-0.5, 0.5]^2
    
    # Try different configurations
    # 50 = 7*7 + 1, so 7 rows of 7 with one extra, or 8 rows
    
    # Let's use a hexagonal lattice with variable spacing
    # For now, use a jittered square grid as initial guess
    # and let the optimizer do the work
    
    m = int(np.ceil(np.sqrt(N)))
    # Create grid
    positions = []
    for i in range(m):
        for j in range(m):
            if len(positions) < N:
                x = (j + 0.5) / m - 0.5
                y = (i + 0.5) / m - 0.5
                positions.append([x, y])
    
    return np.array(positions[:N])


def _initial_hexagonal_lattice():
    """Generate a hexagonal lattice initial configuration."""
    # Hexagonal close packing
    # Row spacing: h = s * sqrt(3)/2 where s is horizontal spacing
    # Alternate rows offset by s/2
    
    # Try to fit 50 circles
    # Let's use rows of 6 or 7 circles
    # 50 = 7 + 7 + 7 + 7 + 7 + 6 + 6 + 3? Not clean
    
    # Use a more systematic approach:
    # Place circles in a hexagonal pattern, then scale to fit
    
    points = []
    # Use a hexagonal grid
    # For row i, place circles at x = (j + 0.5*(i%2)) * dx + offset
    # y = i * dy
    
    # Let's try 8 rows with varying counts
    # 50 circles: rows of 7,7,7,7,7,7,7,1? No
    # 50 = 6*8 + 2, or 7*7 + 1
    
    # Simpler: use a square grid with hexagonal offset
    # 7x7 grid = 49, add 1 more
    m = 7
    for i in range(m):
        for j in range(m):
            if len(points) < N:
                x = (j - (m-1)/2) * 0.15  # spacing
                y = (i - (m-1)/2) * 0.13
                # Hex offset for odd rows
                if i % 2 == 1:
                    x += 0.075
                points.append([x, y])
    
    # Add remaining circles if needed
    while len(points) < N:
        x = np.random.uniform(-0.4, 0.4)
        y = np.random.uniform(-0.4, 0.4)
        points.append([x, y])
    
    return np.array(points[:N])


def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    # Initial configuration
    centers = _initial_hexagonal_lattice()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Use Nelder-Mead optimization with multiple restarts
    # First, try to improve from the initial guess
    
    flat = centers.flatten()
    
    # Run optimization with time check
    try:
        result = minimize(
            _negative_radius,
            flat,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        if result.fun < -best_radius:
            best_radius = -result.fun
            best_centers = result.x.reshape(N, 2)
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            return best_centers
    
    except Exception:
        pass
    
    # Try multiple random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() - start_time > deadline - 20:
            break
        
        # Generate a new random initial configuration
        # Use a jittered grid
        m = int(np.ceil(np.sqrt(N)))
        new_centers = []
        for i in range(m):
            for j in range(m):
                if len(new_centers) < N:
                    x = (j + 0.5) / m - 0.5 + np.random.uniform(-0.02, 0.02)
                    y = (i + 0.5) / m - 0.5 + np.random.uniform(-0.02, 0.02)
                    new_centers.append([x, y])
        
        new_centers = np.array(new_centers[:N])
        
        try:
            result = minimize(
                _negative_radius,
                new_centers.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result.fun < -best_radius:
                best_radius = -result.fun
                best_centers = result.x.reshape(N, 2)
                
        except Exception:
            continue
    
    # Final polish: try to nudge points to improve
    if time.time() - start_time < deadline - 5:
        # Try a few more iterations with tighter tolerance
        try:
            result = minimize(
                _negative_radius,
                best_centers.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            
            if result.fun < -best_radius:
                best_radius = -result.fun
                best_centers = result.x.reshape(N, 2)
                
        except Exception:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
