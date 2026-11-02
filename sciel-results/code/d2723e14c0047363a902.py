"""Improved packing: hexagonal lattice init + L-BFGS-B with smooth soft-min objective."""
import numpy as np
import time
import math

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time


def _softmin(x, alpha=1000.0):
    """Smooth approximation of min(x) using log-sum-exp (negative)."""
    # For large alpha, -log(sum(exp(-alpha*x)))/alpha -> min(x)
    x_min = x.min()
    # Shift for numerical stability
    x_shifted = x - x_min
    # Use log-sum-exp trick
    max_val = x_shifted.max()
    log_sum = np.log(np.sum(np.exp(-alpha * (x_shifted)))) + alpha * max_val
    return x_min - log_sum / alpha


def _compute_radius(centers):
    """Compute the radius for a given configuration."""
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = dists.min()
    
    # Distance to walls: for each center, min distance to any wall
    # Square is [-0.5, 0.5]^2
    wall_dists = np.minimum(0.5 - np.abs(centers), axis=1)
    min_wall_dist = wall_dists.min()
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _objective_smooth(centers_flat, alpha=2000.0):
    """Smooth objective: negative of soft-min of constraints."""
    centers = centers_flat.reshape(-1, 2)
    
    # Pairwise distances (off-diagonal)
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(centers), axis=1)
    
    # All constraints: pairwise distances and wall distances
    # We want to maximize min of (pairwise_dists/2, wall_dists)
    # So minimize negative soft-min
    all_constraints = np.concatenate([dists.ravel() / 2.0, wall_dists.ravel()])
    
    return -_softmin(all_constraints, alpha)


def _hexagonal_init(n, square_half=0.5):
    """Generate hexagonal packing initialization for n circles in a square."""
    # Hexagonal packing: rows offset by half spacing
    # Try different row spacings to fit n circles
    best_centers = None
    best_radius = 0
    
    # The hexagonal packing has spacing d between centers
    # Row spacing is d * sqrt(3)/2
    # For n circles, we need to figure out how many rows
    
    # Try various configurations
    for rows in range(1, int(math.sqrt(n)) + 5):
        cols_per_row = []
        remaining = n
        for r in range(rows):
            if r < rows - 1:
                cols = int(math.ceil(remaining / (rows - r)))
            else:
                cols = remaining
            cols_per_row.append(cols)
            remaining -= cols
        
        if remaining < 0:
            continue
        
        # Compute required spacing
        # Width needed: max_cols * d
        # Height needed: (rows-1) * d * sqrt(3)/2
        max_cols = max(cols_per_row)
        
        # Fit in square: 
        # d * max_cols <= 1 (width)
        # d * (rows-1) * sqrt(3)/2 <= 1 (height)
        d_width = 1.0 / max_cols
        d_height = 1.0 / ((rows - 1) * math.sqrt(3) / 2) if rows > 1 else 1.0
        d = min(d_width, d_height)
        
        # Generate centers
        centers = []
        y_offset = 0.5 - (rows - 1) * d * math.sqrt(3) / 4  # Center vertically
        for r, ncols in enumerate(cols_per_row):
            x_offset = 0.5 - (ncols - 1) * d / 2  # Center horizontally
            if r % 2 == 1:
                x_offset += d / 2  # Offset odd rows
            for c in range(ncols):
                x = x_offset + c * d
                y = y_offset + r * d * math.sqrt(3) / 2
                centers.append([x, y])
        
        centers = np.array(centers[:n])
        
        # Check if all inside square
        if np.all(centers >= -0.5 - 1e-9) and np.all(centers <= 0.5 + 1e-9):
            r = _compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers
    
    if best_centers is None:
        # Fallback to grid
        m = int(math.ceil(math.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:n]
    
    return best_centers


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with hexagonal packing
    centers = _hexagonal_init(N)
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # L-BFGS-B optimization with smooth objective
    # Use multiple restarts with different alpha values
    alphas = [500, 1000, 2000, 5000]
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        
        x0 = best_centers.flatten()
        
        def obj(x):
            if time.time() > deadline:
                return 1e10
            return _objective_smooth(x, alpha)
        
        result = minimize(
            obj,
            x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < obj(x0):
            new_centers = result.x.reshape(-1, 2)
            # Clamp to square
            new_centers = np.clip(new_centers, -0.5, 0.5)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers
    
    # Final polish with higher alpha
    if time.time() < deadline - 10:
        for alpha in [10000, 20000]:
            if time.time() > deadline:
                break
            
            x0 = best_centers.flatten()
            
            def obj(x):
                if time.time() > deadline:
                    return 1e10
                return _objective_smooth(x, alpha)
            
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < obj(x0):
                new_centers = result.x.reshape(-1, 2)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_centers
# EVOLVE-BLOCK-END
