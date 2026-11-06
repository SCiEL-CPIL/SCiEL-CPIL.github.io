"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centres):
    """Compute the maximum common radius for given centres (vectorized)."""
    if centres.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check boundaries: centres must be in [-0.5, 0.5]^2
    if np.any(centres < -0.5 - 1e-12) or np.any(centres > 0.5 + 1e-12):
        return 0.0
    
    # Distance to boundaries
    boundary_dist = np.minimum(
        np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
        np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1])
    )
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances using broadcasting
    # diff[i, j] = centres[i] - centres[j]
    # dist[i, j] = ||centres[i] - centres[j]||
    # We only need upper triangle, but full matrix is easier to vectorize
    # To save memory/time, we can just compute all and take min of upper triangle
    # Or just min of all non-diagonal.
    
    # Efficient way:
    # d2 = sum((c_i - c_j)^2)
    # We can use: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    # But simple broadcasting is fine for N=50.
    
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Zero out diagonal to avoid min being 0
    np.fill_diagonal(dist, np.inf)
    
    min_pair = np.min(dist) / 2.0
    
    return min(min_boundary, min_pair)

def _negative_radius(centres_flat):
    """Objective to minimize (negative of radius)."""
    centres = centres_flat.reshape(N, 2)
    r = _compute_radius(centres)
    return -r

def _hexagonal_initial(n_rows, row_counts, total_n):
    """
    Create a hexagonal lattice packing with given row counts.
    Returns centres array of shape (total_n, 2).
    """
    # We'll place rows from top to bottom
    # Each row has 'row_counts[i]' circles
    # Hexagonal: alternating rows are offset by half the horizontal spacing
    # Horizontal spacing between circle centres in a row: 2*r
    # Vertical spacing between rows: sqrt(3)*r
    # We want to find r and offsets such that all fit in [-0.5, 0.5]^2
    
    # Start with an initial guess for r
    r = 0.07
    h = 2 * r  # horizontal spacing
    v = np.sqrt(3) * r  # vertical spacing
    
    centres = []
    for i, count in enumerate(row_counts):
        # Row i: y position
        y = 0.5 - v/2 - i * v
        # x positions: centered
        xs = np.arange(count) * h - (count - 1) * h / 2
        # Offset alternating rows
        if i % 2 == 1:
            xs += h / 2
        for x in xs:
            centres.append([x, y])
    
    centres = np.array(centres[:total_n])
    return centres

def _optimize_from_initial(centres0, deadline):
    """Optimize centres from an initial configuration."""
    def obj(x):
        c = x.reshape(N, 2)
        return -_compute_radius(c)
    
    x0 = centres0.flatten()
    
    # Use L-BFGS-B with reduced iterations to ensure speed
    # The vectorized objective is fast, but 1000 iters might still be slow if each iter is not tiny
    # Let's cap it to 200 iters per start, and rely on multiple starts if time permits.
    
    result = minimize(
        obj,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    best_c = result.x.reshape(N, 2)
    best_r = _compute_radius(best_c)
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_centres = None
    best_r = 0.0
    
    # Try various hexagonal lattice configurations
    # Total circles: 50
    # Try different row count distributions
    
    row_configs = [
        [7, 6, 7, 6, 7, 6, 7, 4],   # 50
        [6, 7, 6, 7, 6, 7, 6, 5],   # 50
        [7, 7, 6, 7, 6, 7, 6, 4],   # 50
        [8, 7, 6, 7, 6, 6, 7, 3],   # 50
        [7, 6, 7, 7, 6, 6, 7, 4],   # 50
        [6, 7, 7, 6, 7, 6, 6, 5],   # 50
    ]
    
    for config in row_configs:
        if sum(config) != N:
            continue
        if time.time() > deadline:
            break
        
        # Generate initial hexagonal packing
        c0 = _hexagonal_initial(len(config), config, N)
        
        # Optimize
        c_opt, r_opt = _optimize_from_initial(c0, deadline)
        
        if r_opt > best_r:
            best_r = r_opt
            best_centres = c_opt.copy()
    
    # If we didn't find anything good, use a grid
    if best_centres is None or best_r < 0.05:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:N]
        best_r = _compute_radius(best_centres)
    
    # Final check: ensure validity
    if not np.all(np.isfinite(best_centres)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    return best_centres
# EVOLVE-BLOCK-END
