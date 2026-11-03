"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def _score(points):
    """Compute the radius for a set of points in [-0.5, 0.5]^2."""
    # Check bounds
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    # Check finite
    if not np.all(np.isfinite(points)):
        return 0.0
    # Min distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    # Min pairwise distance
    if len(points) < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
    return min(min_pair / 2.0, min_boundary)


def _hex_grid(n):
    """Generate a hexagonal grid packing."""
    # Try to find a good hexagonal arrangement
    # For hexagonal packing, rows are offset
    # Let's try a few configurations
    best_score = 0
    best_points = None
    
    # Approximate hexagonal grid
    # Row spacing: sqrt(3)/2 * spacing
    # Column spacing: spacing
    # Offset every other row by spacing/2
    
    # We want to fit N points.
    # Let k be the number of rows.
    # Points per row varies.
    
    # Simple approach: generate a large hex grid and crop/adjust
    # Or optimize spacing and offset.
    
    # Let's try a grid with hexagonal symmetry
    # x = i * s + (j % 2) * s/2
    # y = j * s * sqrt(3)/2
    
    # We need to fit this in [-0.5, 0.5]^2.
    
    # Let's iterate over possible number of rows
    for rows in range(10, 30):
        # Average points per row
        cols = N // rows
        # Try to fit
        # Width needed: (cols - 1) * s + s/2 (for offset) approx cols * s
        # Height needed: (rows - 1) * s * sqrt(3)/2
        
        # Solve for s such that it fits in 1x1 square
        # s <= 1 / cols
        # s * sqrt(3)/2 * (rows - 1) <= 1  => s <= 1 / ((rows - 1) * sqrt(3)/2)
        
        s_x = 1.0 / cols
        s_y = 1.0 / ((rows - 1) * np.sqrt(3) / 2)
        s = min(s_x, s_y)
        
        # Generate points
        points = []
        for j in range(rows):
            offset = (j % 2) * s / 2
            for i in range(cols):
                x = -0.5 + s/2 + i * s + offset
                y = -0.5 + s * np.sqrt(3) / 4 + j * s * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points[:N])
        if len(points) == N:
            sc = _score(points)
            if sc > best_score:
                best_score = sc
                best_points = points.copy()
                
    return best_points, best_score


def _square_grid(n):
    """Generate a square grid packing."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:n]
    return points, _score(points)


def _refine(points, time_budget):
    """Simple local optimization to improve the packing."""
    current_score = _score(points)
    start_time = time.time()
    step = 0.01
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        for i in range(len(points)):
            # Try small moves
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = points.copy()
                    new_points[i] += np.array([dx, dy])
                    if np.all((new_points >= -0.5) & (new_points <= 0.5)):
                        new_score = _score(new_points)
                        if new_score > current_score:
                            current_score = new_score
                            points = new_points
                            improved = True
    return points, current_score


def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Start with hex grid
    hex_points, hex_score = _hex_grid(N)
    if hex_points is None:
        hex_points, hex_score = _square_grid(N)
        
    # Start with square grid
    sq_points, sq_score = _square_grid(N)
    
    best_points = hex_points if hex_score > sq_score else sq_points
    best_score = max(hex_score, sq_score)
    
    # Refine if time permits
    if time.time() < deadline - 10:
        best_points, best_score = _refine(best_points, deadline - time.time() - 5)
        
    return best_points
# EVOLVE-BLOCK-END
