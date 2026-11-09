# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations


def _check_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
    # Check bounds
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
        if s <= 0:
            return False
    # Check non-overlap
    for i in range(n):
        for j in range(i+1, n):
            cx_i, cy_i = centers[i]
            cx_j, cy_j = centers[j]
            s_i, s_j = sides[i], sides[j]
            t = (s_i + s_j)/2 - tol
            if abs(cx_i - cx_j) < t and abs(cy_i - cy_j) < t:
                return False
    return True


def _serpentine_packing(n=10):
    """
    Create an initial serpentine packing.
    Place squares in rows, alternating left-to-right and right-to-left.
    """
    # Start with a reasonable grid-like arrangement
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    
    centers = np.zeros((n, 2))
    sides = np.full(n, side)
    
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    
    # Serpentine: alternate direction each row
    for i in range(n):
        r = rows[i]
        c = cols[i]
        if r % 2 == 0:
            col_pos = c
        else:
            col_pos = (k - 1) - c
        centers[i, 0] = (col_pos + 0.5) * side
        centers[i, 1] = (r + 0.5) * side
    
    return centers, sides


def _max_side_for_position(centers, sides, i, tol=1e-7):
    """
    Given fixed positions of all squares except i, find the maximum side length
    for square i that maintains feasibility.
    """
    n = len(sides)
    cx_i, cy_i = centers[i]
    
    # Start with max possible from bounds
    max_s = min(2 * cx_i, 2 * (1 - cx_i), 2 * cy_i, 2 * (1 - cy_i))
    
    # Check against each other square
    for j in range(n):
        if j == i:
            continue
        cx_j, cy_j = centers[j]
        s_j = sides[j]
        
        # For no overlap: |cx_i - cx_j| >= (s_i + s_j)/2 OR |cy_i - cy_j| >= (s_i + s_j)/2
        # This means: s_i <= 2*|cx_i - cx_j| - s_j OR s_i <= 2*|cy_i - cy_j| - s_j
        # So s_i <= max(2*|cx_i - cx_j| - s_j, 2*|cy_i - cy_j| - s_j)
        dx = abs(cx_i - cx_j)
        dy = abs(cy_i - cy_j)
        constraint = max(2*dx - s_j, 2*dy - s_j)
        if constraint < max_s:
            max_s = constraint
    
    return max(0, max_s - tol)


def _coordinate_ascent(centers, sides, tol=1e-7, max_iter=500, time_limit=300):
    """
    1D coordinate ascent: for each square, try moving in x and y directions
    to find a position that allows a larger side length.
    """
    n = len(sides)
    best_sum = np.sum(sides)
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                    break
            
            cx_i, cy_i = centers[i]
            s_i = sides[i]
            
            # Try to increase side length by repositioning
            # First, find max side at current position
            new_s = _max_side_for_position(centers, sides, i, tol)
            
            if new_s > s_i + 1e-9:
                sides[i] = new_s
                best_sum = np.sum(sides)
                improved = True
                continue
            
            # Try moving in x direction
            # Sample several candidate x positions
            for dx in [0.01, -0.01, 0.02, -0.02, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15]:
                new_cx = cx_i + dx
                if new_cx < s_i/2 or new_cx > 1 - s_i/2:
                    continue
                old_cx = centers[i, 0]
                centers[i, 0] = new_cx
                new_s = _max_side_for_position(centers, sides, i, tol)
                if new_s > s_i + 1e-9:
                    sides[i] = new_s
                    best_sum = np.sum(sides)
                    improved = True
                    break
                else:
                    centers[i, 0] = old_cx
            
            # Try moving in y direction
            for dy in [0.01, -0.01, 0.02, -0.02, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15]:
                new_cy = cy_i + dy
                if new_cy < sides[i]/2 or new_cy > 1 - sides[i]/2:
                    continue
                old_cy = centers[i, 1]
                centers[i, 1] = new_cy
                new_s = _max_side_for_position(centers, sides, i, tol)
                if new_s > sides[i] + 1e-9:
                    sides[i] = new_s
                    best_sum = np.sum(sides)
                    improved = True
                    break
                else:
                    centers[i, 1] = old_cy
        
        if not improved:
            break
    
    return centers, sides, best_sum


def _random_packing(n=10, n_samples=50, time_limit=60):
    """
    Generate multiple random packings and return the best one.
    """
    best_centers = None
    best_sides = None
    best_sum = -np.inf
    start_time = time.time()
    
    for _ in range(n_samples):
        if time.time() - start_time > time_limit:
            break
        
        # Random initial positions
        sides = np.random.uniform(0.1, 0.3, n)
        centers = np.random.uniform(sides/2, 1 - sides/2, (n, 2))
        
        # Run coordinate ascent
        c, s, s_sum = _coordinate_ascent(centers, sides, time_limit=5)
        
        if s_sum > best_sum:
            best_sum = s_sum
            best_centers = c.copy()
            best_sides = s.copy()
    
    if best_centers is None:
        return _serpentine_packing(n)
    
    return best_centers, best_sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    n = 10
    
    # Start with serpentine packing
    centers, sides = _serpentine_packing(n)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_sum = np.sum(sides)
    
    # Run coordinate ascent from serpentine
    c, s, s_sum = _coordinate_ascent(centers, sides, max_iter=1000, time_limit=120)
    if s_sum > best_sum:
        best_sum = s_sum
        best_centers = c.copy()
        best_sides = s.copy()
    
    # Try random restarts if time permits
    if time.time() < deadline - 100:
        c, s = _random_packing(n, n_samples=20, time_limit=100)
        c, s, s_sum = _coordinate_ascent(c, s, max_iter=500, time_limit=50)
        if s_sum > best_sum:
            best_sum = s_sum
            best_centers = c.copy()
            best_sides = s.copy()
    
    # Final verification and cleanup
    tol = 1e-7
    for i in range(n):
        cx, cy = best_centers[i]
        s = best_sides[i]
        # Clamp to ensure feasibility
        best_centers[i, 0] = np.clip(cx, s/2, 1 - s/2)
        best_centers[i, 1] = np.clip(cy, s/2, 1 - s/2)
    
    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
