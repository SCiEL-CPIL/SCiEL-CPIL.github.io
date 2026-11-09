# EVOLVE-BLOCK-START
import numpy as np
import time

def _check_validity(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Check bounds
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                return False
    return True

def _objective(centers, sides):
    """Return sum of sides if valid, else -infinity."""
    if not _check_validity(centers, sides):
        return -np.inf
    return np.sum(sides)

def _can_place(centers, sides, i, s_try, cx_try, cy_try, tol=1e-7):
    """Check if square i with side s_try at (cx_try, cy_try) is valid."""
    n = len(sides)
    # Bounds
    if cx_try - s_try/2 < -tol or cx_try + s_try/2 > 1 + tol:
        return False
    if cy_try - s_try/2 < -tol or cy_try + s_try/2 > 1 + tol:
        return False
    # Overlap with others
    for j in range(n):
        if j == i:
            continue
        t = (s_try + sides[j])/2 - tol
        if abs(cx_try - centers[j,0]) < t and abs(cy_try - centers[j,1]) < t:
            return False
    return True

def _max_side_for_square(centers, sides, i, max_s=1.0, tol=1e-7):
    """
    Find the maximum side length for square i that can be placed without
    overlapping others and within bounds. Uses bisection on side length,
    and for each side length, searches for a valid position.
    Returns (best_s, best_cx, best_cy).
    """
    n = len(sides)
    best_s = sides[i]
    best_cx = centers[i, 0]
    best_cy = centers[i, 1]
    
    # Bisection on side length
    lo = sides[i]
    hi = min(max_s, 1.0)
    
    # First check if current side is valid at current position
    if not _can_place(centers, sides, i, sides[i], centers[i,0], centers[i,1], tol):
        return sides[i], centers[i,0], centers[i,1]
    
    for _ in range(30):  # ~30 bisection steps for precision
        mid = (lo + hi) / 2
        if mid <= sides[i]:
            break
        
        # Try to find a valid position for square i with side mid
        found = False
        # Search grid around current position and boundaries
        cx_range = np.linspace(mid/2, 1 - mid/2, 15)
        cy_range = np.linspace(mid/2, 1 - mid/2, 15)
        
        # Start near current position, then expand
        search_order = []
        cur_cx = centers[i, 0]
        cur_cy = centers[i, 1]
        # Add current position first
        search_order.append((cur_cx, cur_cy))
        # Add nearby points
        for dcx in [-0.05, -0.02, 0.02, 0.05, -0.1, 0.1]:
            for dcy in [-0.05, -0.02, 0.02, 0.05, -0.1, 0.1]:
                nx = cur_cx + dcx
                ny = cur_cy + dcy
                if mid/2 <= nx <= 1 - mid/2 and mid/2 <= ny <= 1 - mid/2:
                    search_order.append((nx, ny))
        # Add grid points
        for gx in cx_range:
            for gy in cy_range:
                search_order.append((gx, gy))
        
        for sx, sy in search_order:
            if _can_place(centers, sides, i, mid, sx, sy, tol):
                found = True
                best_s = mid
                best_cx = sx
                best_cy = sy
                break
        
        if found:
            lo = mid
        else:
            hi = mid
    
    return best_s, best_cx, best_cy

def _coordinate_ascent(centers, sides, max_iter=5000, tol=1e-10):
    """
    Coordinate ascent: for each square, try to grow it by finding the maximum
    side length that keeps it non-overlapping with all others and within bounds.
    """
    n = len(sides)
    centers = centers.copy()
    sides = sides.copy()
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            best_s, best_cx, best_cy = _max_side_for_square(centers, sides, i)
            
            if best_s > sides[i] + tol:
                sides[i] = best_s
                centers[i, 0] = best_cx
                centers[i, 1] = best_cy
                improved = True
        
        if not improved:
            break
    
    return centers, sides

def _local_search(centers, sides, time_limit=30):
    """
    Local search with random perturbations and coordinate ascent refinement.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _objective(best_centers, best_sides)
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        # Generate a perturbation
        centers_p = best_centers.copy()
        sides_p = best_sides.copy()
        
        # Perturb centers
        for i in range(n):
            centers_p[i, 0] += rng.normal(0, 0.02)
            centers_p[i, 1] += rng.normal(0, 0.02)
            # Perturb sides
            sides_p[i] *= (1 + rng.normal(0, 0.01))
        
        # Clip to valid range
        for i in range(n):
            sides_p[i] = max(sides_p[i], 0.01)
            centers_p[i, 0] = np.clip(centers_p[i, 0], sides_p[i]/2, 1 - sides_p[i]/2)
            centers_p[i, 1] = np.clip(centers_p[i, 1], sides_p[i]/2, 1 - sides_p[i]/2)
        
        score = _objective(centers_p, sides_p)
        if score > best_score + 1e-10:
            best_centers = centers_p
            best_sides = sides_p
            best_score = score
            # Try coordinate ascent from here
            ca_centers, ca_sides = _coordinate_ascent(centers_p, sides_p, max_iter=200)
            ca_score = _objective(ca_centers, ca_sides)
            if ca_score > best_score + 1e-10:
                best_centers = ca_centers
                best_sides = ca_sides
                best_score = ca_score
    
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
    n = 10
    deadline = time.time() + 580  # Leave 20s safety margin
    
    # Start with a good analytical seed: 2 large squares + 8 small squares
    # One known good configuration:
    # Two squares of side ~0.5 in the bottom row, and 8 smaller squares above
    # Let's use a better seed:
    
    # Seed 1: Grid (baseline)
    k = int(np.ceil(np.sqrt(n)))
    side_grid = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_grid = (cols + 0.5) * side_grid
    cy_grid = (rows + 0.5) * side_grid
    centers_grid = np.stack([cx_grid, cy_grid], axis=1).astype(np.float64)
    sides_grid = np.full(n, side_grid, dtype=np.float64)
    
    # Seed 2: 2+8 configuration
    # Two large squares at bottom, 8 small ones above
    # Large squares: side ~0.45, positioned at bottom
    # Small squares: side ~0.27, in a 4x2 grid above
    centers_28 = np.zeros((n, 2))
    sides_28 = np.zeros(n)
    
    # Two large squares at bottom
    s_large = 0.45
    centers_28[0] = [s_large/2, s_large/2]
    centers_28[1] = [1 - s_large/2, s_large/2]
    sides_28[0] = s_large
    sides_28[1] = s_large
    
    # 8 small squares in a 4x2 grid above
    s_small = 0.27
    y_base = s_large + s_small/2
    for i in range(8):
        col = i % 4
        row = i // 4
        centers_28[2+i, 0] = (col + 0.5) * (1.0/4)
        centers_28[2+i, 1] = y_base + row * s_small
        sides_28[2+i] = s_small
    
    # Check validity of seed 2
    if not _check_validity(centers_28, sides_28):
        # Adjust to make it valid
        # Reduce small squares
        for s_small_try in np.linspace(0.25, 0.15, 20):
            centers_28_test = centers_28.copy()
            sides_28_test = sides_28.copy()
            y_base = s_large + s_small_try/2
            for i in range(8):
                col = i % 4
                row = i // 4
                centers_28_test[2+i, 0] = (col + 0.5) * (1.0/4)
                centers_28_test[2+i, 1] = y_base + row * s_small_try
                sides_28_test[2+i] = s_small_try
            if _check_validity(centers_28_test, sides_28_test):
                centers_28 = centers_28_test
                sides_28 = sides_28_test
                break
    
    # Choose the better seed
    score_grid = _objective(centers_grid, sides_grid)
    score_28 = _objective(centers_28, sides_28)
    
    if score_28 > score_grid:
        centers = centers_28.copy()
        sides = sides_28.copy()
    else:
        centers = centers_grid.copy()
        sides = sides_grid.copy()
    
    # Apply coordinate ascent
    if time.time() < deadline - 10:
        centers, sides = _coordinate_ascent(centers, sides, max_iter=500)
    
    # Apply local search
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        centers, sides = _local_search(centers, sides, time_limit=min(remaining_time, 120))
    
    # Final coordinate ascent with more iterations for polish
    if time.time() < deadline - 2:
        centers, sides = _coordinate_ascent(centers, sides, max_iter=1000)
    
    # Final high-precision polish: try to squeeze out more by growing each square
    if time.time() < deadline - 1:
        for _ in range(3):
            improved = False
            for i in range(n):
                best_s, best_cx, best_cy = _max_side_for_square(centers, sides, i)
                if best_s > sides[i] + 1e-12:
                    sides[i] = best_s
                    centers[i, 0] = best_cx
                    centers[i, 1] = best_cy
                    improved = True
            if not improved:
                break
    
    # Ensure validity
    if not _check_validity(centers, sides):
        # Fall back to grid
        return centers_grid, sides_grid
    
    return centers, sides

# EVOLVE-BLOCK-END
