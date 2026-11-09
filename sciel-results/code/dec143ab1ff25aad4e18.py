# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 2^(k-2) + 1 points in general
    position (no three collinear) inside the unit square so as to MINIMIZE the number of
    convex k-gons whose vertices are among the points. Here k = 7, so n = 2^5 + 1 = 33.
    
    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates of the points,
            each lying in the unit square [0, 1] x [0, 1].
    """
    n = 33
    
    # 1. Initialization on a sparse grid
    # Create a 10x10 grid in [0.05, 0.95] to stay safely inside
    grid_size = 10
    margin = 0.05
    x_coords = np.linspace(margin, 1 - margin, grid_size)
    y_coords = np.linspace(margin, 1 - margin, grid_size)
    grid_points = np.array([[x, y] for x in x_coords for y in y_coords])
    
    # Randomly select 33 points from the 100 grid points
    np.random.seed(42)
    selected_indices = np.random.choice(len(grid_points), n, replace=False)
    points = grid_points[selected_indices].copy()
    
    # Add small jitter to break collinearity and ensure general position
    # The grid points are collinear in rows/cols, so we MUST jitter.
    points += np.random.normal(0, 0.01, (n, 2))
    
    # Clamp to bounds
    points = np.clip(points, 1e-5, 1 - 1e-5)
    
    # Ensure validity (jitter might cause close points or collinearity issues, though unlikely with 0.01 sigma)
    # We will rely on the hill climbing to fix if invalid, but better to start valid.
    if not _is_valid(points):
        # Fallback to random if grid init fails validity (rare)
        points = np.random.rand(n, 2)
        points = np.clip(points, 1e-5, 1 - 1e-5)

    best_points = points.copy()
    best_count = _count_convex_7gons(points)
    
    # Pre-compute combinations for speed
    combos = list(itertools.combinations(range(n), 7))
    
    # 2. Stochastic Hill Climbing
    start_time = time.time()
    max_time = 580  # Leave 20s buffer
    
    current_points = points.copy()
    current_count = best_count
    
    iteration = 0
    while time.time() - start_time < max_time:
        iteration += 1
        
        # Pick a random point to move
        idx = np.random.randint(n)
        
        # Propose a move
        # 50% chance: small Gaussian perturbation
        # 50% chance: jump to a nearby grid point or random point
        if np.random.rand() < 0.5:
            # Small perturbation
            new_point = current_points[idx] + np.random.normal(0, 0.02, 2)
        else:
            # Jump to a random point in the square (or grid)
            # Using grid points as targets helps "pack" density
            if np.random.rand() < 0.5:
                target_idx = np.random.randint(len(grid_points))
                new_point = grid_points[target_idx] + np.random.normal(0, 0.005, 2)
            else:
                new_point = np.random.rand(2)
        
        # Apply move
        candidate = current_points.copy()
        candidate[idx] = new_point
        
        # Check bounds
        if np.any(candidate < -1e-6) or np.any(candidate > 1 + 1e-6):
            continue
            
        # Check validity (collinearity, distance)
        if not _is_valid(candidate):
            continue
            
        # Count convex 7-gons
        new_count = _count_convex_7gons_from_combos(candidate, combos)
        
        # Acceptance criterion
        if new_count < current_count:
            current_points = candidate
            current_count = new_count
            if new_count < best_count:
                best_count = new_count
                best_points = candidate.copy()
        elif new_count == current_count:
            # Accept with some probability to escape local minima
            if np.random.rand() < 0.5:
                current_points = candidate
                # current_count stays same
        else:
            # Reject
            pass
            
    return best_points


def _is_valid(points):
    """Check if points are in bounds, finite, no close pairs, no collinear triples."""
    n = len(points)
    # Bounds and Finite
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    if not np.all(np.isfinite(points)):
        return False
        
    # Distance check
    for i in range(n):
        for j in range(i + 1, n):
            d2 = np.sum((points[i] - points[j]) ** 2)
            if d2 < 1e-12:
                return False
                
    # Collinearity check
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Determinant of (p2-p1, p3-p1)
                det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if abs(det) < 1e-9:
                    return False
    return True


def _is_convex_7(pts):
    """Check if 7 points form a convex polygon."""
    # Compute centroid
    c = np.mean(pts, axis=0)
    # Compute angles
    dx = pts[:, 0] - c[0]
    dy = pts[:, 1] - c[1]
    angles = np.arctan2(dy, dx)
    # Sort by angle
    order = np.argsort(angles)
    sorted_pts = pts[order]
    
    # Check turns
    # For a convex polygon sorted by angle around centroid, all cross products of consecutive edges must have the same sign.
    # We assume CCW, so cross > 0.
    for i in range(7):
        p1 = sorted_pts[i]
        p2 = sorted_pts[(i + 1) % 7]
        p3 = sorted_pts[(i + 2) % 7]
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < 1e-9:
            return False
    return True


def _count_convex_7gons(points):
    """Count convex 7-gons. Used for initial count."""
    combos = list(itertools.combinations(range(len(points)), 7))
    return _count_convex_7gons_from_combos(points, combos)


def _count_convex_7gons_from_combos(points, combos):
    """Count convex 7-gons using pre-computed combinations."""
    count = 0
    for combo in combos:
        pts = points[list(combo)]
        if _is_convex_7(pts):
            count += 1
    return count

# EVOLVE-BLOCK-END
