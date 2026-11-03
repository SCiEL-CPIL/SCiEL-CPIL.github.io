"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area of triangle formed by pi, pj, pk
                area = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute score: min triangle area / convex hull area."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _halton_sequence(index, base):
    """Generate the index-th number in the Halton sequence for the given base."""
    f = 1.0
    result = 0.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result

def _generate_halton(n, seed=1):
    """Generate n points using 2D Halton sequence, scaled to [0,1]^2."""
    points = []
    for i in range(1, n + 1):
        x = _halton_sequence(i, 2)
        y = _halton_sequence(i, 3)
        points.append([x, y])
    return np.array(points)

def _generate_grid_initialization(n):
    """Generate points on a grid-like pattern for n points."""
    # For n=13, use a 4x4 grid and pick 13 points
    # Or use a hexagonal packing pattern
    # Let's use a circular arrangement with some inner points
    points = []
    # Outer ring: 8 points
    for i in range(8):
        angle = 2 * np.pi * i / 8
        r = 0.5
        points.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    # Inner ring: 4 points
    for i in range(4):
        angle = 2 * np.pi * i / 4 + np.pi / 4
        r = 0.25
        points.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    # Center: 1 point
    points.append([0.5, 0.5])
    return np.array(points)

def _local_search(points, max_iter=10000, step_size=0.01, rng=None):
    """Perform local search to improve the minimum triangle area."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    for iteration in range(max_iter):
        # Pick a random point to move
        idx = rng.integers(0, N)
        old_pos = current_points[idx].copy()
        
        # Generate a new position
        new_pos = old_pos + rng.normal(0, step_size, size=2)
        new_pos = np.clip(new_pos, 0.01, 0.99)
        
        # Try the move
        current_points[idx] = new_pos
        new_score = _score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            # Revert
            current_points[idx] = old_pos
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate multiple initializations
    initializations = []
    
    # Halton sequence
    halton_points = _generate_halton(N)
    initializations.append(halton_points)
    
    # Grid-like initialization
    grid_points = _generate_grid_initialization(N)
    initializations.append(grid_points)
    
    # Random with fixed seed
    rng = np.random.default_rng(42)
    random_points = rng.random((N, 2))
    initializations.append(random_points)
    
    # More Halton with different offsets
    for offset in range(2, 6):
        halton_offset = _generate_halton(N, seed=offset)
        initializations.append(halton_offset)
    
    best_points = None
    best_score = -float('inf')
    
    for init_points in initializations:
        if time.time() > deadline:
            break
        
        # Run local search from this initialization
        searched_points, score = _local_search(
            init_points, 
            max_iter=5000, 
            step_size=0.01,
            rng=np.random.default_rng(hash(tuple(init_points[0])) % (2**32))
        )
        
        if score > best_score:
            best_score = score
            best_points = searched_points
    
    # If no valid result, return the first initialization
    if best_points is None:
        best_points = initializations[0]
    
    return best_points
# EVOLVE-BLOCK-END
