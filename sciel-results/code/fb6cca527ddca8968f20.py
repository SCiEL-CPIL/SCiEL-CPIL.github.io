import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

N = 13

# EVOLVE-BLOCK-START

def _calc_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns a large positive value for invalid/degenerate inputs to be minimized.
    """
    points = np.asarray(points)
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 1e6 # Degenerate hull, bad score
    except Exception:
        return 1e6

    # 2. Calculate Minimum Triangle Area
    # There are C(13, 3) = 286 triangles
    # Vectorized calculation
    idx = np.array(list(itertools.combinations(range(N), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-9:
        return 1e6 # Degenerate triangle, bad score
        
    return min_area / hull_area

def _objective(x):
    points = x.reshape(N, 2)
    score = _calc_score(points)
    return -score # We want to maximize score, so minimize -score

def construct():
    start_time = time.time()
    # Safety margin: stop 10s before the 150s limit
    deadline = start_time + 140.0
    
    best_points = None
    best_score = -np.inf
    
    def try_optimize(init_points, name="unknown"):
        nonlocal best_points, best_score
        if time.time() > deadline:
            return
            
        # Normalize initial points to have a reasonable scale (hull area ~ 1)
        try:
            hull = ConvexHull(init_points)
            area = hull.volume
            if area < 1e-6:
                return
            scale = np.sqrt(area)
            init_points = init_points / scale
        except Exception:
            # If hull fails, try to fix by adding small noise
            init_points = init_points + np.random.normal(0, 0.01, init_points.shape)
            try:
                hull = ConvexHull(init_points)
                area = hull.volume
                if area < 1e-6:
                    return
                scale = np.sqrt(area)
                init_points = init_points / scale
            except Exception:
                return

        try:
            # Use L-BFGS-B for faster convergence
            res = minimize(_objective, init_points.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < -best_score:
                new_points = res.x.reshape(N, 2)
                # Verify the score with the actual calculation
                new_score = -res.fun
                # Double check with direct calculation to avoid optimizer artifacts
                direct_score = _calc_score(new_points)
                if direct_score < 1e5: # Valid score
                    new_score = direct_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points
        except Exception:
            pass

    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    try_optimize(p1, "13-gon")
    
    # 2. Regular 12-gon + Center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    p2 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    p2 = np.vstack([p2, [[0,0]]])
    try_optimize(p2, "12-gon+center")
    
    # 3. Two concentric hexagons + center
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer = np.column_stack([np.cos(angles6), np.sin(angles6)])
    inner = 0.5 * np.column_stack([np.cos(angles6), np.sin(angles6)])
    p3 = np.vstack([outer, inner, [[0,0]]])
    try_optimize(p3, "2xHex+center")
    
    # 4. Grid based: 4x4 grid, remove 3 points
    # Create a 4x4 grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    grid = np.array([[xi, yi] for xi in x for yi in y])
    # Remove 3 corners to get 13 points
    # Keep (0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3), (2,0), (2,1), (2,2), (2,3), (3,1)
    # Indices to remove: (3,0), (3,2), (3,3) -> indices 12, 14, 15 in row-major?
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # Row 3: 12,13,14,15
    # Remove 12, 14, 15
    keep_indices = [i for i in range(16) if i not in [12, 14, 15]]
    p4 = grid[keep_indices]
    try_optimize(p4, "Grid4x4-3")
    
    # 5. Random initializations
    rng = np.random.RandomState(123)
    for i in range(5):
        p5 = rng.rand(N, 2) * 2 - 1
        try_optimize(p5, f"rand{i}")

    # If no valid points found, return a default regular polygon
    if best_points is None:
        return p1
        
    return best_points

# EVOLVE-BLOCK-END
