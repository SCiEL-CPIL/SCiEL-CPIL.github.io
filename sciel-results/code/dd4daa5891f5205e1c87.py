"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by the points.
    points: (N, 2) array of coordinates.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized calculation of all triangle areas
    # i, j, k are indices
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    
    # To be fast, we can iterate over i, j and vectorize over k?
    # Or just triple loop with numpy ops on slices.
    # For N=11, triple loop is 11*10*9/6 = 165 triangles.
    # Python loop overhead for 165 iterations is negligible.
    
    min_area = float('inf')
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx12 = x2 - x1
            dy12 = y2 - y1
            # Vectorized over k
            # k must be > j
            if j + 1 >= n:
                continue
            x3 = points[j+1:, 0]
            y3 = points[j+1:, 1]
            
            cross = dx12 * (y3 - y1) - dy12 * (x3 - x1)
            areas = 0.5 * np.abs(cross)
            current_min = np.min(areas)
            if current_min < min_area:
                min_area = current_min
    return min_area

def _is_inside_triangle(points):
    """
    Check if points are inside the equilateral triangle with vertices
    (0,0), (1,0), (0.5, sqrt(3)/2).
    Returns True if all points are inside (with small tolerance).
    """
    tol = 1e-9
    # y >= 0
    if np.any(points[:, 1] < -tol):
        return False
    # y <= sqrt(3) * x  => y - sqrt(3)*x <= 0
    if np.any(points[:, 1] - np.sqrt(3) * points[:, 0] > tol):
        return False
    # y <= -sqrt(3) * x + sqrt(3) => y + sqrt(3)*x - sqrt(3) <= 0
    if np.any(points[:, 1] + np.sqrt(3) * points[:, 0] - np.sqrt(3) > tol):
        return False
    return True

def _penalty(points):
    """
    Calculate penalty for points outside the triangle.
    """
    # Distance to boundaries
    # 1. y < 0
    p1 = np.sum(np.maximum(0, -points[:, 1]))
    # 2. y > sqrt(3) x
    p2 = np.sum(np.maximum(0, points[:, 1] - np.sqrt(3) * points[:, 0]))
    # 3. y > -sqrt(3) x + sqrt(3)
    p3 = np.sum(np.maximum(0, points[:, 1] + np.sqrt(3) * points[:, 0] - np.sqrt(3)))
    return p1 + p2 + p3

def _objective(x):
    """
    Objective function for minimization.
    x: flat array of 2*N coordinates.
    """
    points = x.reshape(N, 2)
    
    # Check validity
    if not _is_inside_triangle(points):
        # Return a large penalty
        # We want to minimize, so large positive value
        return 100.0 + _penalty(points)
    
    area = _min_triangle_area(points)
    # We want to maximize area, so minimize -area
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Bounds for differential_evolution
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1) for _ in range(N)] + [(0, np.sqrt(3)/2) for _ in range(N)]
    # Interleave x and y? No, reshape assumes [x1, y1, x2, y2, ...] or [x1..xN, y1..yN]?
    # My reshape is x.reshape(N, 2), so x = [x1, y1, x2, y2, ...]
    # So bounds should be [x1, y1, x2, y2, ...]
    bounds = []
    for _ in range(N):
        bounds.append((0, 1))
        bounds.append((0, np.sqrt(3)/2))
        
    best_result = None
    best_score = -float('inf')
    
    # Run differential_evolution
    # popsize=20 is reasonable for 22 dims
    # maxiter=200 might take a while, let's monitor time
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-8, 
            seed=42,
            polish=True,
            init='sobol' # Sobol sequence is better for uniform coverage
        )
        
        if result.success or result.fun < 100: # Check if it found a valid solution
            if _is_inside_triangle(result.x.reshape(N, 2)):
                area = _min_triangle_area(result.x.reshape(N, 2))
                if area > best_score:
                    best_score = area
                    best_result = result.x.reshape(N, 2)
                    
    except Exception as e:
        pass
        
    # If time remains, try to polish the best result found so far
    if best_result is not None and time.time() < deadline:
        try:
            # Use Nelder-Mead to refine
            res = minimize(
                _objective, 
                best_result.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < 100: # Valid
                if _is_inside_triangle(res.x.reshape(N, 2)):
                    area = _min_triangle_area(res.x.reshape(N, 2))
                    if area > best_score:
                        best_score = area
                        best_result = res.x.reshape(N, 2)
        except Exception:
            pass
            
    # Fallback if no valid result found (should not happen with penalty)
    if best_result is None:
        # Return a simple grid or random points
        np.random.seed(42)
        points = np.random.rand(N, 2)
        points[:, 0] *= 1.0
        points[:, 1] *= np.sqrt(3)/2
        # Ensure inside
        for i in range(N):
            while not _is_inside_triangle(points[i:i+1]):
                points[i] = np.random.rand(2) * np.array([1.0, np.sqrt(3)/2])
        best_result = points
        best_score = _min_triangle_area(best_result)
        
    return best_result
# EVOLVE-BLOCK-END
