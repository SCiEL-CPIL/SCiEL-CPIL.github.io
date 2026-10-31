"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _triangle_area(p, q, r):
    """Compute area of triangle formed by points p, q, r."""
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(pts[i], pts[j], pts[k])
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, eps=1e-6):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    if y < -eps or y > (np.sqrt(3) / 2) + eps:
        return False
    # Left edge: from (0,0) to (0.5, sqrt(3)/2): y = sqrt(3)*x => x >= y/sqrt(3)
    # Right edge: from (1,0) to (0.5, sqrt(3)/2): y = -sqrt(3)*(x-1) => x <= 1 - y/sqrt(3)
    if x < -eps or x > 1 + eps:
        return False
    # More precise: check barycentric coordinates
    # For triangle (0,0), (1,0), (0.5, sqrt(3)/2):
    # Point (x,y) is inside if:
    # y >= 0
    # y <= sqrt(3)*x  (below the line from (0,0) to (0.5, sqrt(3)/2)... wait, that's the left edge)
    # Actually: left edge from (0,0) to (0.5, sqrt(3)/2): parametric: (0.5t, sqrt(3)/2 * t), so x = 0.5t, y = sqrt(3)/2 * t => y = sqrt(3)*x
    # Right edge from (1,0) to (0.5, sqrt(3)/2): parametric: (1-0.5t, sqrt(3)/2 * t), so x = 1-0.5t, y = sqrt(3)/2 * t => y = sqrt(3)*(1-x)
    # Bottom edge: y = 0
    # So inside means: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    if y < -eps:
        return False
    if y > np.sqrt(3) * x + eps:
        return False
    if y > np.sqrt(3) * (1 - x) + eps:
        return False
    return True

def _objective(x):
    """Objective function to minimize: negative of min triangle area."""
    pts = x.reshape(N, 2)
    # Check all points are inside the triangle
    for i in range(N):
        if not _inside_triangle(pts[i], eps=-1e-4):
            # Return a large penalty if any point is outside
            return 1e6
    min_area = _min_triangle_area(pts)
    return -min_area

def _project_to_triangle(p):
    """Project point p to the nearest point inside the equilateral triangle."""
    x, y = p
    # Clamp to the triangle
    # The triangle is defined by:
    # y >= 0
    # y <= sqrt(3)*x
    # y <= sqrt(3)*(1-x)
    
    # We'll use a simple iterative projection
    for _ in range(20):
        # Check constraints
        # Constraint 1: y >= 0
        if y < 0:
            y = 0
        # Constraint 2: y <= sqrt(3)*x => x >= y/sqrt(3)
        if x < y / np.sqrt(3):
            x = y / np.sqrt(3)
        # Constraint 3: y <= sqrt(3)*(1-x) => x <= 1 - y/sqrt(3)
        if x > 1 - y / np.sqrt(3):
            x = 1 - y / np.sqrt(3)
        # Also ensure x in [0, 1]
        if x < 0:
            x = 0
        if x > 1:
            x = 1
    return np.array([x, y])

def _random_inside_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    if u + v < 1:
        return np.array([u + 0.5 * v, np.sqrt(3) / 2 * v])
    else:
        # Retry
        return _random_inside_triangle(rng)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_area = -float('inf')
    
    # Use multiple random restarts with local optimization
    rng = np.random.RandomState(42)
    
    n_restarts = 0
    while time.time() < deadline - 5.0:  # Leave 5s for final evaluation
        n_restarts += 1
        
        # Generate random initial points
        pts = np.array([_random_inside_triangle(rng) for _ in range(N)])
        
        # Run local optimization
        try:
            result = minimize(
                _objective,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Project result to triangle
            result_pts = result.x.reshape(N, 2)
            for i in range(N):
                result_pts[i] = _project_to_triangle(result_pts[i])
            
            # Check if all points are inside
            valid = all(_inside_triangle(result_pts[i]) for i in range(N))
            
            if valid:
                area = _min_triangle_area(result_pts)
                if area > best_area:
                    best_area = area
                    best_pts = result_pts.copy()
                    
        except Exception as e:
            # If optimization fails, try the initial random points
            area = _min_triangle_area(pts)
            if area > best_area:
                best_area = area
                best_pts = pts.copy()
    
    # If we haven't found any valid points, generate some random ones
    if best_pts is None:
        best_pts = np.array([_random_inside_triangle(rng) for _ in range(N)])
    
    return best_pts
# EVOLVE-BLOCK-END
