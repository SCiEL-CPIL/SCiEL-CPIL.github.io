# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for validity: finite, non-collinear
    if not np.all(np.isfinite(points)):
        return 0.0
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 1e-12:
        return 0.0
        
    # Vectorized area calculation for all triangles
    # points shape (n, 2)
    # We need to compute area for all combinations of 3 points.
    # Number of triangles is n*(n-1)*(n-2)/6. For n=13, this is 286.
    
    # Generate indices for all combinations of 3 points
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    # Vectorized:
    # x1, y1 = p1[:, 0], p1[:, 1]
    # x2, y2 = p2[:, 0], p2[:, 1]
    # x3, y3 = p3[:, 0], p3[:, 1]
    # term1 = x1 * (y2 - y3)
    # term2 = x2 * (y3 - y1)
    # term3 = x3 * (y1 - y2)
    # areas = 0.5 * np.abs(term1 + term2 + term3)
    
    # Alternative vectorized approach using cross product magnitude
    # v21 = p2 - p1
    # v31 = p3 - p1
    # cross = v21[:, 0] * v31[:, 1] - v21[:, 1] * v31[:, 0]
    # areas = 0.5 * np.abs(cross)
    
    v21 = p2 - p1
    v31 = p3 - p1
    cross = v21[:, 0] * v31[:, 1] - v21[:, 1] * v31[:, 0]
    areas = 0.5 * np.abs(cross)
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Leave 20s margin for safety
    
    best_score = -np.inf
    best_points = None
    
    # Objective function to minimize (negative of score)
    def objective(x):
        points = x.reshape(-1, 2)
        score = compute_score(points)
        if score == 0.0:
            return 1e6 # Penalty for invalid
        return -score
    
    # Generate diverse starting points
    rng = np.random.default_rng(seed=42)
    starts = []
    
    # 1. Regular grid-like starts
    for shift in [0.0, 0.1, 0.2, 0.3]:
        pts = np.zeros((n, 2))
        # Arrange points in a circle
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + shift
        radii = 1.0
        pts[:, 0] = radii * np.cos(angles)
        pts[:, 1] = radii * np.sin(angles)
        starts.append(pts.flatten())
        
    # 2. Concentric circles
    for r_inner in [0.2, 0.4, 0.6]:
        pts = np.zeros((n, 2))
        # 1 center
        pts[0] = [0, 0]
        # 6 inner
        angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False)
        pts[1:7, 0] = r_inner * np.cos(angles_inner)
        pts[1:7, 1] = r_inner * np.sin(angles_inner)
        # 6 outer
        angles_outer = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        pts[7:13, 0] = 1.0 * np.cos(angles_outer)
        pts[7:13, 1] = 1.0 * np.sin(angles_outer)
        starts.append(pts.flatten())
        
    # 3. Random starts
    for _ in range(20):
        pts = rng.random((n, 2)) * 2 - 1 # Range [-1, 1]
        starts.append(pts.flatten())

    # 4. Known good configurations (approximations)
    # Example: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    starts.append(pts.flatten())
    
    # Example: Hexagonal lattice subset
    pts = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2],
        [1.5, np.sqrt(3)/2],
        [1.5, -np.sqrt(3)/2],
        [2, 0],
        [-1.5, np.sqrt(3)/2],
        [-1.5, -np.sqrt(3)/2],
        [-2, 0]
    ])
    starts.append(pts.flatten())

    # Optimize from each start
    for x0 in starts:
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or (not res.success and res.fun < 1e5):
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(-1, 2)
        except Exception:
            continue

    if best_points is None:
        # Fallback to random if optimization failed completely
        return rng.random((n, 2))
        
    return best_points

# EVOLVE-BLOCK-END
