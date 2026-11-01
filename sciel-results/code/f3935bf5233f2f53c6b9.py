import numpy as np
import time
from scipy.spatial import ConvexHull

N = 14

def compute_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    # Ensure points are in a valid convex position? 
    # The problem says "Place 14 points in a planar convex region". 
    # It does not require the points themselves to be in convex position (i.e., on the hull).
    # However, if a point is inside the hull, it might form smaller triangles.
    # The score is normalized by the convex hull area of the *point set*.
    
    # Calculate Convex Hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, .volume is area
    except Exception:
        # If all points are collinear or identical, area is 0
        return 0.0

    if hull_area == 0:
        return 0.0

    # Calculate all triangle areas
    # N=14, C(14,3) = 364 triangles. This is fast.
    min_area = float('inf')
    n = len(points)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |det(p2-p1, p3-p1)|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area

    return min_area / hull_area

def construct():
    # Initial guess: 3x5 grid minus one point
    # Grid: x in 0..4, y in 0..2
    x_vals = np.linspace(0, 4, 5)
    y_vals = np.linspace(0, 2, 3)
    grid_points = np.array([[x, y] for x in x_vals for y in y_vals])
    
    # Remove one point to get 14. Let's remove a corner to keep symmetry? 
    # Or remove a center point? 
    # Removing a corner (0,0) leaves the hull as the rest of the grid.
    # Let's remove (0,0).
    points = grid_points[1:].copy()
    
    # Normalize to unit square initially? 
    # The score is scale invariant (area/area), so normalization doesn't change the score.
    # But keeping coordinates bounded helps numerical stability.
    # Current bounds: x in [0,4], y in [0,2].
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    current_points = points.copy()
    current_score = compute_score(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 0.01 # Initial temperature
    T_min = 1e-5
    alpha = 0.999 # Cooling rate
    step_size = 0.1
    
    while time.time() - start_time < timeout:
        # Generate a neighbor
        new_points = current_points.copy()
        idx = np.random.randint(0, N)
        # Gaussian perturbation
        new_points[idx] += np.random.normal(0, step_size, 2)
        
        # Optional: Keep points within a reasonable bounding box?
        # The problem says "planar convex region normalized to unit area".
        # It doesn't strictly constrain the points to be inside the unit square,
        # but the hull area is used for normalization.
        # However, if points go far apart, the hull area grows, but so do triangle areas.
        # Scale invariance means we can fix the hull area to 1 or just let it vary.
        # Let's allow free movement but maybe constrain to a box to prevent drift.
        # Actually, if we just maximize the ratio, the system will find the optimal shape.
        
        new_score = compute_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            if np.random.rand() < np.exp((new_score - current_score) / T):
                current_points = new_points
                current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # If stuck, maybe restart or increase step size?
            # For now, just keep going.
            
    return best_points
