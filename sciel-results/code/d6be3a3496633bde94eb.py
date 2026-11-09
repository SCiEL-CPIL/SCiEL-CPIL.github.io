import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    # points: (N, 2)
    # Returns min area of any triangle formed by 3 points
    min_area = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    # points: (N, 2)
    # Assumes points are in convex position (e.g., on a circle in angular order)
    # If not, we should sort them by angle first if they are on a circle.
    # For general points, we'd use ConvexHull, but for circle points, sorting by angle is sufficient.
    # Let's assume the input to the objective function are angles, and we generate sorted points.
    pass

def construct():
    # Initial guess: regular polygon
    x0 = 2 * np.pi * np.arange(N) / N
    
    # Bounds for angles: [0, 2*pi]
    bounds = [(0, 2 * np.pi) for _ in range(N)]
    
    def objective(angles):
        # Sort angles to ensure convex ordering for area calculation
        # Note: The set of points is the same regardless of order, but 
        # for calculating the hull area as a simple polygon, we need order.
        # The min triangle area is independent of order.
        angles_sorted = np.sort(angles)
        
        # Generate points
        points = np.column_stack((np.cos(angles_sorted), np.sin(angles_sorted)))
        
        # Calculate min triangle area
        min_area = _min_triangle_area(points)
        
        # Calculate hull area
        # Since points are on a circle and sorted by angle, the convex hull is the polygon formed by these points.
        # Area of polygon = 0.5 * sum(x_i y_{i+1} - x_{i+1} y_i)
        x = points[:, 0]
        y = points[:, 1]
        # Wrap around
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        hull_area = 0.5 * np.sum(x * y_next - x_next * y)
        
        if hull_area == 0:
            return 0.0
            
        score = min_area / hull_area
        # We want to maximize score, so minimize negative score
        return -score

    # Run optimization
    # popsize and maxiter need to be tuned for 120s limit.
    # N=14, 364 triangles per eval. 
    # Let's try popsize=10, maxiter=20. 
    # 10 * 20 * 14 * (cost of eval)
    # Cost of eval is dominated by the triple loop (364 iterations).
    # This should be fast enough.
    
    try:
        res = differential_evolution(objective, bounds, seed=42, maxiter=30, popsize=10, tol=0.01)
        best_angles = res.x
        # Sort best angles for final construction
        best_angles_sorted = np.sort(best_angles)
        points = np.column_stack((np.cos(best_angles_sorted), np.sin(best_angles_sorted)))
        return points
    except Exception:
        # Fallback to regular polygon if optimization fails
        angles = 2 * np.pi * np.arange(N) / N
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        return points
