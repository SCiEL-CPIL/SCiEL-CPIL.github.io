import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    
    def triangle_area(p1, p2, p3):
        return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0

    def min_triangle_area(points):
        n = len(points)
        min_area = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a = triangle_area(points[i], points[j], points[k])
                    if a < min_area:
                        min_area = a
        return min_area

    def hull_area(points):
        try:
            hull = ConvexHull(points)
            return hull.volume
        except:
            return 0.0

    def score(points):
        a_min = min_triangle_area(points)
        a_hull = hull_area(points)
        if a_hull < 1e-9 or a_min < 1e-9:
            return 0.0
        return a_min / a_hull

    def objective(x):
        points = x.reshape(13, 2)
        # Check for validity
        if not np.all(np.isfinite(points)):
            return 1e6
        a_hull = hull_area(points)
        if a_hull < 1e-6:
            return 1e6
        a_min = min_triangle_area(points)
        if a_min < 1e-6:
            return 1e6
        return -a_min / a_hull

    # Initial guess: Hexagonal cluster
    # Center
    # Inner ring (6 points)
    # Outer ring (6 points)
    
    r_inner = 1.0
    r_outer = 2.0
    
    points = np.zeros((13, 2))
    points[0] = [0, 0]
    
    # Inner ring
    for i in range(6):
        angle = i * np.pi / 3
        points[i + 1] = [r_inner * np.cos(angle), r_inner * np.sin(angle)]
        
    # Outer ring
    # The outer points in the hex lattice are at angles 0, 60, 120, 180, 240, 300
    # but shifted? No, they are aligned with the inner ones in this specific subset.
    # In the lattice derivation:
    # Inner: (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5, -0.866), (0.5, -0.866)
    # Outer: (2,0), (1, 1.732), (-1, 1.732), (-2,0), (-1, -1.732), (1, -1.732)
    # Note: Outer points are NOT at the same angles as inner points in a simple radial sense?
    # Inner angles: 0, 60, 120, 180, 240, 300.
    # Outer angles: 0, 60, 120, 180, 240, 300.
    # Yes, they are aligned radially.
    
    for i in range(6):
        angle = i * np.pi / 3
        points[i + 7] = [r_outer * np.cos(angle), r_outer * np.sin(angle)]

    # Optimization
    start_time = time.time()
    deadline = start_time + 600  # 600s limit
    
    best_points = points.copy()
    best_score = score(best_points)
    
    x0 = points.flatten()
    
    # Use Powell's method for robustness
    # We can run multiple restarts if time permits
    restarts = 0
    max_restarts = 10
    
    while time.time() < deadline - 10:
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'maxfev': 5000})
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(13, 2)
            # Perturb for next restart if time allows
            # Simple perturbation
            if time.time() < deadline - 30:
                x0 = best_points.flatten() + np.random.normal(0, 0.1, 26)
                restarts += 1
                if restarts > max_restarts:
                    break
            else:
                break
        except Exception:
            break

    return best_points
