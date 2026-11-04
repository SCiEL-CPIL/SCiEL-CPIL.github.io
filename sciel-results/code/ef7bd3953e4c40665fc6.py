import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11

def _project_to_triangle(points):
    """
    Project points onto the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2
    pts = np.array(points, dtype=float)
    
    for i in range(len(pts)):
        x, y = pts[i]
        
        # Check if inside
        # 1. y >= 0
        # 2. y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
        # 3. y <= -sqrt(3)*(x-1) => -sqrt(3)*x + sqrt(3) - y >= 0  =>  sqrt(3)*(1-x) - y >= 0
        
        if y < 0:
            # Project to y=0 (bottom edge)
            y = 0
        # Check other constraints after first projection
        if y > sqrt3 * x:
            # Project to line y = sqrt(3) x
            # Distance from (x,y) to line sqrt(3)x - y = 0
            # Normal vector (sqrt(3), -1), norm = 2
            # Signed distance d = (sqrt(3)x - y) / 2
            # Since y > sqrt(3)x, d is negative. We want to move towards positive.
            # Actually, let's just solve for intersection or use projection formula.
            # Line: sqrt(3)x - y = 0.
            # Projection of (x,y) onto line ax+by+c=0:
            # x' = x - a*(ax+by+c)/(a^2+b^2)
            # y' = y - b*(ax+by+c)/(a^2+b^2)
            a, b, c = sqrt3, -1, 0
            val = a*x + b*y + c
            x = x - a*val/(a*a+b*b)
            y = y - b*val/(a*a+b*b)
        if y > -sqrt3 * (x - 1):
            # Project to line y = -sqrt(3)x + sqrt(3) => sqrt(3)x + y - sqrt(3) = 0
            a, b, c = sqrt3, 1, -sqrt3
            val = a*x + b*y + c
            x = x - a*val/(a*a+b*b)
            y = y - b*val/(a*a+b*b)
            
        # After projections, check if it drifted outside due to numerical errors or conflicting projections
        # A safer approach for simple convex polygon is to check again and clamp if necessary
        # But for equilateral triangle, single projection to violated edge is usually sufficient if we iterate or if violations are small.
        # Let's do a second pass check for safety.
        if y < 0:
            y = 0
        if y > sqrt3 * x + 1e-12:
            # Re-project
            a, b, c = sqrt3, -1, 0
            val = a*x + b*y + c
            x = x - a*val/(a*a+b*b)
            y = y - b*val/(a*a+b*b)
        if y > -sqrt3 * (x - 1) + 1e-12:
            a, b, c = sqrt3, 1, -sqrt3
            val = a*x + b*y + c
            x = x - a*val/(a*a+b*b)
            y = y - b*val/(a*a+b*b)
            
        pts[i] = [x, y]
    return pts

def _min_area(points):
    """
    Calculate the minimum area of triangles formed by the points.
    """
    min_a = float('inf')
    # Precompute cross products? No, N is small (11). C(11,3) = 165.
    for i, j, k in itertools.combinations(range(N), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
        area = 0.5 * abs(cross)
        if area < min_a:
            min_a = area
    return min_a

def _objective(points_flat):
    """
    Objective function for minimization.
    Returns negative of the minimum area.
    """
    points = points_flat.reshape(N, 2)
    # Project to ensure validity (though Nelder-Mead might wander)
    points = _project_to_triangle(points)
    a = _min_area(points)
    return -a

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2
    
    best_points = None
    best_score = -float('inf')
    
    # Generate initial configurations
    inits = []
    
    # 1. Symmetric "Star" configuration
    # Center
    pts = []
    pts.append([0.5, sqrt3/6])
    # 10 points arranged in 5 pairs symmetric around center?
    # Or 2 rings?
    # Let's try a grid-like distribution first
    # 3x3 grid + 2 extra?
    
    # Config 1: Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        while len(pts) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                x = u + 0.5*v
                y = h*v
                pts.append([x,y])
        inits.append(np.array(pts))
        pts = []
        
    # Config 2: Grid
    # Approximate grid in triangle
    # Rows of points
    # Row 0 (bottom): 4 points
    # Row 1: 3 points
    # Row 2: 2 points
    # Row 3: 1 point
    # Total 10. Need 11.
    # Let's try:
    # y=0.1: x=0.1, 0.3, 0.5, 0.7, 0.9 (5 pts) - check if inside
    # y=0.3: x=0.2, 0.4, 0.6, 0.8 (4 pts)
    # y=0.5: x=0.3, 0.5, 0.7 (3 pts)
    # Total 12. Too many.
    
    # Let's use a known good structure for small N.
    # For N=11, a configuration with 1 center and 10 surrounding is common.
    # Angles: 0, 36, 72...
    # Radii: r1 for 5 points, r2 for 5 points?
    
    # Let's just use multiple random starts and one "centroid + ring" start.
    
    # Ring start
    pts = [[0.5, sqrt3/6]]
    # 5 points on inner ring
    r_inner = 0.2
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi/5 # offset
        x = 0.5 + r_inner * np.cos(angle)
        y = sqrt3/6 + r_inner * np.sin(angle)
        # Need to check if inside triangle. Triangle is not a circle.
        # This might push points outside. We'll project them.
        pts.append([x,y])
    # 5 points on outer ring
    r_outer = 0.4
    for i in range(5):
        angle = 2 * np.pi * i / 5
        x = 0.5 + r_outer * np.cos(angle)
        y = sqrt3/6 + r_outer * np.sin(angle)
        pts.append([x,y])
    inits.append(np.array(pts))
    
    # Another random start with different seed
    rng2 = np.random.RandomState(123)
    pts = []
    for _ in range(5):
        while len(pts) < N:
            u = rng2.random()
            v = rng2.random()
            if u + v < 1:
                x = u + 0.5*v
                y = h*v
                pts.append([x,y])
        inits.append(np.array(pts))
        pts = []
        
    # Run Nelder-Mead on each init
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Flatten
        x0 = init.flatten()
        
        # Nelder-Mead options
        options = {
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'disp': False
        }
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', options=options)
            if res.success or res.fun < -best_score:
                # Check validity
                final_pts = _project_to_triangle(res.x.reshape(N, 2))
                score = _min_area(final_pts)
                if score > best_score:
                    best_score = score
                    best_points = final_pts
        except Exception as e:
            pass # Ignore failures
            
        # Check time
        if time.time() > deadline:
            break
            
    # If no points found, fallback to random
    if best_points is None:
        pts = []
        rng = np.random.RandomState(0)
        while len(pts) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                x = u + 0.5*v
                y = h*v
                pts.append([x,y])
        best_points = np.array(pts)
        
    return best_points
