import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    # Initial configuration: symmetric pattern
    # Centroid of triangle
    cx, cy = 0.5, np.sqrt(3)/6
    
    # Create initial points
    points = []
    
    # 1 point at center
    points.append([cx, cy])
    
    # 6 points in a hexagon around center
    r1 = 0.15
    for i in range(6):
        angle = np.pi/3 * i
        x = cx + r1 * np.cos(angle)
        y = cy + r1 * np.sin(angle)
        points.append([x, y])
    
    # 4 more points near vertices/edges
    # Near vertex (0,0)
    points.append([0.1, 0.05])
    # Near vertex (1,0)
    points.append([0.9, 0.05])
    # Near vertex (0.5, sqrt(3)/2)
    points.append([0.5, 0.8])
    # One more, maybe at another edge midpoint
    points.append([0.25, 0.2])
    
    points = np.array(points)
    
    # Optimize
    def min_triangle_area(pts):
        pts = pts.reshape(-1, 2)
        min_area = np.inf
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = pts[i]
                    b = pts[j]
                    c = pts[k]
                    # Area of triangle
                    area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                    if area < min_area:
                        min_area = area
        return min_area
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Check if all points are inside the triangle
        # Triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
        # Point (x,y) is inside if:
        # y >= 0
        # y <= sqrt(3)*x (left edge)
        # y <= sqrt(3)*(1-x) (right edge)
        for i in range(N):
            px, py = pts[i]
            if py < -1e-6 or py > np.sqrt(3)*px + 1e-6 or py > np.sqrt(3)*(1-px) + 1e-6:
                return 1.0  # Penalty for being outside
        return -min_triangle_area(pts)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    best_pts = result.x.reshape(-1, 2)
    
    # Clamp to be inside triangle
    for i in range(N):
        px, py = best_pts[i]
        if py < 0:
            py = 0.01
        if py > np.sqrt(3)*px:
            # Move inward
            t = 0.95
            px = t * px + (1-t) * 0.5
            py = t * py + (1-t) * np.sqrt(3)/6
        if py > np.sqrt(3)*(1-px):
            t = 0.95
            px = t * px + (1-t) * 0.5
            py = t * py + (1-t) * np.sqrt(3)/6
        best_pts[i] = [px, py]
    
    return best_pts
