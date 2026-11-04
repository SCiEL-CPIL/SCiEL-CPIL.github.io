"""Heilbronn triangles: maximize the smallest triangle area for 11 points in an equilateral triangle."""
import numpy as np
import itertools
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit
    
    # Vertices of the equilateral triangle
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    def min_triangle_area(points):
        """Compute the minimum area of any triangle formed by 3 of the points."""
        min_a = float('inf')
        for i, j, k in itertools.combinations(range(N), 3):
            p1, p2, p3 = points[i], points[j], points[k]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            area = 0.5 * abs(cross)
            if area < min_a:
                min_a = area
        return min_a
    
    def is_inside_triangle(pt, tol=1e-9):
        """Check if a point is inside the equilateral triangle."""
        x, y = pt
        if y < -tol:
            return False
        if y > np.sqrt(3)*x + tol:
            return False
        if y > -np.sqrt(3)*(x-1) + tol:
            return False
        return True
    
    def project_to_triangle(pt):
        """Project a point to the nearest point inside the triangle."""
        x, y = pt
        # Check against three edges
        # Edge AB: y >= 0
        if y < 0:
            y = 0
        # Edge AC: y <= sqrt(3)*x  =>  sqrt(3)*x - y >= 0
        if y > np.sqrt(3)*x:
            # Project onto line y = sqrt(3)*x
            # Distance to line: |sqrt(3)*x - y| / sqrt(3+1) = (sqrt(3)*x - y)/2
            # Foot of perpendicular
            t = (x*np.sqrt(3) + y*3) / (3 + 1)  # projection parameter
            x_proj = (x + np.sqrt(3)*y) / 4 * 2  # simplified
            # Better: use proper projection
            # Line: sqrt(3)*x - y = 0, normal = (sqrt(3), -1)/2
            d = (np.sqrt(3)*x - y) / 2.0
            x = x - d * np.sqrt(3)
            y = y + d
        # Edge BC: y <= -sqrt(3)*(x-1)  =>  sqrt(3)*(x-1) + y <= 0
        if y > -np.sqrt(3)*(x-1):
            d = (np.sqrt(3)*(x-1) + y) / 2.0
            x = x - d * np.sqrt(3)
            y = y + d
        # Clamp to triangle bounds
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(np.sqrt(3)/2, y))
        return np.array([x, y])
    
    def log_sum_exp_surrogate(points, temperature=0.05):
        """Compute a smooth surrogate: log-sum-exp of negative areas, scaled by temperature."""
        areas = []
        for i, j, k in itertools.combinations(range(N), 3):
            p1, p2, p3 = points[i], points[j], points[k]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            area = 0.5 * abs(cross)
            areas.append(area)
        areas = np.array(areas)
        # We want to maximize the min area, so minimize -min area
        # Use log-sum-exp of (-areas/temperature) as a smooth approximation of -min(areas)/temperature
        max_neg = np.max(-areas / temperature)
        lse = max_neg + np.log(np.sum(np.exp(-areas / temperature - max_neg)))
        return lse  # minimizing this maximizes min area
    
    def gradient_descent(points, lr=0.01, steps=200, temperature=0.05):
        """Riemannian-like gradient ascent on min triangle area using surrogate."""
        pts = np.array(points, dtype=float)
        for step in range(steps):
            if time.time() > deadline:
                break
            # Compute gradient of the surrogate w.r.t. each point
            grad = np.zeros_like(pts)
            for i, j, k in itertools.combinations(range(N), 3):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                area = 0.5 * abs(cross)
                if area < 1e-12:
                    continue
                # Gradient of area w.r.t. each point
                # area = 0.5 * |cross|
                # d(cross)/dp1 = (-(p3[1]-p1[1]), (p3[0]-p1[0]))  ... need to be careful
                # cross = (p2x-p1x)(p3y-p1y) - (p2y-p1y)(p3x-p1x)
                # d(cross)/dp1x = -(p3y-p1y) + (p2y-p1y) = p2y - p3y
                # d(cross)/dp1y = (p3x-p1x) - (p2x-p1x) = p3x - p2x
                # d(cross)/dp2x = (p3y-p1y)
                # d(cross)/dp2y = -(p3x-p1x)
                # d(cross)/dp3x = -(p2y-p1y)
                # d(cross)/dp3y = (p2x-p1x)
                sign = 1.0 if cross > 0 else -1.0
                g1x = sign * (p2[1] - p3[1])
                g1y = sign * (p3[0] - p2[0])
                g2x = sign * (p3[1] - p1[1])
                g2y = sign * (-(p3[0] - p1[0]))
                g3x = sign * (-(p2[1] - p1[1]))
                g3y = sign * (p2[0] - p1[0])
                
                # For maximizing min area, we push in direction of increasing area
                # For the surrogate, weight by softmax
                # Simple: just use the area gradient directly
                w = 1.0 / (area + 1e-10)  # weight smaller triangles more
                
                grad[i] += w * np.array([g1x, g1y])
                grad[j] += w * np.array([g2x, g2y])
                grad[k] += w * np.array([g3x, g3y])
            
            # Update
            pts += lr * grad
            
            # Project back to triangle
            for i in range(N):
                pts[i] = project_to_triangle(pts[i])
        
        return pts
    
    # Initialize with a reasonable configuration
    best_points = None
    best_score = 0.0
    
    # Try multiple random initializations
    n_init = 5
    for init_i in range(n_init):
        if time.time() > deadline:
            break
        
        if init_i == 0:
            # Regular grid-like initialization
            points = []
            # Place points in a roughly uniform pattern
            for row in range(3):
                for col in range(3):
                    if len(points) >= N:
                        break
                    # Barycentric coordinates for a 3x3 grid
                    u = col / 2.0
                    v = row / 2.0
                    if u + v <= 1.0:
                        pt = u * B + v * C
                        points.append(pt)
            # Add remaining points
            while len(points) < N:
                u, v = np.random.rand(2)
                if u + v < 1.0:
                    pt = u * B + v * C
                    points.append(pt)
        else:
            # Random initialization
            points = []
            for _ in range(N):
                u, v = np.random.rand(2)
                if u + v < 1.0:
                    pt = u * B + v * C
                    points.append(pt)
                else:
                    points.append(np.random.rand(2) * np.array([1.0, np.sqrt(3)/2]))
        
        pts = np.array(points, dtype=float)
        
        # Run gradient ascent
        pts = gradient_descent(pts, lr=0.02, steps=300, temperature=0.05)
        
        score = min_triangle_area(pts)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # If we haven't found anything good, use a fallback
    if best_points is None:
        best_points = np.array([project_to_triangle(np.random.rand(2) * np.array([1.0, np.sqrt(3)/2])) for _ in range(N)])
    
    return best_points
