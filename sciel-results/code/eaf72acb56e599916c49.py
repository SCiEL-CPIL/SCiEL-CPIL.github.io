import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0
        
    # Calculate min triangle area
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def _make_regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _make_config(boundary_n, interior_n, interior_positions='center'):
    points = []
    if boundary_n > 0:
        points.extend(_make_regular_polygon(boundary_n).tolist())
    
    if interior_n > 0:
        if interior_positions == 'center':
            # Place interior points in a small regular polygon around center
            if interior_n == 1:
                points.append([0.0, 0.0])
            else:
                # Small radius to avoid coinciding with center
                r = 0.1
                int_pts = _make_regular_polygon(interior_n, r)
                points.extend(int_pts.tolist())
                
    return np.array(points)

def construct():
    best_points = None
    best_score = -1.0
    
    # Try different splits of 13 points into boundary and interior
    # Boundary points form a regular polygon, interior points are placed symmetrically
    splits = [
        (13, 0),
        (12, 1),
        (11, 2),
        (10, 3),
        (9, 4),
        (8, 5),
        (7, 6),
        (6, 7),
    ]
    
    for b, i in splits:
        if b < 3: continue
        pts = _make_config(b, i)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts
            
    return best_points
