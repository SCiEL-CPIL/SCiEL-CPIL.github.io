import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import random

N = 14

def _triangle_area(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns a negative value if invalid.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return -1e6
    except Exception:
        return -1e6
        
    min_area = float('inf')
    n = len(points)
    # Vectorized calculation of all triangle areas is faster
    # But for N=14, 364 triangles is small, loop is fine.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
                    
    if min_area < 1e-9:
        return -1e6
        
    return min_area / hull_area

def _negative_score(points_flat):
    points = points_flat.reshape(N, 2)
    return -_score(points)

def _hexagonal_init():
    """Generate a hexagonal lattice based initial configuration for N=14."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Ring 1 (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    
    # We need 7 more points.
    # Ring 2 has 12 points at distance sqrt(3).
    # We pick 7 of them. To maintain symmetry, we might pick specific ones.
    # Or just add 7 random points from the ring 2 set and let optimizer fix.
    # Actually, a better init for 14 might be a slightly distorted hexagonal grid.
    # Let's add 7 points from the next ring.
    # Ring 2 points:
    # (1.5, sqrt(3)/2), (1, sqrt(3)), (0, sqrt(3)), (-1, sqrt(3)), 
    # (-1.5, sqrt(3)/2), (-1.5, -sqrt(3)/2), (-1, -sqrt(3)), 
    # (0, -sqrt(3)), (1, -sqrt(3)), (1.5, -sqrt(3)/2), (1.5, sqrt(3)/2) is duplicate?
    # Let's list them properly.
    # Neighbors of (1,0) are (1.5, sqrt(3)/2) and (1.5, -sqrt(3)/2).
    # Neighbors of (0.5, sqrt(3)/2) are (1.5, sqrt(3)/2) and (0.5, sqrt(3)*1.5)?? No.
    # Lattice vectors: u=(1,0), v=(0.5, sqrt(3)/2).
    # Points are i*u + j*v.
    # Ring 1: (1,0), (0,1)->(0.5, .866), (-1,1)->(-.5, .866), (-1,0), (0,-1)->(.5, -.866), (1,-1)->(1.5, -.866)
    # Wait, (1,-1) is (1,0) - (0.5, .866) = (0.5, -.866).
    # My previous ring 1 was polar.
    # Let's stick to polar for simplicity of init, then optimize.
    
    # Add 7 points from a larger radius, say R=1.8, with angles spread out.
    for i in range(7):
        angle = i * 2 * np.pi / 7 + np.pi/7 # Offset to interleave
        pts.append([1.8 * np.cos(angle), 1.8 * np.sin(angle)])
        
    return np.array(pts)

def _regular_polygon_init():
    """Generate a regular 14-gon initial configuration."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.array([[np.cos(a), np.sin(a)] for a in angles])

def _random_init():
    """Generate random points in a unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.random.uniform(0.5, 1.0, N) # Avoid center clustering
    return np.array([[r * np.cos(a), r * np.sin(a)] for r, a in zip(radii, angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_score = -1e6
    best_points = _hexagonal_init()
    
    # List of initial configurations to try
    inits = [
        _hexagonal_init(),
        _regular_polygon_init(),
        _random_init()
    ]
    
    # Add some perturbed versions of the best ones
    for _ in range(3):
        inits.append(_hexagonal_init() + np.random.normal(0, 0.1, (N, 2)))
        inits.append(_regular_polygon_init() + np.random.normal(0, 0.1, (N, 2)))
        
    for init_pts in inits:
        if time.time() > deadline:
            break
            
        x0 = init_pts.flatten()
        
        try:
            res = minimize(_negative_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success or res.fun < -best_score:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res.x.reshape(N, 2)
                    
        except Exception:
            continue
            
    # Final check and return
    # Ensure the returned points are valid
    final_score = _score(best_points)
    if final_score < 0:
        # Fallback to a known good configuration if optimization failed badly
        best_points = _hexagonal_init()
        
    return best_points
