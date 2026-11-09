import numpy as np
from scipy.optimize import minimize_scalar

N = 11

def construct():
    # Generate points
    pts = []
    # Center
    pts.append((0.0, 0.0))
    
    # 6 neighbors
    sqrt3 = np.sqrt(3)
    neighbors = [
        (0.0, sqrt3),
        (1.5, sqrt3/2),
        (1.5, -sqrt3/2),
        (0.0, -sqrt3),
        (-1.5, -sqrt3/2),
        (-1.5, sqrt3/2)
    ]
    pts.extend(neighbors)
    
    # 4 extensions
    # Extend the 6 neighbors in their respective directions
    # Directions are the vectors of the neighbors themselves (since they are at distance sqrt3 and angle is the direction)
    # Wait, the vector from center to neighbor IS the direction.
    # So extension = 2 * neighbor_vector
    extensions = [
        (3.0, sqrt3),      # 2 * (1.5, sqrt3/2)
        (3.0, -sqrt3),     # 2 * (1.5, -sqrt3/2)
        (-3.0, sqrt3),     # 2 * (-1.5, sqrt3/2)
        (-3.0, -sqrt3)     # 2 * (-1.5, -sqrt3/2)
    ]
    pts.extend(extensions)
    
    pts = np.array(pts)
    
    # Center the cluster
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    
    # Function to evaluate side length for a given rotation
    def calc_side(theta):
        # Normal vectors of the outer hexagon
        # Hexagon with side S has width S*sqrt(3) in the direction normal to the sides.
        # The 6 normals are at angles theta, theta+60, ..., theta+300
        # We need the max projection in each normal direction and the min projection.
        # Width W_i = max(proj_i) - min(proj_i)
        # Required S_i = W_i / sqrt(3)
        # S = max(S_i)
        
        angles = theta + np.arange(0, 360, 60) * np.pi / 180
        widths = []
        for a in angles:
            n = np.array([np.cos(a), np.sin(a)])
            projs = pts_centered @ n
            w = np.max(projs) - np.min(projs)
            widths.append(w / sqrt3)
        return max(widths)
    
    # Optimize theta in [0, pi/3]
    res = minimize_scalar(calc_side, bounds=(0, np.pi/3), method='bounded')
    best_theta = res.x
    best_side = res.fun
    
    # Construct return values
    # Inner points: pts (original, not centered? No, center of outer is centroid)
    # If outer center is centroid, then inner points are pts (which are already absolute)
    # Wait, I calculated pts_centered for optimization.
    # The absolute positions are pts.
    # The outer center is centroid.
    # The outer angle is best_theta.
    
    inner_list = []
    for p in pts:
        inner_list.append([p[0], p[1], 0.0])
        
    return inner_list, (centroid[0], centroid[1]), best_side, np.degrees(best_theta)
