import numpy as np
from scipy.optimize import minimize
import time

# Constants
N = 11
DEADLINE = 110.0 # Safety margin before 120s limit
START_TIME = time.time()

def hex_axial_to_pixel(q, r):
    """Convert axial coordinates to pixel coordinates for a flat-top hexagon grid? 
    Actually, orientation doesn't matter for the relative geometry as long as consistent.
    Standard pointy-top:
    x = sqrt(3) * (q + r/2)
    y = 3/2 * r
    """
    x = np.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def generate_shapes():
    """Generate a list of candidate point sets for 11 unit hexagons."""
    shapes = []
    
    # Shape 1: Compact cluster
    # Axial coords
    coords1 = [
        (0, 0), (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (-1, 1),
        (2, 0), (2, 1), (-2, 0) # Need 11, currently 9. Add 2 more.
    ]
    # Let's construct specific known good 11-hex shapes
    
    # Shape A: 3-4-3-1
    # r=0: q=-1,0,1,2
    # r=1: q=-1,0,1
    # r=-1: q=-1,0,1
    # r=2: q=0
    coords_A = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, -1), (0, -1), (1, -1),
        (0, 2)
    ]
    shapes.append([hex_axial_to_pixel(q, r) for q, r in coords_A])

    # Shape B: 2-3-3-3
    # r=0: q=0,1
    # r=1: q=-1,0,1
    # r=-1: q=-1,0,1
    # r=2: q=-1,0,1  -> Wait, let's make it symmetric-ish
    coords_B = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, -1), (0, -1), (1, -1),
        (-1, 2), (0, 2), (1, 2)
    ]
    shapes.append([hex_axial_to_pixel(q, r) for q, r in coords_B])
    
    # Shape C: Flower of 7 + 4 in a line
    # Center (0,0)
    # Neighbors: (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1)
    # Extra: (2,0), (3,0), (2,1), (3,1) ?
    coords_C = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (2, 1), (3, 0), (3, 1)
    ]
    shapes.append([hex_axial_to_pixel(q, r) for q, r in coords_C])

    # Shape D: Another compact one
    # r=0: -1, 0, 1, 2, 3 (5)
    # r=1: -1, 0, 1, 2 (4)
    # r=-1: -1, 0, 1 (3)
    # Total 12? No.
    # r=0: -1, 0, 1, 2 (4)
    # r=1: -1, 0, 1, 2 (4)
    # r=-1: -1, 0, 1 (3)
    # Total 11.
    coords_D = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, -1), (0, -1), (1, -1)
    ]
    shapes.append([hex_axial_to_pixel(q, r) for q, r in coords_D])

    return shapes

def objective(params, points):
    """
    params: [cx, cy, alpha]
    points: Nx2 array of inner hexagon centers.
    Returns: Side length S of the outer hexagon.
    """
    cx, cy, alpha = params
    # Shift points
    pts = points - np.array([cx, cy])
    
    # Normals of the outer hexagon
    # The outer hexagon has side S.
    # Distance from center to side (inradius) r = S * sqrt(3) / 2.
    # S = 2 * r / sqrt(3).
    # We want to minimize S, so we minimize r_max = max projection.
    
    # The 3 unique normal directions for a hexagon rotated by alpha
    # Angles: alpha, alpha + 60, alpha + 120
    angles = alpha + np.array([0, np.pi/3, 2*np.pi/3])
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Project all points onto all 3 normals
    # projections shape: (N, 3)
    projections = pts @ normals.T
    
    # Max absolute projection
    r_max = np.max(np.abs(projections))
    
    # Side length
    S = 2 * r_max / np.sqrt(3)
    
    return S

def construct():
    global START_TIME, DEADLINE
    
    if time.time() - START_TIME > DEADLINE:
        # Fallback: return seed-like result to avoid timeout penalty if we are too late
        # But we start fresh, so this shouldn't happen immediately.
        pass

    shapes = generate_shapes()
    
    best_S = float('inf')
    best_params = None
    best_shape_idx = 0
    
    for idx, shape in enumerate(shapes):
        if time.time() - START_TIME > DEADLINE:
            break
            
        points = np.array(shape)
        
        # Initial guess: centroid and alpha=0
        centroid = np.mean(points, axis=0)
        x0 = [centroid[0], centroid[1], 0.0]
        
        # Try a few initial alphas
        for alpha_init in [0.0, np.pi/6, np.pi/3]:
            if time.time() - START_TIME > DEADLINE:
                break
                
            x0[2] = alpha_init
            
            try:
                res = minimize(objective, x0, args=(points,), method='Nelder-Mead', 
                               options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 500})
                if res.success and res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x
                    best_shape_idx = idx
            except Exception:
                continue

    if best_params is None:
        # Fallback if optimization failed
        points = np.array(shapes[0])
        centroid = np.mean(points, axis=0)
        best_params = [centroid[0], centroid[1], 0.0]
        best_S = objective(best_params, points)
        best_shape_idx = 0

    # Extract final values
    cx, cy, alpha = best_params
    alpha_deg = np.degrees(alpha)
    
    # Construct inner list
    # Inner hexagons are at the shape's points, angle 0
    points = np.array(shapes[best_shape_idx])
    inner = [[float(p[0]), float(p[1]), 0.0] for p in points]
    
    # Center
    center = (float(cx), float(cy))
    
    # Side
    side = float(best_S)
    
    # Angle
    angle = float(alpha_deg)
    
    return inner, center, side, angle

