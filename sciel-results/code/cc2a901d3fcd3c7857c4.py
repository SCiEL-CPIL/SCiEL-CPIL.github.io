import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3.0)

def hex_norm(p):
    """
    Calculate the hexagonal norm of a point p (x, y).
    The hexagon is aligned with the x-axis (flat top/bottom).
    Normals are at 0, 60, 120, 180, 240, 300 degrees.
    """
    x, y = p
    # Normals: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # |x|
    # |0.5x + sqrt(3)/2 y|
    # |-0.5x + sqrt(3)/2 y|
    # etc.
    v1 = abs(x)
    v2 = abs(0.5 * x + SQRT3 / 2.0 * y)
    v3 = abs(-0.5 * x + SQRT3 / 2.0 * y)
    return max(v1, v2, v3)

def calc_S(centers, angles_deg):
    """
    Calculate the minimum outer side length S for given centers and angles.
    angles_deg are the rotations of the inner hexagons.
    """
    max_norm = 0.0
    # Vertices of a unit hexagon at origin, rotated by angle.
    # Base vertices at 0, 60, ...
    base_angles = np.deg2rad(np.arange(0, 360, 60))
    
    for i in range(len(centers)):
        cx, cy = centers[i]
        angle_rad = np.deg2rad(angles_deg[i])
        
        # Vertices of the i-th hexagon
        for ba in base_angles:
            # Vertex of unit hexagon at origin
            vx = np.cos(ba)
            vy = np.sin(ba)
            
            # Rotate by angle_rad
            rx = vx * np.cos(angle_rad) - vy * np.sin(angle_rad)
            ry = vx * np.sin(angle_rad) + vy * np.cos(angle_rad)
            
            # Translate to center
            px = cx + rx
            py = cy + ry
            
            n = hex_norm((px, py))
            if n > max_norm:
                max_norm = n
                
    # S = (2 / sqrt(3)) * max_norm
    return (2.0 / SQRT3) * max_norm

def overlap_penalty(centers):
    """
    Calculate penalty for overlapping hexagons.
    Assumes minimum distance is SQRT3.
    """
    penalty = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < SQRT3:
                penalty += (SQRT3 - dist) ** 2
    return penalty

def objective(x):
    """
    x is a flat array of 2*N coordinates: [x1, y1, x2, y2, ...]
    """
    centers = np.array(x).reshape(N, 2)
    # Angles are fixed at 30 degrees for all
    angles = np.full(N, 30.0)
    
    S = calc_S(centers, angles)
    pen = overlap_penalty(centers)
    
    # Weight for penalty
    # S is around 3-4. Penalty should be significant.
    return S + 10.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_score = 40.0  # Initial seed side length
    best_centers = None
    best_angles = None
    
    # Initial guesses
    # 1. Lattice cluster
    # Generate a hexagonal cluster of 12 points
    # Center (0,0)
    # Ring 1: 6 points at distance sqrt(3)
    # Ring 2: take 5 points from ring 2
    
    centers_list = []
    centers_list.append((0.0, 0.0))
    
    # Ring 1
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers_list.append((SQRT3 * np.cos(angle), SQRT3 * np.sin(angle)))
        
    # Ring 2: distance 2*sqrt(3)
    # We need 5 more.
    # Place them at angles 0, 60, 120, 180, 240?
    for k in range(5):
        angle = np.deg2rad(k * 60)
        centers_list.append((2 * SQRT3 * np.cos(angle), 2 * SQRT3 * np.sin(angle)))
        
    initial_centers = np.array(centers_list)
    x0 = initial_centers.flatten()
    
    # Run optimization
    result = None
    try:
        # Use Powell method, no gradient needed
        result = minimize(objective, x0, method='Powell', options={'maxiter': 500, 'disp': False})
    except Exception:
        pass
    
    if result is not None and result.fun < 40.0:
        best_centers = result.x.reshape(N, 2)
        best_angles = np.full(N, 30.0)
        best_score = calc_S(best_centers, best_angles)
        
    # Check time
    if time.time() > deadline:
        # Return best found so far
        if best_centers is None:
            # Fallback to seed
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
        else:
            inner = []
            for i in range(N):
                inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])
            return inner, (0.0, 0.0), best_score, 0.0

    # Try a few more random restarts if time permits
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 10.0:
            break
            
        # Random perturbation of the initial guess
        noise = np.random.normal(0, 1.0, size=(N, 2))
        x_rand = initial_centers.flatten() + noise
        
        try:
            res = minimize(objective, x_rand, method='Powell', options={'maxiter': 200, 'disp': False})
            if res.fun < best_score:
                best_centers = res.x.reshape(N, 2)
                best_angles = np.full(N, 30.0)
                best_score = calc_S(best_centers, best_angles)
        except Exception:
            continue

    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    else:
        inner = []
        for i in range(N):
            inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])
        return inner, (0.0, 0.0), best_score, 0.0

