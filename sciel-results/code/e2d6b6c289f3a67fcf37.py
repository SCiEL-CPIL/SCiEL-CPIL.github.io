import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define the 11 hexagon centers in axial coordinates
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (-2, 0)
    ]
    
    # Convert to cartesian and SCALE by sqrt(3) to ensure touching
    scale = math.sqrt(3)
    pts = []
    for q, r in axial:
        x = scale * math.sqrt(3) * (q + r / 2.0)
        y = scale * 1.5 * r
        pts.append((x, y))
    pts = np.array(pts)
    
    # Center the points at origin
    cx_mean = np.mean(pts[:, 0])
    cy_mean = np.mean(pts[:, 1])
    pts_centered = pts - np.array([cx_mean, cy_mean])
    
    def width_centers(theta):
        nx = math.cos(theta)
        ny = math.sin(theta)
        projs = pts_centered @ np.array([nx, ny])
        return np.max(projs) - np.min(projs)

    def width_hex_unit(rel_angle):
        # rel_angle is the angle of the normal relative to the hexagon's orientation
        # Find distance to nearest multiple of pi/3
        period = math.pi / 3.0
        # Map to [0, period)
        m = round(rel_angle / period)
        delta = abs(rel_angle - m * period)
        # Width is sqrt(3) / cos(delta)
        return math.sqrt(3) / math.cos(delta)

    def calc_S(params):
        theta_outer = params[0]
        alpha = params[1] # rotation of inner hexagons
        
        max_S = 0.0
        for k in range(6):
            n_ang = theta_outer + k * math.pi / 3.0
            w_c = width_centers(n_ang)
            rel_ang = n_ang - alpha
            w_h = width_hex_unit(rel_ang)
            
            # S * sqrt(3) >= w_c + w_h
            S_k = (w_c + w_h) / math.sqrt(3)
            if S_k > max_S:
                max_S = S_k
        return max_S

    # Optimize theta_outer and alpha
    # Periodicity: theta_outer has period pi/3. alpha has period pi/3.
    # Search in [0, pi/3] x [0, pi/3]
    
    best_S = float('inf')
    best_params = [0.0, 0.0]
    
    # Coarse grid search to find good initial point
    n_grid = 20
    for i in range(n_grid):
        for j in range(n_grid):
            t = i * (math.pi / 3.0) / (n_grid - 1)
            a = j * (math.pi / 3.0) / (n_grid - 1)
            S = calc_S([t, a])
            if S < best_S:
                best_S = S
                best_params = [t, a]
                
    # Fine optimization
    try:
        res = minimize(calc_S, best_params, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
    except Exception:
        pass
        
    # Final polish with tighter bounds if time allows
    if time.time() - START_TIME < TIME_LIMIT - 20:
        try:
            res2 = minimize(calc_S, best_params, method='Powell', 
                       options={'maxiter': 2000, 'ftol': 1e-10})
            if res2.fun < best_S:
                best_S = res2.fun
                best_params = res2.x
        except Exception:
            pass

    theta_outer = best_params[0]
    alpha = best_params[1]
    
    # Ensure angles are in a reasonable range for output
    angle_deg_inner = math.degrees(alpha)
    angle_deg_outer = math.degrees(theta_outer)
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg_inner)])
        
    center = [float(cx_mean), float(cy_mean)]
    side = float(best_S)
    rotation = float(angle_deg_outer)
    
    return inner, center, side, rotation
