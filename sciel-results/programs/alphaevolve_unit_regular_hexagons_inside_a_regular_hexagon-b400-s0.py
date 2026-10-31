import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 20.0

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate 11 points on a triangular lattice with slight expansion to prevent overlap
    # Axial coordinates (q, r)
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Scale factor slightly greater than 1 to ensure no overlap
    SCALE = 1.001
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
        
    pts = np.array(pts)
    
    # Function to compute minimal side length S for a given outer rotation beta (radians)
    # Optimizing the outer center (cx, cy) using Linear Programming
    def calc_S(beta):
        # Check time limit
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        # Normals of the outer hexagon
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Setup LP to minimize D (max absolute projection)
        # Variables: x, y, D
        # Minimize D
        c = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # Constraint 1: (px - x)*nx + (py - y)*ny <= D
                # px*nx - x*nx + py*ny - y*ny <= D
                # -nx*x - ny*y - D <= -(px*nx + py*ny)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                # Constraint 2: -((px - x)*nx + (py - y)*ny) <= D
                # (x - px)*nx + (y - py)*ny <= D
                # nx*x + ny*y - D <= px*nx + py*ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for x, y, D
        # D must be >= 0
        # x, y can be anything reasonable, but we can bound them to the range of pts
        min_x = np.min(pts[:, 0]) - 10
        max_x = np.max(pts[:, 0]) + 10
        min_y = np.min(pts[:, 1]) - 10
        max_y = np.max(pts[:, 1]) + 10
        
        bounds = [(min_x, max_x), (min_y, max_y), (0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                # S = (D + APOTHEM) / APOTHEM
                S = (D + APOTHEM) / APOTHEM
                return S
            else:
                return float('inf')
        except Exception:
            return float('inf')

    best_beta = 0.0
    best_S = calc_S(0.0)
    best_cx, best_cy = np.mean(pts, axis=0)
    
    # We need to recover the best cx, cy from the LP for the best beta
    # The calc_S function above only returns S. We need to modify it to return cx, cy as well or re-solve.
    
    # Let's re-structure to keep the best solution
    
    def solve_lp(beta):
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        c = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            return res.x[0], res.x[1], res.x[2]
        return None, None, float('inf')

    # Re-evaluate best_beta to get center
    best_cx, best_cy, best_D = solve_lp(best_beta)
    if best_D is float('inf'):
         # Fallback
         best_cx, best_cy = np.mean(pts, axis=0)
         # Calculate D for centroid
         max_D = 0
         for nx, ny in normals: # normals from last call? No, need to recompute for best_beta
             pass
         # Just use the S from calc_S
         pass

    # Minimize S over beta in [0, pi/6]
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize_scalar(calc_S, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 100})
            if res.fun < best_S:
                best_beta = res.x
                best_S = res.fun
                # Update center
                best_cx, best_cy, best_D = solve_lp(best_beta)
    except Exception:
        pass
    
    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
