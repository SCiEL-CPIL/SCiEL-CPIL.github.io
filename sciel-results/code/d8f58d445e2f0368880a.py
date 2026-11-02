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
        # Fallback
        pts = []
        pts.append((0.0, 0.0))
        for i in range(6):
            ang = i * math.pi / 3
            r = math.sqrt(3)
            pts.append((r * math.cos(ang), r * math.sin(ang)))
        for i in range(4):
            ang = i * math.pi / 2
            r = 2 * math.sqrt(3)
            pts.append((r * math.cos(ang), r * math.sin(ang)))
        
        max_proj = 0
        for k in range(6):
            ang = math.pi/6 + k*math.pi/3
            nx, ny = math.cos(ang), math.sin(ang)
            for px, py in pts:
                proj = abs(px*nx + py*ny)
                if proj > max_proj:
                    max_proj = proj
        S = (max_proj + APOTHEM) / APOTHEM
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [0.0, 0.0], float(S), 0.0

    # Initial configuration
    pts_init = []
    pts_init.append((0.0, 0.0))
    for i in range(6):
        ang = i * math.pi / 3
        r = math.sqrt(3)
        pts_init.append((r * math.cos(ang), r * math.sin(ang)))
    for i in range(4):
        ang = i * math.pi / 2
        r = 2 * math.sqrt(3)
        pts_init.append((r * math.cos(ang), r * math.sin(ang)))
        
    x0 = []
    for px, py in pts_init:
        x0.append(px)
        x0.append(py)
    x0.append(0.0) # theta
    
    def objective(params):
        pts = np.array(params[:2*N]).reshape(N, 2)
        theta = params[2*N]
        
        max_proj = 0.0
        for k in range(6):
            ang = theta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_abs = np.max(np.abs(projs))
            if max_abs > max_proj:
                max_proj = max_abs
        
        S = (max_proj + APOTHEM) / APOTHEM
        
        penalty = 0.0
        min_dist = math.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return S + 1000.0 * penalty
    
    best_S = float('inf')
    best_params = None
    
    np.random.seed(42)
    for restart in range(10):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        if restart == 0:
            x_current = x0.copy()
        else:
            x_current = x0.copy()
            for i in range(N):
                x_current[2*i] += np.random.uniform(-1.0, 1.0)
                x_current[2*i+1] += np.random.uniform(-1.0, 1.0)
            x_current[2*N] = np.random.uniform(-math.pi/6, math.pi/6)
            
        try:
            res = minimize(objective, x_current, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        best_params = x0
        
    pts = np.array(best_params[:2*N]).reshape(N, 2)
    theta = best_params[2*N]
    
    max_proj = 0.0
    for k in range(6):
        ang = theta + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        projs = pts[:, 0] * nx + pts[:, 1] * ny
        max_abs = np.max(np.abs(projs))
        if max_abs > max_proj:
            max_proj = max_abs
            
    S = (max_proj + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), 0.0])
        
    center = [0.0, 0.0]
    side = float(S)
    rotation = float(math.degrees(theta))
    
    return inner, center, side, rotation
