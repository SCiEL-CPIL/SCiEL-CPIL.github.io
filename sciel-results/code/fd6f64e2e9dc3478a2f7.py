import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0

def hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(angle + k * np.pi / 3),
                      cy + side * np.sin(angle + k * np.pi / 3)] for k in range(6)])

def vertex_in_hex(vx, vy, cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    dx = vx - cx
    dy = vy - cy
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        theta = angle + k * np.pi / 3
        proj = dx * np.cos(theta) + dy * np.sin(theta)
        if proj > inradius:
            return False
    return True

def construct():
    start_time = time.time()
    
    def decode(params):
        inner = []
        for i in range(N):
            inner.append([params[i*3], params[i*3+1], params[i*3+2]])
        return inner, (0.0, 0.0), params[N*3], 0.0
    
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return 1e10
        inner, center, side, angle = decode(params)
        if side < 1.0:
            return 1e10
        
        penalty = 0.0
        for i in range(N):
            x, y, ang = inner[i]
            verts = hex_vertices(x, y, 1.0, ang)
            for v in verts:
                if not vertex_in_hex(v[0], v[1], 0.0, 0.0, side, 0.0):
                    penalty += 1.0
        
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, a1 = inner[i]
                x2, y2, a2 = inner[j]
                dx, dy = x1-x2, y1-y2
                if dx*dx + dy*dy > 6.25:
                    continue
                verts1 = hex_vertices(x1, y1, 1.0, a1)
                verts2 = hex_vertices(x2, y2, 1.0, a2)
                for v in verts1:
                    if vertex_in_hex(v[0], v[1], x2, y2, 1.0, a2):
                        penalty += 1.0
                for v in verts2:
                    if vertex_in_hex(v[0], v[1], x1, y1, 1.0, a1):
                        penalty += 1.0
        
        return side + 50.0 * penalty
    
    s3 = np.sqrt(3)
    positions = [
        (0, 0),
        (2, 0), (1, s3), (-1, s3), (-2, 0), (-1, -s3), (1, -s3),
        (3, s3), (2, 2*s3), (0, 2*s3), (-2, 2*s3), (-3, s3),
    ]
    init = np.zeros(37)
    for i in range(N):
        init[i*3] = positions[i][0]
        init[i*3+1] = positions[i][1]
    init[N*3] = 5.0
    
    best_params = init.copy()
    best_val = objective(init)
    
    result = minimize(objective, init, method='Nelder-Mead',
                     options={'maxiter': 8000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
    if result.fun < best_val:
        best_params = result.x
        best_val = result.fun
    
    if time.time() - start_time < DEADLINE - 10:
        for seed in [1, 2, 3]:
            if time.time() - start_time > DEADLINE - 10:
                break
            np.random.seed(seed)
            pert = init + np.random.randn(37) * 0.3
            pert[N*3] = max(3.0, 5.0 + np.random.randn() * 0.5)
            try:
                r = minimize(objective, pert, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
                if r.fun < best_val:
                    best_params = r.x
                    best_val = r.fun
            except Exception:
                pass
    
    inner, center, side, angle = decode(best_params)
    return inner, center, side, angle
