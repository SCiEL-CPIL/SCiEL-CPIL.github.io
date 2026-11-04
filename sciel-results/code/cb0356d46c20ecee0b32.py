import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.pi/6 + np.arange(6) * np.pi/3
    return np.column_stack([cx + side*np.cos(angles), cy + side*np.sin(angles)])

def max_violation(params, outer_side, ocx, ocy, oangle):
    """Max amount any inner vertex violates outer hexagon boundary."""
    a = np.radians(oangle)
    cos_a, sin_a = np.cos(-a), np.sin(-a)
    apothem = outer_side * np.sqrt(3) / 2
    max_v = 0.0
    for i in range(N):
        ix, iy, ia = params[3*i], params[3*i+1], params[3*i+2]
        verts = hex_vertices(ix, iy, 1.0, ia)
        for vx, vy in verts:
            dx, dy = vx - ocx, vy - ocy
            lx = dx*cos_a - dy*sin_a
            ly = dx*sin_a + dy*cos_a
            for k in range(6):
                theta = np.radians(60*k)
                d = lx*np.cos(theta) + ly*np.sin(theta)
                v = d - apothem
                if v > max_v:
                    max_v = v
    return max_v

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    hex_pos = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2),(-2,1)]
    init = np.zeros(3*N)
    for i,(q,r) in enumerate(hex_pos):
        init[3*i] = 1.5*q
        init[3*i+1] = sqrt3*(r+q/2)
    
    best_score = float('inf')
    best_x = None
    np.random.seed(123)
    
    for restart in range(30):
        if time.time() > deadline - 8.0:
            break
        p0 = init + (np.random.normal(0,0.3,3*N) if restart>0 else 0)
        
        for os in [4.5, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1]:
            if time.time() > deadline - 3.0:
                break
            
            def obj(x):
                return max(0, max_violation(x[:3*N], os, x[3*N], x[3*N+1], x[3*N+2]))
            
            x0 = np.concatenate([p0, [0,0,0]])
            try:
                res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter':300})
                if res.fun < 1e-4 and res.x is not None:
                    if os < best_score:
                        best_score = os
                        best_x = res.x.copy()
            except:
                pass
    
    if best_x is not None:
        inner = [[best_x[3*i], best_x[3*i+1], best_x[3*i+2]] for i in range(N)]
        return inner, (best_x[3*N], best_x[3*N+1]), best_score, best_x[3*N+2]
    
    inner = [[1.5*q, sqrt3*(r+q/2), 0.0] for q,r in hex_pos]
    return inner, (0.0, 0.0), 5.0, 0.0

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
