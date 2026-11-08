import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = time.time() + 140.0

def _hex_vertices(cx, cy, side, angle_rad):
    angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _compute_enclosing(pos, angles, sides):
    """Fast computation of minimum enclosing regular hexagon."""
    n = len(pos)
    apothem = np.sqrt(3) / 2.0
    
    all_verts = []
    for j in range(n):
        v = _hex_vertices(pos[j, 0], pos[j, 1], sides[j], angles[j])
        all_verts.append(v)
    all_verts = np.concatenate(all_verts, axis=0)
    
    best_s = np.inf
    best_c = np.array([0.0, 0.0])
    best_theta = 0.0
    
    for theta in np.linspace(0, np.pi / 3, 9):
        normals = np.column_stack([np.cos(theta + np.arange(6) * np.pi / 3),
                                   np.sin(theta + np.arange(6) * np.pi / 3)])
        M = np.max(all_verts @ normals.T, axis=0)
        
        # Solve for optimal center using a quick Nelder-Mead
        def f(c):
            return np.max(M - c @ normals.T)
        c0 = np.mean(all_verts, axis=0)
        res = minimize(f, c0, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
        c = res.x
        s = np.max(M - c @ normals.T) / apothem
        if s < best_s:
            best_s = s
            best_c = c
            best_theta = theta
    
    return best_c[0], best_c[1], best_s, best_theta

def construct():
    global DEADLINE
    sides = np.ones(N)
    sqrt3 = np.sqrt(3)
    
    configs = []
    # Honeycomb cluster
    pos = [[0, 0]]
    for k in range(6):
        a = k * np.pi / 3
        pos.append([sqrt3 * np.cos(a), sqrt3 * np.sin(a)])
    for k in range(5):
        a = k * 2 * np.pi / 5 + np.pi / 6
        pos.append([2 * sqrt3 * np.cos(a), 2 * sqrt3 * np.sin(a)])
    configs.append(np.array(pos[:N]))
    
    # 3x4
    pos = []
    for r in range(3):
        for c in range(4):
            pos.append([c * sqrt3 + (r % 2) * sqrt3 / 2, r * 1.5])
    configs.append(np.array(pos))
    
    best_side = np.inf
    best_data = None
    
    for cfg in configs:
        if time.time() > DEADLINE - 40:
            break
        angles0 = np.zeros(N)
        x0 = np.concatenate([cfg.flatten(), angles0])
        
        def obj(params):
            if time.time() > DEADLINE - 15:
                return 1e10
            p = params[:2*N].reshape(N, 2)
            a = params[2*N:]
            try:
                _, _, s, _ = _compute_enclosing(p, a, sides)
            except:
                return 1e10
            return s
        
        bounds = [(-12, 12)] * 2*N + [(-np.pi/2, np.pi/2)] * N
        try:
            res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 80, 'ftol': 1e-9})
            if res.fun < best_side:
                p = res.x[:2*N].reshape(N, 2)
                a = res.x[2*N:]
                cx, cy, s, th = _compute_enclosing(p, a, sides)
                best_side = s
                best_data = (p, a, cx, cy, s, th)
        except:
            pass
    
    if best_data is None:
        p = configs[0]
        a = np.zeros(N)
        cx, cy, s, th = _compute_enclosing(p, a, sides)
        best_data = (p, a, cx, cy, s, th)
    
    p, a, cx, cy, s, th = best_data
    inner = [[p[i, 0], p[i, 1], np.degrees(a[i])] for i in range(N)]
    return inner, (cx, cy), s, np.degrees(th)
