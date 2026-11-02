import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0

def hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    return np.array([[cx + s * np.cos(a + k * np.pi / 3.0),
                      cy + s * np.sin(a + k * np.pi / 3.0)] for k in range(6)])

def min_enclosing_hex(pts, outer_ang):
    """pts: (n,2) array. Returns (side, cx, cy)."""
    a = np.radians(outer_ang)
    normals = np.array([[np.cos(a + k * np.pi / 3.0), np.sin(a + k * np.pi / 3.0)] for k in range(6)])
    proj = pts @ normals.T
    mn = proj.min(axis=0)
    mx = proj.max(axis=0)
    mid = (mn + mx) / 2.0
    hr = (mx - mn) / 2.0
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for k in range(6):
        nx, ny = normals[k]
        A_ub.append([nx, ny, -1.0])
        b_ub.append(mid[k])
        A_ub.append([-nx, -ny, 1.0])
        b_ub.append(mid[k])
    for k in range(6):
        A_ub.append([0.0, 0.0, -1.0])
        b_ub.append(-hr[k])
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None,None),(None,None),(0,None)], method='highs')
    if res.success:
        apothem = res.x[2]
        return 2.0 * apothem / np.sqrt(3.0), res.x[0], res.x[1]
    else:
        apothem = np.max(hr)
        return 2.0 * apothem / np.sqrt(3.0), 0.0, 0.0

def get_all_pts(inner):
    pts = []
    for x, y, ang in inner:
        pts.append(hex_vertices(x, y, 1.0, ang))
    return np.vstack(pts)

def objective(params):
    inner = np.array(params).reshape(N, 3)
    pts = get_all_pts(inner)
    best = np.inf
    for oa in (0.0, 30.0):
        s, _, _ = min_enclosing_hex(pts, oa)
        if s < best:
            best = s
    return best

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT - 10:
        # Emergency return
        inner = [[np.sqrt(3)*k, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 50.0, 0.0
    
    d = np.sqrt(3.0)
    # Initial: 1 center + 6 ring + 5 outer
    init = [(0.0, 0.0, 0.0)]
    for k in range(6):
        ang = k * 60.0
        init.append((d*np.cos(np.radians(ang)), d*np.sin(np.radians(ang)), 0.0))
    for k in range(5):
        ang = k*72.0 + 30.0
        r = 2.0*d
        init.append((r*np.cos(np.radians(ang)), r*np.sin(np.radians(ang)), 0.0))
    
    params0 = np.array([v for p in init for v in p])
    bounds = [(-4,4),(-4,4),(-180,180)]*N
    
    best_params = params0
    best_score = objective(params0)
    
    elapsed = time.time() - START_TIME
    if elapsed < 70:
        try:
            res = differential_evolution(objective, bounds, seed=42, maxiter=40, tol=1e-3, popsize=12)
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except:
            pass
    
    elapsed = time.time() - START_TIME
    if elapsed < 110:
        try:
            res2 = minimize(objective, best_params, method='L-BFGS-B', bounds=bounds, options={'maxiter':150})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except:
            pass
    
    inner = np.array(best_params).reshape(N, 3)
    pts = get_all_pts(inner)
    
    best_side = np.inf
    best_outer = (0.0, 0.0, 100.0, 0.0)
    for oa in np.arange(0, 60, 2.0):
        s, cx, cy = min_enclosing_hex(pts, oa)
        if s < best_side:
            best_side = s
            best_outer = (cx, cy, s, oa)
    
    # Refine
    oa0 = best_outer[3]
    for oa in np.arange(oa0-2, oa0+2, 0.25):
        s, cx, cy = min_enclosing_hex(pts, oa)
        if s < best_side:
            best_side = s
            best_outer = (cx, cy, s, oa)
    
    cx, cy, side, oa = best_outer
    inner_list = [[float(inner[i,0]), float(inner[i,1]), float(inner[i,2])] for i in range(N)]
    return inner_list, (float(cx), float(cy)), float(side), float(oa)
