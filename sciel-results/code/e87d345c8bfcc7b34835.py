import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(center, side, angle_deg):
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def min_separation(p1, p2):
    def get_normals(p):
        edges = np.roll(p, -1, axis=0) - p
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return normals / norms

    n1 = get_normals(p1)
    n2 = get_normals(p2)
    min_dist = float('inf')
    for n in np.vstack((n1, n2)):
        proj1 = np.dot(p1, n)
        proj2 = np.dot(p2, n)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        gap = min(max1, max2) - max(min1, min2)
        dist = -gap
        if dist < min_dist:
            min_dist = dist
    return min_dist

def point_in_poly_dist(p, poly):
    inside = True
    for i in range(len(poly)):
        p1 = poly[i]
        p2 = poly[(i + 1) % len(poly)]
        v1 = p2 - p1
        v2 = p - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if i == 0:
            sign = np.sign(cross)
            if sign == 0: continue
        else:
            if np.sign(cross) != sign:
                inside = False
                break
    if inside:
        return 0.0
    min_d = float('inf')
    for i in range(len(poly)):
        p1 = poly[i]
        p2 = poly[(i + 1) % len(poly)]
        v = p2 - p1
        w = p - p1
        c1 = np.dot(w, v)
        if c1 <= 0:
            d = np.linalg.norm(w)
        else:
            c2 = np.dot(v, v)
            if c2 <= c1:
                d = np.linalg.norm(w - v)
            else:
                t = c1 / c2
                v_proj = t * v
                d = np.linalg.norm(w - v_proj)
        if d < min_d:
            min_d = d
    return min_d

def objective(params):
    S = params[0]
    if S < 1.0: S = 1.0
    inner_hexes = []
    for i in range(N):
        x = params[1 + 3*i]
        y = params[2 + 3*i]
        t = params[3 + 3*i]
        verts = hex_vertices((x, y), 1.0, t)
        inner_hexes.append(verts)
    outer_verts = hex_vertices((0.0, 0.0), S, 0.0)
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = min_separation(inner_hexes[i], inner_hexes[j])
            if d < 0:
                penalty += -d
    for i in range(N):
        for p in inner_hexes[i]:
            d = point_in_poly_dist(p, outer_verts)
            penalty += d
    return S + 100.0 * penalty

def construct():
    start_time = time.time()
    deadline = 110.0
    best_valid_params = None
    best_valid_score = float('inf')
    
    # Initial guess
    init_params = np.zeros(1 + 3*N)
    init_params[0] = 6.0
    positions = []
    for i in range(3):
        for j in range(4):
            if len(positions) >= N: break
            x = (j - 1.5) * np.sqrt(3)
            y = (i - 1.5) * 1.5
            if i % 2 == 1:
                x += np.sqrt(3)/2
            positions.append((x, y))
    for i in range(N):
        init_params[1 + 3*i] = positions[i][0]
        init_params[2 + 3*i] = positions[i][1]
        init_params[3 + 3*i] = 0.0
        
    try:
        res = minimize(objective, init_params, method='Powell', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_valid_score:
             # Check validity
             # We need to re-calculate penalty to be sure
             # For simplicity, assume low objective implies low penalty if S is reasonable
             # But let's extract penalty from objective?
             # obj = S + 100*pen. pen = (obj - S)/100
             S = res.x[0]
             pen = (res.fun - S) / 100.0
             if pen < 1e-5:
                 best_valid_params = res.x
                 best_valid_score = S
    except Exception:
        pass
        
    if time.time() - start_time < deadline - 5 and best_valid_params is not None:
        try:
            res2 = minimize(objective, best_valid_params, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})
            S = res2.x[0]
            pen = (res2.fun - S) / 100.0
            if pen < 1e-5 and S < best_valid_score:
                best_valid_params = res2.x
                best_valid_score = S
        except Exception:
            pass
            
    if best_valid_params is None:
        # Fallback to seed-like behavior if optimization failed
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
        
    S = best_valid_params[0]
    inner = []
    for i in range(N):
        x = best_valid_params[1 + 3*i]
        y = best_valid_params[2 + 3*i]
        t = best_valid_params[3 + 3*i]
        inner.append([x, y, t])
        
    return inner, (0.0, 0.0), S, 0.0
