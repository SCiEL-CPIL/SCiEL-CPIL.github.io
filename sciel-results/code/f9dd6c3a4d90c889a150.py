import numpy as np
from scipy.optimize import minimize

N = 11

def get_hex_vertices(cx, cy, angle, side=1.0):
    base_angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    angles = base_angles + angle
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def get_outer_hex_vertices(S, angle):
    base_angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    angles = base_angles + angle
    x = S * np.cos(angles)
    y = S * np.sin(angles)
    return np.column_stack((x, y))

def min_distance_hex_hex(v1, v2):
    min_d = float('inf')
    for i in range(6):
        p = v1[i]
        a = v2[i]
        b = v2[(i+1)%6]
        ab = b - a
        ap = p - a
        len_sq = np.dot(ab, ab)
        if len_sq < 1e-10:
            d = np.linalg.norm(ap)
        else:
            t = np.dot(ap, ab) / len_sq
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
            d = np.linalg.norm(p - proj)
        if d < min_d:
            min_d = d
            
    for i in range(6):
        p = v2[i]
        a = v1[i]
        b = v1[(i+1)%6]
        ab = b - a
        ap = p - a
        len_sq = np.dot(ab, ab)
        if len_sq < 1e-10:
            d = np.linalg.norm(ap)
        else:
            t = np.dot(ap, ab) / len_sq
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
            d = np.linalg.norm(p - proj)
        if d < min_d:
            min_d = d
            
    return min_d

def construct():
    # Variables: x1, y1, a1, ..., xN, yN, aN, S, angle_outer
    # Total 3*N + 2 variables
    
    def objective(params):
        S = params[-2]
        angle_outer = params[-1]
        
        if S < 0.5:
            S = 0.5
            
        centers_x = params[0:2*N:3]
        centers_y = params[1:2*N+1:3]
        angles = params[2:2*N+2:3]
        
        inner_verts = []
        for i in range(N):
            v = get_hex_vertices(centers_x[i], centers_y[i], angles[i])
            inner_verts.append(v)
            
        outer_verts = get_outer_hex_vertices(S, angle_outer)
        
        penalty = 0.0
        
        # Check non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                d = min_distance_hex_hex(inner_verts[i], inner_verts[j])
                if d < 0:
                    penalty += d * d
        
        # Check containment using signed distance to lines
        # For each inner vertex, check against all outer edges
        for i in range(N):
            v = inner_verts[i]
            for p in v:
                for k in range(6):
                    a = outer_verts[k]
                    b = outer_verts[(k+1)%6]
                    ab = b - a
                    ap = p - a
                    cross = ab[0]*ap[1] - ab[1]*ap[0]
                    # Normal vector magnitude is |ab|
                    # Signed distance to the line is cross / |ab|
                    # Since vertices are CCW, cross > 0 means inside.
                    # cross < 0 means outside.
                    if cross < 0:
                        # Distance to the line
                        d_line = -cross / np.linalg.norm(ab)
                        # We only penalize if it's outside
                        penalty += d_line * d_line
                        
        return S + 100.0 * penalty

    # Initial guess
    d = np.sqrt(3) + 0.1
    x0 = np.zeros(3 * N + 2)
    
    x0[0] = 0.0
    x0[1] = 0.0
    x0[2] = 0.0
    
    idx = 1
    for k in range(6):
        ang = k * np.pi / 3
        x0[3*idx] = d * np.cos(ang)
        x0[3*idx+1] = d * np.sin(ang)
        x0[3*idx+2] = 0.0
        idx += 1
        
    for i in range(4):
        ang = (i * np.pi / 2) + np.pi/4
        r = 2.5 * d
        x0[3*idx] = r * np.cos(ang)
        x0[3*idx+1] = r * np.sin(ang)
        x0[3*idx+2] = 0.0
        idx += 1
        
    x0[-2] = 4.0
    x0[-1] = 0.0

    best_S = float('inf')
    best_params = None
    
    for start_idx in range(10):
        if start_idx > 0:
            np.random.seed(start_idx)
            x0_perturbed = x0.copy()
            for i in range(N):
                x0_perturbed[3*i] += np.random.normal(0, 0.5)
                x0_perturbed[3*i+1] += np.random.normal(0, 0.5)
                x0_perturbed[3*i+2] = np.random.uniform(0, np.pi)
            x0_perturbed[-2] = 4.0
            x0_perturbed[-1] = np.random.uniform(0, np.pi)
            current_x0 = x0_perturbed
        else:
            current_x0 = x0
            
        try:
            res = minimize(objective, current_x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    if best_params is None:
        inner = [[0.0, 0.0, 0.0] for _ in range(N)]
        return inner, (0.0, 0.0), 10.0, 0.0

    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([x, y, np.degrees(ang)])
    
    outer_center = (0.0, 0.0)
    outer_side = best_params[-2]
    outer_angle = best_params[-1]
    
    return inner, outer_center, outer_side, outer_angle
