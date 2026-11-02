import numpy as np
import scipy.optimize
import time

N = 12
DEADLINE = 145.0

def get_hex_vertices(cx, cy, r, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3.0 + angle_rad
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    return np.column_stack((xs, ys))

def point_in_hexagon(pt, hex_verts):
    x, y = pt
    for i in range(6):
        x1, y1 = hex_verts[i]
        x2, y2 = hex_verts[(i + 1) % 6]
        cross = (x1 - x) * (y2 - y) - (y1 - y) * (x2 - x)
        if cross < -1e-9:
            return False
    return True

def point_to_hex_dist(pt, verts):
    if point_in_hexagon(pt, verts):
        return 0.0
    min_dist = float('inf')
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        v = p2 - p1
        w = pt - p1
        c1 = np.dot(w, v)
        if c1 <= 0:
            dist = np.linalg.norm(pt - p1)
        else:
            c2 = np.dot(w, v) - np.dot(v, v)
            if c2 >= 0:
                dist = np.linalg.norm(pt - p2)
            else:
                b = c1 / np.dot(v, v)
                closest = p1 + b * v
                dist = np.linalg.norm(pt - closest)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def calculate_violation_smooth(state):
    S = state[0]
    X_out, Y_out, Phi_out = state[1], state[2], state[3]
    outer_verts = get_hex_vertices(X_out, Y_out, S, Phi_out)
    
    total_penalty = 0.0
    inner_verts_list = []
    
    for i in range(N):
        base_idx = 4 + i * 3
        xi, yi, ai = state[base_idx], state[base_idx + 1], state[base_idx + 2]
        verts = get_hex_vertices(xi, yi, 1.0, ai)
        inner_verts_list.append(verts)
        
        for v in verts:
            d = point_to_hex_dist(v, outer_verts)
            if d > 0:
                total_penalty += d * d * 50.0

    for i in range(N):
        for j in range(i + 1, N):
            verts_a = inner_verts_list[i]
            verts_b = inner_verts_list[j]
            
            axes_a = []
            for k in range(6):
                p1 = verts_a[k]
                p2 = verts_a[(k + 1) % 6]
                edge = p2 - p1
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9: n /= norm
                axes_a.append(n)
            
            axes_b = []
            for k in range(6):
                p1 = verts_b[k]
                p2 = verts_b[(k + 1) % 6]
                edge = p2 - p1
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9: n /= norm
                axes_b.append(n)
            
            for axis in axes_a + axes_b:
                proj_a = np.dot(verts_a, axis)
                min_a, max_a = np.min(proj_a), np.max(proj_a)
                proj_b = np.dot(verts_b, axis)
                min_b, max_b = np.min(proj_b), np.max(proj_b)
                
                overlap = min(max_a, max_b) - max(min_a, min_b)
                if overlap > 0:
                    total_penalty += overlap * overlap * 50.0

    return total_penalty

def objective(state):
    S = state[0]
    penalty = calculate_violation_smooth(state)
    if penalty > 1e-6:
        return S + penalty
    return S

def get_initial_guess():
    S = 8.5
    X_out, Y_out, Phi_out = 0.0, 0.0, 0.0
    inner = []
    for r in range(3):
        for c in range(4):
            x = c * np.sqrt(3)
            y = r * 1.5
            if r % 2 == 1:
                x += np.sqrt(3) / 2
            inner.append([x, y, 0.0])
            
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in inner:
        p[0] -= cx
        p[1] -= cy
        
    state = [S, X_out, Y_out, Phi_out]
    for p in inner:
        state.extend(p)
    return np.array(state)

def construct():
    start_time = time.time()
    
    # Initial guess
    state0 = get_initial_guess()
    
    best_state = state0
    best_score = objective(state0)
    
    # Global Search: Differential Evolution
    bounds = [(5.0, 15.0), (-5.0, 5.0), (-5.0, 5.0), (0.0, 360.0)]
    for _ in range(N):
        bounds.append((-10.0, 10.0))
        bounds.append((-10.0, 10.0))
        bounds.append((0.0, 360.0))
        
    # Run DE if time permits
    if time.time() - start_time < 10:
        try:
            res = scipy.optimize.differential_evolution(
                objective, 
                bounds, 
                maxiter=30, 
                popsize=12, 
                tol=1e-5, 
                seed=42,
                polish=False
            )
            if res.fun < best_score:
                best_score = res.fun
                best_state = res.x
        except Exception:
            pass

    # Local Search: Nelder-Mead
    # Multiple restarts
    states_to_polish = [best_state]
    for _ in range(3):
        perturbed = best_state + np.random.normal(0, 0.5, size=best_state.shape)
        states_to_polish.append(perturbed)
        
    time_limit = DEADLINE - 5.0
    
    for state in states_to_polish:
        if time.time() > time_limit:
            break
        try:
            res_nm = scipy.optimize.minimize(
                objective, 
                state, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_state = res_nm.x
        except Exception:
            pass

    S = best_state[0]
    X_out, Y_out, Phi_out = best_state[1], best_state[2], best_state[3]
    
    inner = []
    for i in range(N):
        base_idx = 4 + i * 3
        xi, yi, ai = best_state[base_idx], best_state[base_idx + 1], best_state[base_idx + 2]
        inner.append([xi, yi, ai])
        
    center = [X_out, Y_out]
    
    return inner, center, S, Phi_out
