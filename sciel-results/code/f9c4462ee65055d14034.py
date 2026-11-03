import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 110.0

def hex_vertices(cx, cy, r, angle_deg):
    angles = np.deg2rad(angle_deg + np.arange(0, 360, 60))
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    normals1 = np.column_stack((-edges1[:, 1], edges1[:, 0]))
    normals2 = np.column_stack((-edges2[:, 1], edges2[:, 0]))
    
    norms1 = np.linalg.norm(normals1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(normals2, axis=1, keepdims=True)
    
    norms1[norms1 == 0] = 1
    norms2[norms2 == 0] = 1
    
    normals1 = normals1 / norms1
    normals2 = normals2 / norms2
    
    for normals in [normals1, normals2]:
        for normal in normals:
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 or max2 < min1:
                return False
                
    return True

def point_in_convex_polygon(p, verts):
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        edge = p2 - p1
        vec = p - p1
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        if cross < -1e-9:
            return False
    return True

def objective(params):
    start_time = time.time()
    if start_time > DEADLINE:
        return 1e9
        
    inner_params = params[:36].reshape(12, 3)
    L = params[36]
    outer_angle = params[37]
    
    if L < 0.5:
        return 1e9
        
    outer_verts = hex_vertices(0, 0, L, outer_angle)
    
    penalty = 0.0
    inner_verts_list = []
    
    for i in range(12):
        x, y, ang = inner_params[i]
        verts = hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
        
        for v in verts:
            if not point_in_convex_polygon(v, outer_verts):
                penalty += 10.0 + np.linalg.norm(v)
                
    for i in range(12):
        for j in range(i + 1, 12):
            if sat_overlap(inner_verts_list[i], inner_verts_list[j]):
                penalty += 10.0

    return L + penalty

def generate_initial_guesses():
    guesses = []
    
    # Guess 1: Grid-like packing
    points = []
    xs = np.array([-1.5, -0.5, 0.5, 1.5]) * np.sqrt(3)
    ys = np.array([0, np.sqrt(3)/2, np.sqrt(3)])
    
    for y in ys:
        for x in xs:
            points.append([x, y, 0.0])
            
    max_r = np.max(np.sqrt(np.array(points)[:, 0]**2 + np.array(points)[:, 1]**2))
    L_init = max_r + 2.0
    
    guess1 = np.array(points).flatten()
    guess1 = np.append(guess1, [L_init, 0.0])
    guesses.append(guess1)
    
    # Guess 2: Perturbed
    guess2 = guess1.copy()
    guess2 += np.random.normal(0, 0.1, len(guess2))
    guesses.append(guess2)
    
    # Guess 3: Random
    guess3 = np.random.uniform(-3, 3, 36)
    guess3 = np.append(guess3, [6.0, 0.0])
    guesses.append(guess3)
    
    return guesses

def construct():
    start_time = time.time()
    best_params = None
    best_cost = np.inf
    
    guesses = generate_initial_guesses()
    
    for guess in guesses:
        if time.time() > DEADLINE - 10:
            break
            
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-3, 'fatol': 1e-3, 'disp': False})
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    if best_params is None:
        inner = np.zeros((12, 3))
        for i in range(12):
            inner[i, 0] = (i % 4) * 2.0 - 3.0
            inner[i, 1] = (i // 4) * 2.0 - 1.0
        return inner, (0.0, 0.0), 10.0, 0.0

    inner = best_params[:36].reshape(12, 3)
    outer_side = best_params[36]
    outer_angle = best_params[37]
    
    if outer_side < 1.0:
        outer_side = 1.0
        
    return inner, (0.0, 0.0), outer_side, outer_angle
