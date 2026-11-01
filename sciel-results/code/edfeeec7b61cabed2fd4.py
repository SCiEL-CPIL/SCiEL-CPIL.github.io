import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def get_hex_vertices(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])) # Pointy-top
    angles = base_angles + angle_rad
    return np.column_stack((np.cos(angles), np.sin(angles)))

def compute_L(params):
    normal_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) # Normals for pointy-top outer?
    # If outer hex is aligned with inner (pointy-top), its normals are at 0, 60...
    # Wait, if inner is pointy-top (vertex at 90), its flat sides are at 0, 60...
    # So if outer is also pointy-top, its flat sides are at 0, 60...
    # So normals are 0, 60...
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_proj = 0.0
    for i in range(N):
        x, y, ang = params[i]
        verts = get_hex_vertices(ang)
        verts = verts + np.array([x, y])
        projs = np.abs(verts @ normals.T)
        current_max = np.max(projs)
        if current_max > max_proj:
            max_proj = current_max
    return max_proj * 2.0 / np.sqrt(3.0)

def initial_guess():
    # Pointy-top hexes
    # Row 1 (y=0): 5 hexes
    # Row 2 (y=1.5): 3 hexes
    # Row 3 (y=-1.5): 3 hexes
    pos = []
    sq3 = np.sqrt(3)
    # Mid row
    for i in range(-2, 3):
        pos.append([i * sq3, 0.0, 30.0])
    # Top row
    # Offset by sq3/2
    for i in range(-1, 2):
        pos.append([i * sq3 + sq3/2, 1.5, 30.0])
    # Bottom row
    for i in range(-1, 2):
        pos.append([i * sq3 + sq3/2, -1.5, 30.0])
    return np.array(pos)

def construct():
    start_time = time.time()
    deadline = 145.0
    
    x0 = initial_guess()
    best_L = compute_L(x0)
    best_x = x0.copy()
    
    def objective(params):
        return compute_L(params)
        
    restarts = 20
    for r in range(restarts):
        if time.time() > deadline:
            break
        if r == 0:
            x_init = x0
        else:
            noise = np.random.normal(0, 0.5, x0.shape)
            x_init = x0 + noise
            
        try:
            res = minimize(objective, x_init, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000})
            if res.fun < best_L:
                best_L = res.fun
                best_x = res.x
        except:
            pass
            
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='Powell',
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
            if res.fun < best_L:
                best_L = res.fun
                best_x = res.x
        except:
            pass
            
    inner = []
    for i in range(N):
        x, y, ang = best_x[i]
        inner.append([float(x), float(y), float(ang)])
    return inner, (0.0, 0.0), float(best_L), 0.0
# EVOLVE-BLOCK-END
