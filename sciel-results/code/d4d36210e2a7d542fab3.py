import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()

def get_hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + np.cos(a), cy + np.sin(a)])
    return np.array(verts)

def check_containment(verts, S):
    # Outer hexagon side S, angle 0 (flat-top)
    # Normals at 30, 90, 150, 210, 270, 330
    normals = []
    for k in range(6):
        a = np.deg2rad(30 + k * 60)
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)
    limit = S * np.sqrt(3) / 2
    # Project vertices onto normals
    # verts shape (6, 2)
    # dot product
    dists = verts @ normals.T # (6, 6)
    # Max projection for each normal
    max_proj = np.max(dists, axis=0)
    # Violation is max(0, max_proj - limit)
    viol = np.max(np.maximum(0, max_proj - limit))
    return viol

def cost(params):
    # params: x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, S
    S = params[-1]
    if S < 1: return 1e6
    
    pos = params[:-1].reshape(N, 3)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < np.sqrt(3):
                penalty += (np.sqrt(3) - d)**2
    
    # Containment penalty
    for i in range(N):
        verts = get_hex_vertices(pos[i,0], pos[i,1], pos[i,2])
        viol = check_containment(verts, S)
        penalty += viol**2
        
    return S + penalty

def construct():
    # Initial guess
    # Center
    x0, y0 = 0, 0
    # 6 neighbors
    neighbors = []
    for k in range(6):
        a = np.deg2rad(30 + k*60) # Flat-top neighbors?
        # Distance sqrt(3)
        nx = np.sqrt(3) * np.cos(a)
        ny = np.sqrt(3) * np.sin(a)
        neighbors.append([nx, ny, 0])
    
    # 4 more... put them randomly near the edges?
    # Or just random
    np.random.seed(42)
    extra = np.random.uniform(-2, 2, (4, 3))
    extra[:, 2] = 0 # angle
    
    init_pos = np.vstack([[x0, y0, 0]] + neighbors + extra.tolist())
    init_S = 4.0
    init_params = np.hstack([init_pos.flatten(), [init_S]])
    
    best_params = init_params
    best_cost = cost(init_params)
    
    start = time.time()
    # Try a few random restarts if time permits
    for restart in range(5):
        if time.time() - START_TIME > 100:
            break
            
        if restart > 0:
            # Perturb
            params = init_params + np.random.normal(0, 0.5, size=init_params.shape)
            params[-1] = max(3.0, params[-1])
        else:
            params = init_params
            
        res = minimize(cost, params, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
            
        if time.time() - START_TIME > 110:
            break
            
    # Extract best
    S = best_params[-1]
    pos = best_params[:-1].reshape(N, 3)
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i,0]), float(pos[i,1]), float(pos[i,2])])
        
    return inner, (0.0, 0.0), float(S), 0.0
