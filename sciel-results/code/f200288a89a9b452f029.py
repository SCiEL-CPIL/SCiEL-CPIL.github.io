import numpy as np
import time
from scipy.optimize import minimize

N = 12

def get_hex_vertices(cx, cy, angle=0.0):
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

def calc_S_and_penalty(inner_pos, cx, cy, phi):
    min_dist = np.sqrt(3.0)
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(inner_pos[i] - inner_pos[j])
            if d < min_dist:
                penalty += (min_dist - d)**2
                
    all_vertices = np.vstack([get_hex_vertices(inner_pos[i, 0], inner_pos[i, 1], 0.0) for i in range(N)])
    normal_angles = phi + np.pi/6 + np.arange(6) * np.pi / 3
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    dots = (all_vertices - np.array([cx, cy])) @ normals.T
    max_proj = np.max(np.abs(dots))
    S = (2.0 / np.sqrt(3.0)) * max_proj
    return S, penalty

def objective(params):
    inner_pos = params[:N*2].reshape(N, 2)
    cx, cy, phi = params[N*2], params[N*2+1], params[N*2+2]
    S, penalty = calc_S_and_penalty(inner_pos, cx, cy, phi)
    return S + 1000.0 * penalty

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial guess: 3x4 grid
    xs = [-np.sqrt(3), 0.0, np.sqrt(3)]
    ys = [-2.25, -0.75, 0.75, 2.25]
    
    init_params = []
    for y in ys:
        for x in xs:
            init_params.extend([x, y])
    init_params.extend([0.0, 0.0, 0.0])
    
    best_params = np.array(init_params)
    best_score = objective(best_params)
    
    # Try a few different initializations
    inits = [best_params]
    
    # Random perturbations
    np.random.seed(42)
    for _ in range(5):
        p = np.array(init_params)
        p[:N*2] += np.random.normal(0, 0.5, N*2)
        p[N*2+2] = np.random.uniform(0, np.pi/3) # Random outer angle
        inits.append(p)
        
    for init in inits:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Final polish with tighter tolerance
    if time.time() < deadline - 2.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    inner_pos = best_params[:N*2].reshape(N, 2)
    cx, cy, phi = best_params[N*2], best_params[N*2+1], best_params[N*2+2]
    
    # Calculate final S
    S, penalty = calc_S_and_penalty(inner_pos, cx, cy, phi)
    
    phi_deg = np.degrees(phi) % 360
    inner = [[float(inner_pos[i, 0]), float(inner_pos[i, 1]), 0.0] for i in range(N)]
    
    return inner, (float(cx), float(cy)), float(S), float(phi_deg)
# EVOLVE-BLOCK-END
