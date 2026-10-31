import numpy as np
import time
from scipy.optimize import minimize

N = 11

HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2]
])

NORMALS = np.array([
    [np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)
])

def objective(params, penalty_weight=1000.0):
    pos = params.reshape(N, 2)
    
    # Calculate L
    points = pos[:, np.newaxis, :] + HEX_VERTS[np.newaxis, :, :]
    dots = np.einsum('nij,kj->nik', points, NORMALS)
    max_proj = np.max(dots)
    L = max_proj / (np.sqrt(3) / 2)
    
    # Calculate Overlap Penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < np.sqrt(3):
                penalty += (np.sqrt(3) - d) ** 2
                
    return L + penalty_weight * penalty

def construct():
    start_time = time.time()
    max_time = 110.0 # Safety margin
    
    # Initial guess: Hexagonal cluster
    axial = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2)
    ]
    
    p0 = np.zeros((N, 2))
    for i, (q, r) in enumerate(axial):
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        p0[i] = [x, y]
        
    best_L = objective(p0)
    best_pos = p0.copy()
    
    # Optimization
    # Use Nelder-Mead or Powell as they don't require gradients and handle non-smoothness better
    # SLSQP is good too.
    
    try:
        res = minimize(objective, p0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_L:
            best_L = res.fun
            best_pos = res.x.reshape(N, 2)
    except Exception:
        pass
        
    # Try a few random restarts if time allows
    restarts = 0
    while time.time() - start_time < max_time:
        # Generate random perturbation of initial guess
        p_rand = p0 + np.random.normal(0, 0.5, (N, 2))
        try:
            res = minimize(objective, p_rand, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_L:
                best_L = res.fun
                best_pos = res.x.reshape(N, 2)
        except Exception:
            pass
        restarts += 1
        if restarts > 20: break # Prevent infinite loop if minimize is fast
        
    inner = []
    for i in range(N):
        x, y = best_pos[i]
        inner.append([x, y, 0.0])
        
    center = (0.0, 0.0)
    side = best_L
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
