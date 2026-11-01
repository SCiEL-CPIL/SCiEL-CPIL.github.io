import numpy as np
from scipy.optimize import minimize
import time

N = 21

def softmin(v, beta):
    v_min = np.min(v)
    exps = np.exp(-beta * (v - v_min))
    return v_min - (1.0 / beta) * np.log(np.sum(exps))

def objective(params, beta):
    n = N
    centers = params[:2*n].reshape(n, 2)
    delta = params[2*n]
    
    W = 1.0 + delta
    H = 1.0 - delta
    
    if W <= 0.01 or H <= 0.01:
        return 1e10
        
    radii = np.zeros(n)
    
    for i in range(n):
        xi, yi = centers[i]
        
        d_bound = np.array([xi, yi, W - xi, H - yi])
        
        diffs = centers - centers[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists[i] = np.inf
        
        d_circles = dists / 2.0
        all_dists = np.concatenate([d_bound, d_circles])
        
        r_i = softmin(all_dists, beta)
        radii[i] = r_i
        
    return -np.sum(radii)

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    best_score = -np.inf
    best_params = None
    
    # Generate diverse initializations
    inits = []
    
    # 1. Grid
    side = 5
    cells = [[(i+0.5)/side, (j+0.5)/side] for i in range(side) for j in range(side)]
    np.random.shuffle(cells)
    inits.append(np.array(cells[:N]))
    
    # 2. Hexagonal
    r_est = 0.12
    hex_init = []
    y = r_est
    row = 0
    while y < 1.0:
        x = r_est + (row % 2) * r_est * np.sqrt(3)/2
        while x < 1.0:
            hex_init.append([x, y])
            x += r_est * np.sqrt(3)
        y += r_est * 1.5
        row += 1
    if len(hex_init) >= N:
        inits.append(np.array(hex_init[:N]))
    else:
        pad = np.random.rand(N - len(hex_init), 2) * 0.8 + 0.1
        inits.append(np.vstack([np.array(hex_init), pad]))
        
    # 3. Random
    inits.append(np.random.rand(N, 2) * 0.8 + 0.1)
    
    # 4. Another random
    inits.append(np.random.rand(N, 2) * 0.8 + 0.1)
    
    betas = [5.0, 20.0, 50.0, 200.0, 1000.0]
    
    for idx, init_centers in enumerate(inits):
        if time.time() > deadline:
            break
            
        current_params = np.concatenate([init_centers.flatten(), [0.0]])
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            res = minimize(
                objective, 
                current_params, 
                args=(beta,), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                current_params = res.x
            elif res.success:
                current_params = res.x
            else:
                # If it failed, keep current_params but maybe nudge?
                pass

    if best_params is None:
        # Fallback
        return np.column_stack((inits[0], np.full(N, 0.1)))
        
    best_centers = best_params[:2*N].reshape(N, 2)
    best_delta = best_params[2*N]
    W = 1.0 + best_delta
    H = 1.0 - best_delta
    
    # Final polish: Calculate exact radii
    radii = np.zeros(N)
    for i in range(N):
        xi, yi = best_centers[i]
        d_bound = np.array([xi, yi, W - xi, H - yi])
        diffs = best_centers - best_centers[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists[i] = np.inf
        d_circles = dists / 2.0
        all_dists = np.concatenate([d_bound, d_circles])
        radii[i] = np.min(all_dists)
        
    return np.column_stack((best_centers, radii))
