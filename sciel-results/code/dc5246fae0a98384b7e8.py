import numpy as np
from scipy.optimize import minimize

N = 11
# Precompute triplets
triplets = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
# triplets shape (165, 3)
i_idx = triplets[:, 0]
j_idx = triplets[:, 1]
k_idx = triplets[:, 2]

def objective(params, alpha):
    v = params[:N]
    w = params[N:]
    
    # Penalty for v + w > 1
    excess = np.maximum(0, v + w - 1)
    penalty = 100.0 * np.sum(excess**2)
    
    # Convert to Cartesian
    x = v + 0.5 * w
    y = (np.sqrt(3) / 2) * w
    points = np.column_stack((x, y))
    
    # Vectorized Area Calculation
    # p_i = (x_i, y_i)
    x1 = x[i_idx]
    y1 = y[i_idx]
    x2 = x[j_idx]
    y2 = y[j_idx]
    x3 = x[k_idx]
    y3 = y[k_idx]
    
    areas = 0.5 * np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    
    # Soft min
    if np.all(areas == 0):
        return penalty
        
    y_vec = -alpha * areas
    max_y = np.max(y_vec)
    if max_y < -100:
        log_sum = max_y
    else:
        log_sum = max_y + np.log(np.sum(np.exp(y_vec - max_y)))
        
    soft_min_val = -1.0 / alpha * log_sum
    val = -soft_min_val
    
    return val + penalty

def construct():
    best_val = -np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    inits = []
    
    for _ in range(30):
        v_rand = rng.random(N)
        w_rand = rng.random(N) * (1 - v_rand)
        params = np.concatenate([v_rand, w_rand])
        inits.append(params)
        
    # Centered init
    v_c = np.full(N, 1/3) + rng.normal(0, 0.1, N)
    w_c = np.full(N, 1/3) + rng.normal(0, 0.1, N)
    v_c = np.clip(v_c, 0, 1)
    w_c = np.clip(w_c, 0, 1)
    mask = v_c + w_c > 1
    if np.any(mask):
        scale = 1.0 / (v_c[mask] + w_c[mask])
        v_c[mask] *= scale
        w_c[mask] *= scale
    inits.append(np.concatenate([v_c, w_c]))

    alphas = [20, 50, 100]
    
    for alpha in alphas:
        for init in inits:
            res = minimize(objective, init, args=(alpha,), method='L-BFGS-B', 
                           bounds=[(0, 1)] * (2*N), options={'maxiter': 500, 'ftol': 1e-8})
            if res.success:
                v_opt = res.x[:N]
                w_opt = res.x[N:]
                
                # Check validity
                if np.any(v_opt + w_opt > 1.0001) or np.any(v_opt < -1e-6) or np.any(w_opt < -1e-6):
                    continue
                    
                x_opt = v_opt + 0.5 * w_opt
                y_opt = (np.sqrt(3) / 2) * w_opt
                points_opt = np.column_stack((x_opt, y_opt))
                
                # Calculate true min area
                x1 = x_opt[i_idx]
                y1 = y_opt[i_idx]
                x2 = x_opt[j_idx]
                y2 = y_opt[j_idx]
                x3 = x_opt[k_idx]
                y3 = y_opt[k_idx]
                min_area = np.min(0.5 * np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))
                
                if min_area > best_val:
                    best_val = min_area
                    best_points = points_opt
                    
    if best_points is None:
        v_rand = np.random.random(N)
        w_rand = np.random.random(N) * (1 - v_rand)
        x = v_rand + 0.5 * w_rand
        y = (np.sqrt(3) / 2) * w_rand
        best_points = np.column_stack((x, y))
        
    return best_points
