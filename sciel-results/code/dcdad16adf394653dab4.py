import numpy as np
from scipy.optimize import minimize
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Initialization: Square grid with noise
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x0 = grid[:N].copy()
    # Add small noise to break symmetry
    np.random.seed(42)
    x0 += np.random.normal(0, 0.001, x0.shape)
    x0 = np.clip(x0, -0.49, 0.49) # Keep inside
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def get_dists(x):
        x = x.reshape(-1, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        b_dists = np.min(np.stack([
            x[:, 0] + 0.5,
            0.5 - x[:, 0],
            x[:, 1] + 0.5,
            0.5 - x[:, 1]
        ], axis=1), axis=1)
        return dists, b_dists

    def objective(x, beta):
        dists, b_dists = get_dists(x)
        iu = np.triu_indices_from(dists, k=1)
        p_dists = dists[iu] / 2.0
        all_dists = np.concatenate([p_dists, b_dists])
        
        # Clip to avoid numerical issues if slightly outside
        all_dists = np.maximum(all_dists, 0)
        
        min_val = np.min(all_dists)
        shifted = all_dists - min_val
        exps = np.exp(-beta * shifted)
        sum_exps = np.sum(exps)
        
        if sum_exps == 0:
            return -min_val
        
        softmin_val = min_val + (1.0 / beta) * np.log(sum_exps)
        return -softmin_val

    best_x = x0
    best_score = -objective(x0, 10)
    
    beta = 10.0
    while time.time() < deadline:
        # Optimize
        try:
            res = minimize(
                lambda x: objective(x, beta),
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            break
            
        if res.success or res.fun < -best_score:
            if res.fun < -best_score:
                best_score = -res.fun
                best_x = res.x
        
        # Increase beta to sharpen the minimum
        beta *= 2.0
        if beta > 1000:
            break
            
    return best_x.reshape(N, 2)
