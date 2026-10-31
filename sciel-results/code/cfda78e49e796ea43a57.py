"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Better initialization: hexagonal-like packing for first row, then grid
    # Use a structured packing that's denser than square grid
    
    # Start with square grid but with slight perturbation
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    
    while len(cells) < N:
        cells.append([0.5, 0.5, 0.01])
    
    x0 = np.array(cells[:N])
    n = len(x0)
    
    def objective(v):
        return -np.sum(v[2::3])
    
    def constraints_func(v):
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        bc = np.concatenate([xs - rs, 1 - xs - rs, ys - rs, 1 - ys - rs])
        
        nc = np.zeros(n * (n-1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dr = rs[i] + rs[j]
                nc[idx] = dx*dx + dy*dy - dr*dr
                idx += 1
        
        return np.concatenate([bc, nc])
    
    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * n
    
    v0 = np.zeros(3 * n)
    v0[0::3] = x0[:, 0]
    v0[1::3] = x0[:, 1]
    v0[2::3] = x0[:, 2]
    
    constraints = {'type': 'ineq', 'fun': constraints_func}
    
    best_result = x0.copy()
    best_val = -np.sum(x0[:, 2])
    
    # SLSQP
    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 150, 'ftol': 1e-7})
        if res.fun < best_val:
            v_opt = res.x
            result = np.zeros((n, 3))
            result[:, 0] = v_opt[0::3]
            result[:, 1] = v_opt[1::3]
            result[:, 2] = v_opt[2::3]
            result[:, 2] = np.clip(result[:, 2], 1e-6, 0.5)
            best_result = result
            best_val = res.fun
    except Exception:
        pass
    
    return best_result

# EVOLVE-BLOCK-END
