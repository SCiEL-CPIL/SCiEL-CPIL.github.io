import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    N = 12
    sqrt3 = np.sqrt(3)
    min_dist = sqrt3
    
    angles = np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    def objective(vars):
        return vars[-1]
    
    def constraints(vars):
        s = vars[-1]
        cons = []
        # Separation
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: np.dot(v[2*i:2*i+2] - v[2*j:2*j+2], v[2*i:2*i+2] - v[2*j:2*j+2]) - min_dist**2
                })
        # Containment
        for i in range(N):
            for k in range(6):
                n = normals[k]
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, n=n: (sqrt3/2) * (v[-1] - 1) - np.dot(n, v[2*i:2*i+2])
                })
        return cons

    # Initial guess function
    def get_initial_guess(seed=0):
        rng = np.random.RandomState(seed)
        # Lattice based
        centers = np.zeros((N, 2))
        # 0
        # 1-6
        for i in range(6):
            angle = np.deg2rad(60 * i)
            centers[i+1] = min_dist * np.array([np.cos(angle), np.sin(angle)])
        # 7-11
        # Pick 5 from ring 2
        # Ring 2 angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
        # Lattice ring 2 has 12 spots.
        # Let's just use random perturbation of a valid lattice subset if possible, or just random.
        # Let's stick to the lattice subset for stability.
        ring2_angles = np.deg2rad([0, 60, 120, 180, 240])
        for i, angle in enumerate(ring2_angles):
            centers[i+7] = 2 * min_dist * np.array([np.cos(angle), np.sin(angle)])
            
        max_proj = 0
        for c in centers:
            for n in normals:
                proj = np.dot(n, c)
                if proj > max_proj:
                    max_proj = proj
        s_init = 1 + 2 * max_proj / sqrt3
        s_init *= 1.05 # 5% buffer
        
        # Add small noise to break symmetry
        centers += rng.normal(0, 0.1, centers.shape)
        
        return np.append(centers.flatten(), s_init)

    best_res = None
    for seed in range(5):
        x0 = get_initial_guess(seed)
        bounds = [(None, None)] * (2 * N) + [(1.0, None)]
        res = minimize(objective, x0, method='SLSQP', constraints=constraints, bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
        if res.success:
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        else:
            # Even if not success, check if it's better than current best
            if best_res is None or res.fun < best_res.fun:
                # Verify constraints roughly
                # If it's close, keep it
                if res.fun < (best_res.fun if best_res else 100):
                    best_res = res

    if best_res is None:
        # Fallback to seed-like solution
        centers = np.array([
            [0, 0],
            [-2.5, 0],
            [2.5, 0],
            [-1.25, 2.17],
            [1.25, 2.17],
            [-1.25, -2.17],
            [1.25, -2.17],
            [-3.75, 2.17],
            [3.75, 2.17],
            [-3.75, -2.17],
            [3.75, -2.17],
            [0, -4]
        ])
        s = 8.0
    else:
        s_opt = best_res.x[-1]
        centers_opt = best_res.x[:-1].reshape(N, 2)
        
    inner_hex_data = np.zeros((N, 3))
    for i in range(N):
        inner_hex_data[i, 0] = centers_opt[i, 0] if best_res else 0
        inner_hex_data[i, 1] = centers_opt[i, 1] if best_res else 0
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_opt if best_res else 8.0
