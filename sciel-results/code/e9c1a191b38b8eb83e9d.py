import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

N = 26

def construct():
    """
    Packs 26 circles in a unit square maximizing the sum of radii.
    Uses Differential Evolution for global search and L-BFGS-B for local refinement.
    """
    best_sum = -1
    best_result = None

    def unpack(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        return x, y, r

    def cost(v, penalty_weight=100.0):
        x, y, r = unpack(v)
        
        # Objective: Maximize sum of radii -> Minimize negative sum
        obj = -np.sum(r)
        
        # Penalty for overlaps
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        violations = np.maximum(0, r_sum - dist)
        np.fill_diagonal(violations, 0)
        
        penalty_overlap = np.sum(violations**2)
        
        # Penalty for boundaries
        pen_x_low = np.maximum(0, r - x)
        pen_x_high = np.maximum(0, x - (1 - r))
        pen_y_low = np.maximum(0, r - y)
        pen_y_high = np.maximum(0, y - (1 - r))
        
        penalty_boundary = np.sum(pen_x_low**2) + np.sum(pen_x_high**2) + \
                          np.sum(pen_y_low**2) + np.sum(pen_y_high**2)
        
        total_penalty = penalty_overlap + penalty_boundary
        
        return obj + penalty_weight * total_penalty

    def is_valid(x, y, r, tol=1e-6):
        # Check boundaries
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        
        # Check overlaps
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        np.fill_diagonal(r_sum, 0)
        
        if np.any(dist < r_sum - tol):
            return False
            
        return True

    # Bounds for Differential Evolution
    # x, y in [0, 1], r in [0.01, 0.3] (0.3 is a safe upper bound for individual circles in a pack of 26)
    bounds_de = []
    for _ in range(N):
        bounds_de.append((0.0, 1.0)) # x
        bounds_de.append((0.0, 1.0)) # y
        bounds_de.append((0.01, 0.3)) # r

    # 1. Global Search with Differential Evolution
    # DE is slow but finds global minima better.
    # We limit the number of iterations to keep within time limits.
    try:
        de_result = differential_evolution(
            cost, 
            bounds_de, 
            maxiter=100, 
            popsize=10, 
            tol=1e-5,
            seed=42,
            polish=False # We will polish manually
        )
        v_de = de_result.x
    except Exception:
        v_de = None

    # 2. Local Refinement with L-BFGS-B
    # We run L-BFGS-B starting from the DE result and several random starts.
    
    starts = []
    if v_de is not None:
        starts.append(v_de)
    
    # Add random starts
    for _ in range(5):
        v_rand = np.zeros(N * 3)
        v_rand[0::3] = np.random.uniform(0.1, 0.9, N)
        v_rand[1::3] = np.random.uniform(0.1, 0.9, N)
        v_rand[2::3] = np.random.uniform(0.05, 0.15, N)
        starts.append(v_rand)

    bounds_lbfgs = []
    for _ in range(N):
        bounds_lbfgs.append((0.0, 1.0)) # x
        bounds_lbfgs.append((0.0, 1.0)) # y
        bounds_lbfgs.append((0.0, 0.5)) # r

    for v0 in starts:
        try:
            res = minimize(cost, v0, method='L-BFGS-B', bounds=bounds_lbfgs, 
                           options={'maxiter': 2000, 'ftol': 1e-12})
            
            x_sol, y_sol, r_sol = unpack(res.x)
            
            if is_valid(x_sol, y_sol, r_sol):
                current_sum = np.sum(r_sol)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = np.column_stack((x_sol, y_sol, r_sol))
                    
        except Exception:
            pass

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
