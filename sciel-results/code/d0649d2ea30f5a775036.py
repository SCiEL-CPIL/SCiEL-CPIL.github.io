import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial guess: 5x5 grid, remove 4 corners
    # This creates a shape that is 5 wide, 5 high, but with corners missing.
    # This allows the edge circles to potentially move outwards if r increases.
    base_coords = []
    for i in range(5):
        for j in range(5):
            # Remove corners
            if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                continue
            base_coords.append([0.1 + 0.2*i, 0.1 + 0.2*j])
    
    x0_base = np.array(base_coords)
    r0 = 0.1
    
    # We will try to optimize.
    # Variables: x1...x21, y1...y21, r
    x0_flat = np.concatenate([x0_base.flatten(), [r0]])
    
    bounds = [(0, 1) for _ in range(2*N)] + [(0, 1)]
    
    def objective(x):
        return -x[2*N]
        
    def constraints_func(x):
        cons = []
        r = x[2*N]
        
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i] - x[2*N]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[N+i] - x[2*N]})
            
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i, j=j: (x[i]-x[j])**2 + (x[N+i]-x[N+j])**2 - 4*x[2*N]**2
                })
        return cons

    best_x = x0_flat
    best_val = objective(x0_flat)
    
    # Try a few random perturbations to escape local minima
    # But keep it fast
    rng = np.random.default_rng(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
            
        # Perturb initial guess
        if attempt == 0:
            x_curr = x0_flat
        else:
            # Small random perturbation to positions, keep r same
            x_curr = x0_flat.copy()
            noise = rng.normal(0, 0.01, size=2*N)
            x_curr[:2*N] += noise
            # Clamp to valid range roughly
            x_curr[:2*N] = np.clip(x_curr[:2*N], 0.05, 0.95)
            
        try:
            res = minimize(
                objective, 
                x_curr, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints_func, 
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Extract best result
    x_opt = best_x
    r_opt = x_opt[2*N]
    x_coords = x_opt[:N]
    y_coords = x_opt[N:2*N]
    
    # Ensure validity: clip r to be small enough if numerical errors occurred
    # Check min distance
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x_coords[i]-x_coords[j])**2 + (y_coords[i]-y_coords[j])**2)
            if d < min_dist:
                min_dist = d
    # Check boundary
    min_boundary = np.inf
    for i in range(N):
        min_boundary = min(min_boundary, x_coords[i], 1-x_coords[i], y_coords[i], 1-y_coords[i])
    
    # The radius must be <= min_dist/2 and <= min_boundary
    valid_r = min(r_opt, min_dist/2 - 1e-6, min_boundary - 1e-6)
    
    if valid_r <= 0:
        # Fallback to seed-like solution
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    result = np.column_stack((x_coords, y_coords, np.full(N, valid_r)))
    return result
