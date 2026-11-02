import numpy as np
import scipy.optimize as opt
import time

N = 32

def hexagonal_init(n, seed=None):
    """Generate n points in a hexagonal lattice pattern within [0,1]^2."""
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
        
    # Hexagonal lattice spacing
    dx = 1.0 / 6.0
    dy = np.sqrt(3) / 12.0
    
    candidates = []
    for j in range(10):
        for i in range(10):
            x = i * dx + (0.5 * dx if j % 2 else 0)
            y = j * dy
            if -0.1 <= x <= 1.1 and -0.1 <= y <= 1.1:
                candidates.append([x, y])
                
    if len(candidates) < n:
        # Fallback to random if grid is too sparse (should not happen with dx=1/6)
        centers = np.random.rand(n, 2)
    else:
        # Randomly select n points
        indices = np.random.choice(len(candidates), n, replace=False)
        centers = np.array(candidates)[indices]
        
    # Clip centers to [0,1] to be safe, though they are generated within bounds mostly
    centers = np.clip(centers, 0, 1)
    
    # Calculate initial radii: half the minimum distance to other points or boundary
    radii = np.zeros(n)
    for i in range(n):
        min_dist = min(centers[i][0], 1-centers[i][0], centers[i][1], 1-centers[i][1])
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, dist / 2.0)
        radii[i] = min_dist * 0.9 # 10% safety margin
        
    return np.column_stack([centers, radii])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_val = -1
    best_solution = None
    
    # Try multiple restarts with different random seeds
    for attempt in range(15):
        if time.time() > deadline:
            break
            
        init = hexagonal_init(N, seed=100 + attempt)
        
        # Flatten variables: x0, y0, r0, x1, y1, r1, ...
        x0 = init.flatten()
        
        # Objective: minimize -sum(r_i)
        def objective(v):
            return -np.sum(v[2::3])
            
        # Constraints definition
        constraints = []
        
        # Boundary and Radius constraints for each circle
        for i in range(N):
            # x_i >= r_i
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i] - v[3*i+2]
            })
            # 1 - x_i >= r_i
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]
            })
            # y_i >= r_i
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]
            })
            # 1 - y_i >= r_i
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]
            })
            # r_i > 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i+2] - 1e-8
            })
            
        # Collision constraints
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2
                })
                
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((1e-8, 0.5)) # r
            
        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, 
                                constraints=constraints, options={'maxiter': 300, 'ftol': 1e-9})
            
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    sol = res.x.reshape(N, 3)
                    best_solution = sol
                    
        except Exception:
            continue
            
    if best_solution is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
