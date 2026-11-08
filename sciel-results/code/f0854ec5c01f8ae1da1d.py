import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -np.inf
    best_solution = None
    
    # Helper to extract solution from vector
    def vec_to_sol(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        return np.column_stack((x, y, r))
    
    # Helper to check validity and compute score
    def evaluate_solution(sol):
        # sol is (N, 3) array
        x = sol[:, 0]
        y = sol[:, 1]
        r = sol[:, 2]
        
        # Check boundaries
        if np.any(r <= 0):
            return -np.inf
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return -np.inf
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return -np.inf
            
        # Check overlaps
        # Compute distance matrix
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Required distance
        req = r[:, None] + r[None, :]
        
        # Check for violations
        # We only care about i < j
        i, j = np.triu_indices(N, k=1)
        if np.any(dist[i, j] < req[i, j] - 1e-9):
            return -np.inf
            
        return np.sum(r)

    # Objective function for optimizer
    # v: (3N,) vector
    # lambda: penalty weight
    def objective(v, lam):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Sum of radii (to maximize, so we minimize negative)
        cost = -np.sum(r)
        
        # Boundary penalties
        # x - r >= 0  => r - x <= 0
        # x + r <= 1  => x + r - 1 <= 0
        # y - r >= 0
        # y + r <= 1
        
        # Violations are max(0, violation)
        # We square them for smoothness
        
        # Left: r - x
        v_left = np.maximum(0, r - x)
        # Right: x + r - 1
        v_right = np.maximum(0, x + r - 1)
        # Bottom: r - y
        v_bottom = np.maximum(0, r - y)
        # Top: y + r - 1
        v_top = np.maximum(0, y + r - 1)
        
        boundary_penalty = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
        
        # Pairwise penalties
        # Vectorized distance calculation
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        req = r[:, None] + r[None, :]
        
        # Violation: req - dist
        viol = np.maximum(0, req - dist)
        
        # Sum over unique pairs (i < j)
        # Using triu_indices
        i, j = np.triu_indices(N, k=1)
        pair_penalty = np.sum(viol[i, j]**2)
        
        cost += lam * (boundary_penalty + pair_penalty)
        
        return cost

    # Initializations
    def get_initialization(seed):
        np.random.seed(seed)
        # Strategy: Grid with jitter
        # 5x5 grid = 25 circles. Add 1 more.
        # Or just random small circles?
        # Let's try a grid of 5x5, then add one in the center, then let it optimize.
        # But 5x5 grid has 25 slots. 
        # Let's use a 5x5 grid for the first 25, and place the 26th randomly.
        
        side = 5
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        
        x_coords = []
        y_coords = []
        r_coords = []
        
        for i in range(side):
            for j in range(side):
                x_coords.append(xs[i] + np.random.uniform(-0.05, 0.05))
                y_coords.append(ys[j] + np.random.uniform(-0.05, 0.05))
                # Initial radius slightly smaller than grid cell to avoid overlap
                r_coords.append(0.5 / side * 0.8)
                
        # Add 26th circle
        x_coords.append(np.random.uniform(0.2, 0.8))
        y_coords.append(np.random.uniform(0.2, 0.8))
        r_coords.append(0.5 / side * 0.8)
        
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        r_coords = np.array(r_coords)
        
        # Clip to valid range
        x_coords = np.clip(x_coords, 0.01, 0.99)
        y_coords = np.clip(y_coords, 0.01, 0.99)
        r_coords = np.clip(r_coords, 0.01, 0.49)
        
        v = np.zeros(3 * N)
        v[0::3] = x_coords
        v[1::3] = y_coords
        v[2::3] = r_coords
        
        return v

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-6, 0.5)) # r

    options = {
        'maxiter': 200,
        'ftol': 1e-12,
        'gtol': 1e-8
    }

    # Continuation method for lambda
    # Start with small lambda to move positions, then increase to enforce constraints
    
    for seed in range(20): # Max 20 restarts
        if time.time() > deadline:
            break
            
        v0 = get_initialization(seed)
        
        # Try a few lambda values
        for lam in [10.0, 100.0, 1000.0]:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(objective, v0, args=(lam,), method='L-BFGS-B', bounds=bounds, options=options)
                
                if res.success or res.fun < np.inf:
                    v_sol = res.x
                    sol = vec_to_sol(v_sol)
                    score = evaluate_solution(sol)
                    
                    if score > best_score:
                        best_score = score
                        best_solution = sol.copy()
                        
            except Exception:
                pass

    # If no valid solution found, return a trivial valid one (grid)
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N])
        
    return best_solution
