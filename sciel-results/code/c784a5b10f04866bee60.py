"""Circle packing in unit square - NLP approach with SLSQP and multi-start."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    # Number of variables: 2*N (centers) + N (radii) = 3*N
    n_vars = 3 * N
    
    def unpack(v):
        centers = v[:2*N].reshape(N, 2)
        radii = v[2*N:]
        return centers, radii
    
    def objective(v):
        # Maximize sum of radii -> minimize negative sum of radii
        _, radii = unpack(v)
        return -np.sum(radii)
    
    def constraints():
        # Build constraint functions
        cons = []
        
        # Boundary constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        # For each circle i:
        #   x_i - r_i >= 0  ->  v[2*i] - v[2*N+i] >= 0
        #   x_i + r_i <= 1  ->  1 - v[2*i] - v[2*N+i] >= 0
        #   y_i - r_i >= 0  ->  v[2*i+1] - v[2*N+i] >= 0
        #   y_i + r_i <= 1  ->  1 - v[2*i+1] - v[2*N+i] >= 0
        
        # Pairwise non-overlap: ||c_i - c_j|| >= r_i + r_j
        # => (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
        
        # Use a single constraint function returning all inequalities for efficiency
        def all_constraints(v):
            centers, radii = unpack(v)
            vals = []
            
            # Boundary constraints
            for i in range(N):
                xi, yi = centers[i]
                ri = radii[i]
                vals.append(xi - ri)      # left
                vals.append(1.0 - xi - ri)  # right
                vals.append(yi - ri)      # bottom
                vals.append(1.0 - yi - ri)  # top
            
            # Pairwise constraints
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist_sq = dx*dx + dy*dy
                    r_sum = radii[i] + radii[j]
                    vals.append(dist_sq - r_sum*r_sum)
            
            return np.array(vals)
        
        return [{'type': 'ineq', 'fun': all_constraints}]
    
    cons = constraints()
    
    # Bounds: centers in [0.01, 0.99], radii in [0.001, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
    for i in range(N):
        bounds.append((0.001, 0.5))  # radius
    
    # Multi-start with different initial configurations
    best_result = None
    best_obj = np.inf
    
    # Generate diverse initial points
    np.random.seed(42)
    
    # Initial configuration 1: grid-like
    def init_grid():
        side = int(np.ceil(np.sqrt(N)))
        v = np.zeros(3*N)
        for idx in range(N):
            i = idx % side
            j = idx // side
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            v[2*idx] = x
            v[2*idx+1] = y
            v[2*N+idx] = r
        return v
    
    # Initial configuration 2: random with small radii
    def init_random(seed_val):
        rng = np.random.RandomState(seed_val)
        v = np.zeros(3*N)
        for i in range(N):
            v[2*i] = 0.15 + 0.7 * rng.rand()
            v[2*i+1] = 0.15 + 0.7 * rng.rand()
            v[2*N+i] = 0.05 + 0.1 * rng.rand()
        return v
    
    # Initial configuration 3: hexagonal-like packing
    def init_hex():
        v = np.zeros(3*N)
        # Hexagonal packing pattern
        rows = int(np.ceil(np.sqrt(N / 1.5)))
        r_est = 0.5 / (rows * 1.5)
        idx = 0
        for row in range(rows):
            cols = int(np.ceil(np.sqrt(N) + (row % 2) * 0.5))
            for col in range(cols):
                if idx >= N:
                    break
                x = 0.5/cols + col * (1.0/cols) + (row % 2) * (0.5/cols)
                y = 0.5/(rows*1.5) + row * (1.0/(rows*1.5))
                x = min(max(x, 0.05), 0.95)
                y = min(max(y, 0.05), 0.95)
                v[2*idx] = x
                v[2*idx+1] = y
                v[2*N+idx] = 0.08 + 0.05 * (1 - row/rows)
                idx += 1
        return v
    
    initial_configs = [
        init_grid(),
        init_hex(),
        init_random(1),
        init_random(2),
        init_random(3),
        init_random(4),
        init_random(5),
    ]
    
    for start_idx, v0 in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
        
        try:
            result = minimize(
                objective,
                v0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.success or result.fun < best_obj:
                # Validate the result
                centers, radii = unpack(result.x)
                valid = True
                
                # Check boundaries
                for i in range(N):
                    if centers[i, 0] < -1e-6 or centers[i, 0] > 1 + 1e-6:
                        valid = False
                    if centers[i, 1] < -1e-6 or centers[i, 1] > 1 + 1e-6:
                        valid = False
                    if radii[i] < -1e-6:
                        valid = False
                
                # Check pairwise
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(centers[i] - centers[j])
                        if d < radii[i] + radii[j] - 1e-6:
                            valid = False
                
                if valid and result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result.x.copy()
                    
        except Exception:
            continue
    
    # If no valid result from optimization, fall back to a safe grid
    if best_result is None:
        v0 = init_grid()
        centers, radii = unpack(v0)
        # Clip to ensure validity
        for i in range(N):
            centers[i] = np.clip(centers[i], radii[i], 1 - radii[i])
        best_result = np.concatenate([centers.flatten(), radii])
    
    centers, radii = unpack(best_result)
    result_array = np.column_stack([centers[:, 0], centers[:, 1], radii])
    
    # Final validation and clipping
    for i in range(N):
        x, y, r = result_array[i]
        r = max(r, 0.001)
        x = min(max(x, r), 1 - r)
        y = min(max(y, r), 1 - r)
        result_array[i] = [x, y, r]
    
    return result_array

def construct():
    return _construct()
# EVOLVE-BLOCK-END
