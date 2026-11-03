"""Circle packing in unit square: pack 26 disjoint circles maximizing sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 26

# EVOLVE-BLOCK-START

def _compute_radii_fast(x, y, max_iter=20, tol=1e-9):
    """
    Compute maximal feasible radii for given centers x, y using vectorized fixed-point iteration.
    """
    n = len(x)
    # Boundary limits
    bounds = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Initialize radii to 0
    r = np.zeros(n)
    
    for _ in range(max_iter):
        r_old = r.copy()
        
        # Distance matrix
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        
        # Constraints: r_i <= d_ij - r_j for all j != i
        # We want min over j of (d_ij - r_j)
        # Subtract r_old from each column j
        constraints = d - r_old[None, :]
        
        # Min over neighbors (axis 1)
        min_constraints = np.min(constraints, axis=1)
        
        # Update r
        r = np.maximum(0.0, np.minimum(bounds, min_constraints))
        
        # Check convergence
        if np.max(np.abs(r - r_old)) < tol:
            break
            
    return r

def _sa_optimize(x_init, y_init, deadline, max_iters=100000):
    """
    Simulated Annealing to maximize sum of radii by optimizing centers.
    """
    x = x_init.copy()
    y = y_init.copy()
    r = _compute_radii_fast(x, y)
    current_score = np.sum(r)
    
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_score = current_score
    
    # SA parameters
    T = 0.005 # Initial temperature
    cooling = 0.9999 # Cooling rate
    step_scale = 0.05 # Initial step size
    
    np.random.seed(42)
    
    start_time = time.time()
    it = 0
    
    while it < max_iters and time.time() < deadline:
        # Pick a random circle
        i = np.random.randint(N)
        
        # Propose move
        dx = np.random.normal(0, step_scale)
        dy = np.random.normal(0, step_scale)
        
        x_new = x[i] + dx
        y_new = y[i] + dy
        
        # Basic bounds check to avoid invalid centers
        if x_new < -0.1 or x_new > 1.1 or y_new < -0.1 or y_new > 1.1:
            it += 1
            continue
            
        x_prop = x.copy()
        y_prop = y.copy()
        x_prop[i] = x_new
        y_prop[i] = y_new
        
        # Compute new radii
        r_prop = _compute_radii_fast(x_prop, y_prop)
        prop_score = np.sum(r_prop)
        
        delta = prop_score - current_score
        
        if delta > 0 or (T > 1e-9 and np.random.random() < np.exp(delta / T)):
            x = x_prop
            y = y_prop
            r = r_prop
            current_score = prop_score
            
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
        
        # Cooling
        T *= cooling
        # Decay step size
        step_scale *= 0.99999
        
        it += 1
        
    return best_x, best_y, best_r, best_score

def _polish_centers(x, y, deadline):
    """
    Polish the centers using derivative-free optimization (Powell) to maximize sum of radii.
    """
    def objective(params):
        x_p = params[:N]
        y_p = params[N:2*N]
        r_p = _compute_radii_fast(x_p, y_p)
        return -np.sum(r_p)
        
    x0 = np.concatenate([x, y])
    
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 500, 'maxfev': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
        x_opt = res.x[:N]
        y_opt = res.x[N:2*N]
        r_opt = _compute_radii_fast(x_opt, y_opt)
        return x_opt, y_opt, r_opt
    except Exception:
        return x, y, _compute_radii_fast(x, y)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize multiple random configurations
    n_starts = 4
    best_score = -np.inf
    best_result = None
    
    for k in range(n_starts):
        if time.time() > deadline - 10:
            break
            
        # Random initial positions
        np.random.seed(100 + k)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        
        # Run SA
        bx, by, br, score = _sa_optimize(x, y, deadline - 5)
        
        # Polish
        px, py, pr = _polish_centers(bx, by, deadline)
        p_score = np.sum(pr)
        
        if p_score > best_score:
            best_score = p_score
            best_result = np.column_stack([px, py, pr])
            
    # If no result, fallback to grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        x_grid = []
        y_grid = []
        r_grid = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x_grid.append((i + 0.5) / side)
                y_grid.append((j + 0.5) / side)
                r_grid.append(0.5 / side - 1e-4)
                count += 1
        best_result = np.column_stack([x_grid, y_grid, r_grid])
        
    return best_result

# EVOLVE-BLOCK-END
