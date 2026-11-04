import numpy as np
from scipy.optimize import linprog

N = 26

def get_max_sum_radii(x, y):
    """
    Solve the LP to find maximum sum of radii for given positions.
    x, y: arrays of shape (N,)
    Returns: sum of radii, array of radii
    """
    # Variables: r[0] ... r[N-1]
    # Objective: maximize sum(r) -> minimize -sum(r)
    c = -np.ones(N)
    
    # Precompute distances
    diff_x = x[:, None] - x[None, :]
    diff_y = y[:, None] - y[None, :]
    D = np.sqrt(diff_x**2 + diff_y**2)
    
    # Wall distances
    wall_dist = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Build A_ub and b_ub
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pair_constraints = len(i_idx)
    
    A_ub = np.zeros((num_pair_constraints + N, N))
    b_ub = np.zeros(num_pair_constraints + N)
    
    # Pair constraints: r_i + r_j <= D[i, j]
    for k in range(num_pair_constraints):
        i = i_idx[k]
        j = j_idx[k]
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        b_ub[k] = D[i, j]
        
    # Wall constraints: r_i <= wall_dist[i]
    for i in range(N):
        row = num_pair_constraints + i
        A_ub[row, i] = 1.0
        b_ub[row] = wall_dist[i]
        
    bounds = [(0, None) for _ in range(N)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return -res.fun, res.x
    else:
        return 0.0, np.zeros(N)

def construct():
    rng = np.random.default_rng(42)
    
    # Random Initialization
    x = rng.random(N)
    y = rng.random(N)
    
    # Initial radii via LP
    best_score, _ = get_max_sum_radii(x, y)
    best_x = x.copy()
    best_y = y.copy()
    
    # Simulated Annealing
    T = 0.1
    cooling = 0.99
    min_T = 1e-4
    steps = 10000
    sigma = 0.1
    
    for step in range(steps):
        i = rng.integers(N)
        
        dx = rng.normal(0, sigma)
        dy = rng.normal(0, sigma)
        
        x_new = x.copy()
        y_new = y.copy()
        x_new[i] = np.clip(x[i] + dx, 0.0, 1.0)
        y_new[i] = np.clip(y[i] + dy, 0.0, 1.0)
        
        score_new, _ = get_max_sum_radii(x_new, y_new)
        
        delta = score_new - best_score
        
        if delta > 0:
            accept = True
        else:
            if T > min_T:
                accept = rng.random() < np.exp(delta / T)
            else:
                accept = False
        
        if accept:
            x = x_new
            y = y_new
            best_score = score_new
            best_x = x.copy()
            best_y = y.copy()
            
        T *= cooling
        
    # Final solve
    _, final_r = get_max_sum_radii(best_x, best_y)
    
    result = np.column_stack((best_x, best_y, final_r))
    return result
