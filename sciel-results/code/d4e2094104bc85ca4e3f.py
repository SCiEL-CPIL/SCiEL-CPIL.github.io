"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
START_TIME = time.time()
TIME_LIMIT = 115  # safety margin before 120s kill

def _is_feasible(params, tol=1e-9):
    """Check if all circles are disjoint and inside the unit square."""
    n = len(params)
    # Check boundaries
    for i in range(n):
        x, y, r = params[i]
        if x - r < -tol or y - r < -tol or x + r > 1.0 + tol or y + r > 1.0 + tol:
            return False
    # Check pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            dx = params[i,0] - params[j,0]
            dy = params[i,1] - params[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < params[i,2] + params[j,2] - tol:
                return False
    return True

def _objective(params):
    """Negative sum of radii (to minimize)."""
    return -np.sum(params[:, 2])

def _make_constraints(n):
    """Create constraint functions for scipy minimize."""
    constraints = []
    # Boundary constraints: x >= r, y >= r, x <= 1-r, y <= 1-r
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i,0] - p[i,2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i,1] - p[i,2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[i,0] - p[i,2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[i,1] - p[i,2]})
    # Pairwise distance constraints: dist >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: 
                np.sqrt((p[i,0]-p[j,0])**2 + (p[i,1]-p[j,1])**2) - p[i,2] - p[j,2]})
    return constraints

def _random_init(n, seed=42):
    """Generate a random feasible initial configuration."""
    rng = np.random.RandomState(seed)
    # Start with small circles in random positions
    params = np.zeros((n, 3))
    for i in range(n):
        params[i, 0] = rng.uniform(0.1, 0.9)
        params[i, 1] = rng.uniform(0.1, 0.9)
        params[i, 2] = 0.02
    return params

def _grid_init(n):
    """Grid-based initialization for better starting point."""
    side = int(np.ceil(np.sqrt(n)))
    params = np.zeros((n, 3))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            params[idx, 0] = (i + 0.5) / side
            params[idx, 1] = (j + 0.5) / side
            params[idx, 2] = 0.5 / side * 0.9
            idx += 1
    return params

def _optimize_from(params, constraints, maxiter=500):
    """Run SLSQP optimization from given starting point."""
    n = len(params)
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0.001, 0.5))  # r
    
    result = minimize(
        _objective,
        params,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-10}
    )
    
    if result.success or _is_feasible(result.x):
        return result.x
    return params

def _polish(params, constraints):
    """High-precision polish pass with tighter tolerance."""
    n = len(params)
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0.001, 0.5))
    
    result = minimize(
        _objective,
        params,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    
    if result.success or _is_feasible(result.x):
        return result.x
    return params

def construct():
    """Construct 26 circles in unit square maximizing sum of radii."""
    global START_TIME
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        # Fallback to grid
        return _grid_init(N)
    
    n = N
    constraints = _make_constraints(n)
    
    # Best solution found so far
    best_params = _grid_init(n)
    best_score = _objective(best_params)
    
    # Try multiple random initializations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Vary the random seed for diversity
        seed = 100 + restart * 7
        params = _random_init(n, seed=seed)
        
        # First optimization pass
        params = _optimize_from(params, constraints, maxiter=300)
        
        # Polish pass
        params = _polish(params, constraints)
        
        # Check feasibility
        if _is_feasible(params):
            score = _objective(params)
            if score < best_score:  # Remember: objective is negative sum
                best_params = params.copy()
                best_score = score
    
    # Final polish on the best
    best_params = _polish(best_params, constraints)
    if not _is_feasible(best_params):
        best_params = _grid_init(n)
    
    return best_params
