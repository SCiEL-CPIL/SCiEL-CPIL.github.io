"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

N = 26
DEADLINE_MARGIN = 10.0  # seconds to spare before the 150s kill

def _feasible(arr, tol=1e-9):
    """Check if arrangement is feasible: inside [0,1]^2 and disjoint."""
    n = len(arr)
    # Boundary check
    for i in range(n):
        x, y, r = arr[i]
        if x < -tol or y < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    # Disjoint check
    for i in range(n):
        for j in range(i + 1, n):
            dx = arr[i, 0] - arr[j, 0]
            dy = arr[i, 1] - arr[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = arr[i, 2] + arr[j, 2]
            if dist_sq < (min_dist - tol) ** 2:
                return False
    return True

def _objective(x, n):
    """Objective: maximize sum of radii. We minimize negative sum."""
    radii = x[2 * n:]
    return -np.sum(radii)

def _constraints(x, n):
    """Return list of constraint dicts."""
    pos = x[:2 * n]
    radii = x[2 * n:]
    
    # Boundary constraints: x_i >= r_i, y_i >= r_i, x_i + r_i <= 1, y_i + r_i <= 1
    # Inequality form: g(x) >= 0 for SLSQP
    # x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+1] - x[2*n+i]})
    
    # Disjointness: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # Equivalent to: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*n+i] + x[2*n+j])**2
            })
    
    return cons

def _bounds(n):
    """Bounds for variables: positions in [0,1], radii in [0, 0.5]."""
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x_i
        bounds.append((0, 1))  # y_i
    for i in range(n):
        bounds.append((1e-6, 0.5))  # r_i
    return bounds

def _initial_grid(n):
    """Create a grid-based initial guess."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.001
            cells.append((x, y, r))
    # Pad if needed
    while len(cells) < n:
        cells.append((0.5, 0.5, 0.01))
    return np.array(cells[:n])

def _optimize_from(x0, n, time_limit=120):
    """Run SLSQP optimization from initial guess x0."""
    start_time = time.time()
    
    def time_check():
        return time.time() - start_time < time_limit
    
    # Build constraints and bounds
    cons = _constraints(x0, n)
    bounds = _bounds(n)
    
    # SLSQP optimization
    result = minimize(
        _objective,
        x0,
        args=(n,),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False}
    )
    
    if result.success or result.fun < _objective(x0, n):
        return result.x
    return x0

def _radius_growth_pass(x, n, time_limit=30):
    """Fix positions, maximize radii via SLSQP with hard constraints."""
    start_time = time.time()
    
    # Fix positions: only optimize radii
    # Variables: radii only (n variables)
    pos = x[:2*n]
    radii0 = x[2*n:]
    
    def obj_radii(r):
        return -np.sum(r)
    
    # Constraints: disjointness and boundary
    def make_cons(pos):
        cons = []
        for i in range(n):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda r, i=i: pos[2*i] - r[i]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda r, i=i: pos[2*i+1] - r[i]})
            # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda r, i=i: 1 - pos[2*i] - r[i]})
            # 1 - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda r, i=i: 1 - pos[2*i+1] - r[i]})
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[2*i] - pos[2*j]
                dy = pos[2*i+1] - pos[2*j+1]
                dist_sq = dx*dx + dy*dy
                cons.append({
                    'type': 'ineq',
                    'fun': lambda r, i=i, j=j, ds=dist_sq: ds - (r[i] + r[j])**2
                })
        return cons
    
    cons = make_cons(pos)
    bounds = [(1e-6, 0.5)] * n
    
    result = minimize(
        obj_radii,
        radii0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
    )
    
    if result.success:
        new_r = np.clip(result.x, 1e-6, 0.5)
        # Verify feasibility
        new_x = np.concatenate([pos, new_r])
        if _feasible(new_x):
            return new_x
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 150 - DEADLINE_MARGIN
    
    # Initial grid arrangement
    x0 = _initial_grid(N)
    x0 = x0.flatten()
    
    best_x = x0.copy()
    best_val = -_objective(x0, N)
    
    # Main optimization
    if time.time() < deadline - 60:
        x1 = _optimize_from(x0, N, time_limit=50)
        if _feasible(x1):
            val1 = -_objective(x1, N)
            if val1 > best_val:
                best_x = x1
                best_val = val1
    
    # Radius growth pass
    if time.time() < deadline - 30:
        x2 = _radius_growth_pass(best_x, N, time_limit=25)
        if _feasible(x2):
            val2 = -_objective(x2, N)
            if val2 > best_val:
                best_x = x2
                best_val = val2
    
    # Final polish
    if time.time() < deadline - 10:
        x3 = _optimize_from(best_x, N, time_limit=8)
        if _feasible(x3):
            val3 = -_objective(x3, N)
            if val3 > best_val:
                best_x = x3
                best_val = val3
    
    # Format result: (x, y, radius) per row
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_x[2*i]
        result[i, 1] = best_x[2*i+1]
        result[i, 2] = best_x[2*N+i]
    
    # Final feasibility check
    if not _feasible(result):
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([x, y, r])
        while len(cells) < N:
            cells.append([0.5, 0.5, 0.01])
        result = np.array(cells[:N])
    
    return result
