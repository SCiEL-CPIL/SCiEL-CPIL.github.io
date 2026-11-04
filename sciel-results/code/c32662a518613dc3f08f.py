"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
import itertools


def _validate(arr):
    """Check that circles are inside unit square and non-overlapping."""
    if arr is None or len(arr) != N:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    # Inside unit square
    if np.any(r < -1e-9) or np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7) or \
       np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    # Non-overlapping: distance >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True


def _score(arr):
    """Sum of radii, 0 if invalid."""
    if not _validate(arr):
        return 0.0
    return float(np.sum(arr[:, 2]))


def _initial_grid():
    """Create a reasonable initial grid of N circles."""
    # Use a 5x5 grid, pick 25 best + 1 extra, or just 5x5=25 and add 1
    # Actually let's use a smarter initialization
    # Start with a 5x5 grid of equal circles, then add one more
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.95])
    # We need 26 circles. Let's use a 5x5 grid (25 circles) and add one more
    # Or better: use a layout that naturally has 26
    # Let's try: 5 columns x 5 rows = 25, plus 1 in the center-ish
    # Actually, let's just use a 5x5 grid and shrink slightly, then add one more
    # For simplicity, let's use a different approach:
    # Place circles in a pattern that fits 26
    
    # Better initialization: use a hexagonal-like packing
    # Or just use a 5x5 grid with slightly smaller circles and add one
    
    # Let's use a simple approach: 5x5 grid, take all 25, and add one more
    # by splitting one cell
    base_r = 0.5 / side * 0.9
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append([(i + 0.5) / side, (j + 0.5) / side, base_r])
    
    # Add a 26th circle: place it in a gap
    # For a 5x5 grid, we can add one more by using a 5x6 or 6x5 arrangement
    # Let's just use a 6x5 grid and take 26
    side_x = 6
    side_y = 5
    base_r = 0.5 / max(side_x, side_y) * 0.9
    positions = []
    for i in range(side_x):
        for j in range(side_y):
            positions.append([(i + 0.5) / side_x, (j + 0.5) / side_y, base_r])
    
    return np.array(positions[:N])


def _optimize_from_start(start, deadline):
    """Run optimization from a given starting point."""
    n = N
    # Variables: x_i, y_i, r_i for i=0..n-1 -> 3n variables
    # Constraints:
    # 1. x_i >= r_i, x_i + r_i <= 1, y_i >= r_i, y_i + r_i <= 1
    # 2. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for i < j
    # Objective: maximize sum(r_i)
    
    def objective(vars):
        r = vars[2*n:3*n]
        return -np.sum(r)
    
    # Inequality constraints (scipy wants g(vars) >= 0)
    # For scipy minimize with SLSQP, we use 'ineq' constraints: f(vars) >= 0
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i]})
        # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2*n+i]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i]})
        # 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[n+i] - v[2*n+i]})
    
    # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[i]-v[j])**2 + (v[n+i]-v[n+j])**2 - (v[2*n+i]+v[2*n+j])**2
            })
    
    # Bounds: r_i >= 0, x_i, y_i in [0, 1]
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x_i
    for i in range(n):
        bounds.append((0, 1))  # y_i
    for i in range(n):
        bounds.append((0, 0.5))  # r_i
    
    x0 = start.flatten()
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
        if result.success and _validate(result.x.reshape(n, 3)):
            return result.x.reshape(n, 3)
    except Exception:
        pass
    
    return None


def _local_search(arr, deadline):
    """Perform local search by trying to move circles to improve sum of radii."""
    if arr is None:
        return None
    
    current_score = _score(arr)
    if current_score <= 0:
        return None
    
    best = arr.copy()
    best_score = current_score
    
    # Try to increase radii and adjust positions
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try to increase radius of circle i slightly
            new_arr = best.copy()
            delta_r = 0.001
            new_arr[i, 2] += delta_r
            
            # Check if still valid
            if _validate(new_arr):
                best = new_arr
                best_score += delta_r
                improved = True
            else:
                # Try to move circle i to make room
                for dx in [-0.01, 0.01]:
                    for dy in [-0.01, 0.01]:
                        test_arr = best.copy()
                        test_arr[i, 0] += dx
                        test_arr[i, 1] += dy
                        test_arr[i, 2] += delta_r
                        if _validate(test_arr):
                            best = test_arr
                            best_score += delta_r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # Try more aggressive moves
            for i in range(N):
                if time.time() > deadline:
                    break
                for scale in [0.9, 0.95, 1.05, 1.1]:
                    new_arr = best.copy()
                    new_arr[i, 2] *= scale
                    if _validate(new_arr):
                        new_score = _score(new_arr)
                        if new_score > best_score:
                            best = new_arr
                            best_score = new_score
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_result = None
    best_score = 0.0
    
    # Start with grid initialization
    initial = _initial_grid()
    if _validate(initial):
        best_result = initial
        best_score = _score(initial)
    
    # Try optimization from the initial grid
    if time.time() < deadline - 30:
        opt_result = _optimize_from_start(initial, deadline)
        if opt_result is not None:
            opt_score = _score(opt_result)
            if opt_score > best_score:
                best_result = opt_result
                best_score = opt_score
    
    # Try local search
    if time.time() < deadline - 10 and best_result is not None:
        ls_result = _local_search(best_result, deadline)
        if ls_result is not None:
            ls_score = _score(ls_result)
            if ls_score > best_score:
                best_result = ls_result
                best_score = ls_score
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Random initial positions
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.02, 0.1, N)
        random_start = np.column_stack([x, y, r])
        
        # Check if valid
        if not _validate(random_start):
            # Try to fix by reducing radii
            for _ in range(10):
                random_start[:, 2] *= 0.9
                if _validate(random_start):
                    break
        
        if _validate(random_start):
            rand_score = _score(random_start)
            if rand_score > best_score:
                best_result = random_start
                best_score = rand_score
            
            if time.time() < deadline - 20:
                opt_result = _optimize_from_start(random_start, deadline)
                if opt_result is not None:
                    opt_score = _score(opt_result)
                    if opt_score > best_score:
                        best_result = opt_result
                        best_score = opt_score
    
    # Final local search polish
    if time.time() < deadline - 5 and best_result is not None:
        final_result = _local_search(best_result, deadline)
        if final_result is not None:
            final_score = _score(final_result)
            if final_score > best_score:
                best_result = final_result
                best_score = final_score
    
    if best_result is None:
        # Fallback to initial grid
        best_result = initial
    
    return best_result
# EVOLVE-BLOCK-END
