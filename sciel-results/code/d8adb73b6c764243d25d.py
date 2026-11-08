"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _grid_initial():
    """Create a grid-based initial configuration of N circles."""
    side = int(np.ceil(np.sqrt(N)))
    # Use a slightly tighter grid to start
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95  # slightly smaller to ensure non-overlap initially
            positions.append([x, y, r])
    
    # If we have fewer than N, add more
    while len(positions) < N:
        idx = len(positions)
        i = idx % side
        j = idx // side
        if j >= side:
            break
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.5 / side * 0.95
        positions.append([x, y, r])
    
    return np.array(positions[:N])


def _objective_and_gradient(params, N):
    """
    Objective: minimize negative sum of radii
    params: flattened array of [x1, y1, r1, x2, y2, r2, ...]
    """
    n = N
    # Unpack
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Objective: -sum(r)
    obj = -np.sum(r)
    
    # Gradient
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr_i = -1
    
    # Constraints (penalty-based for L-BFGS-B which doesn't support general constraints well)
    # We'll use a penalty approach with soft constraints
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # Circle-circle: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    
    # Use a barrier/penalty approach
    penalty = 0.0
    penalty_grad = np.zeros_like(params)
    
    # Boundary penalties
    for i in range(n):
        # x constraint: r <= x and r <= 1-x
        for val, sign in [(x[i], 1), (1-x[i], 1)]:
            if r[i] > val:
                viol = r[i] - val
                penalty += 100 * viol**2
                # d(penalty)/dr_i = 200 * viol
                penalty_grad[2*i + 2] += 200 * viol
                # d(penalty)/d(x_i) or d/d(1-x_i)
                # For r <= x: violation = r - x, d/dx_i = -200*viol
                # For r <= 1-x: violation = r - (1-x), d/dx_i = +200*viol
                if val == x[i]:
                    penalty_grad[i] -= 200 * viol
                else:
                    penalty_grad[i] += 200 * viol
        
        for val, sign in [(y[i], 1), (1-y[i], 1)]:
            if r[i] > val:
                viol = r[i] - val
                penalty += 100 * viol**2
                penalty_grad[2*i + 2] += 200 * viol
                if val == y[i]:
                    penalty_grad[i + n] -= 200 * viol
                else:
                    penalty_grad[i + n] += 200 * viol
    
    # Circle-circle penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            dist = np.sqrt(dist_sq + 1e-12)
            if dist < min_dist:
                viol = min_dist - dist
                penalty += 100 * viol**2
                # Gradient computation
                # d(viol)/dx_i = -dx/dist, d(viol)/dx_j = dx/dist
                # d(viol)/dr_i = 1, d(viol)/dr_j = 1
                coeff = 200 * viol / dist
                penalty_grad[i] += coeff * dx
                penalty_grad[j] -= coeff * dx
                penalty_grad[i + n] += coeff * dy
                penalty_grad[j + n] -= coeff * dy
                penalty_grad[2*i + 2] += 200 * viol
                penalty_grad[2*j + 2] += 200 * viol
    
    total_obj = obj + penalty
    total_grad = grad + penalty_grad
    
    return total_obj, total_grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration
    init = _grid_initial()
    
    # Flatten
    x0 = init.flatten()
    n = N
    total_params = 3 * n
    
    # Bounds: x in [0,1], y in [0,1], r in [0.001, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.001, 0.5))  # r
    
    # Try multiple random restarts with different initial configurations
    best_obj = np.inf
    best_params = x0.copy()
    
    def _run_optimization(params_init, max_iter=500):
        """Run L-BFGS-B optimization."""
        result = minimize(
            lambda p: _objective_and_gradient(p, n)[0],
            params_init,
            jac=lambda p: _objective_and_gradient(p, n)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        return result
    
    # First, try with the grid initial
    result = _run_optimization(x0, max_iter=300)
    if result.fun < best_obj:
        best_obj = result.fun
        best_params = result.x.copy()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Not enough time for more restarts
        pass
    else:
        # Try a few random restarts
        np.random.seed(42)
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            # Random initial positions
            x_rand = np.random.uniform(0.1, 0.9, size=n)
            y_rand = np.random.uniform(0.1, 0.9, size=n)
            r_rand = np.random.uniform(0.02, 0.1, size=n)
            
            params_rand = np.zeros(3*n)
            params_rand[0::3] = x_rand
            params_rand[1::3] = y_rand
            params_rand[2::3] = r_rand
            
            result = _run_optimization(params_rand, max_iter=200)
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
    
    # Extract best solution
    x_best = best_params[0::3]
    y_best = best_params[1::3]
    r_best = best_params[2::3]
    
    # Validation and fixing: ensure circles are inside and non-overlapping
    # Clip radii to ensure boundary constraints
    for i in range(n):
        # Ensure inside unit square
        r_max = min(x_best[i], 1 - x_best[i], y_best[i], 1 - y_best[i])
        r_best[i] = min(r_best[i], r_max - 1e-6)
        r_best[i] = max(r_best[i], 0.001)
    
    # Final check for overlaps and fix if needed
    for i in range(n):
        for j in range(i+1, n):
            dx = x_best[i] - x_best[j]
            dy = y_best[i] - y_best[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r_best[i] + r_best[j]
            if dist < min_dist:
                # Reduce the smaller radius
                if r_best[i] < r_best[j]:
                    r_best[i] = max(0.001, (dist - r_best[j]) * 0.99)
                else:
                    r_best[j] = max(0.001, (dist - r_best[i]) * 0.99)
    
    # Ensure all radii are positive and circles are inside
    for i in range(n):
        r_best[i] = max(r_best[i], 0.001)
        r_best[i] = min(r_best[i], min(x_best[i], 1-x_best[i], y_best[i], 1-y_best[i]) - 1e-6)
        if r_best[i] < 0.001:
            r_best[i] = 0.001
    
    result_array = np.column_stack([x_best, y_best, r_best])
    
    return result_array

# EVOLVE-BLOCK-END
