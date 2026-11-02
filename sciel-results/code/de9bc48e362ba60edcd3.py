"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _grid_initial(n, rng):
    """Create a grid-based initial configuration with slight jitter."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side + rng.uniform(-0.01, 0.01)
            y = (j + 0.5) / side + rng.uniform(-0.01, 0.01)
            r = 0.5 / side * 0.95
            cells.append([x, y, r])
    # Pad if needed
    while len(cells) < n:
        cells.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.02])
    return np.array(cells[:n])

def _random_initial(n, rng):
    """Create a random initial configuration."""
    x = rng.uniform(0.05, 0.95, size=n)
    y = rng.uniform(0.05, 0.95, size=n)
    r = rng.uniform(0.02, 0.1, size=n)
    return np.column_stack([x, y, r])

def _penalty_objective(params, n, mu):
    """Objective: maximize sum of radii with penalties for violations."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Base objective: negative sum of radii (we minimize)
    obj = -np.sum(r)
    
    # Penalty for circles outside unit square
    obj += mu * np.sum(np.maximum(0, x - (1 - r))**2)
    obj += mu * np.sum(np.maximum(0, y - (1 - r))**2)
    obj += mu * np.sum(np.maximum(0, r - x)**2)
    obj += mu * np.sum(np.maximum(0, r - y)**2)
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                obj += mu * overlap**2
    
    return obj

def _penalty_grad(params, n, mu):
    """Gradient of the penalty objective."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2*n:3*n] -= 1.0
    
    # Gradient of boundary penalties
    # x > 1 - r
    mask_x_right = x > (1 - r)
    grad[:n][mask_x_right] += 2 * mu * (x[mask_x_right] - (1 - r[mask_x_right]))
    # y > 1 - r
    mask_y_top = y > (1 - r)
    grad[n:2*n][mask_y_top] += 2 * mu * (y[mask_y_top] - (1 - r[mask_y_top]))
    # r > x
    mask_r_x = r > x
    grad[2*n:3*n][mask_r_x] += 2 * mu * (r[mask_r_x] - x[mask_r_x])
    grad[:n][mask_r_x] -= 2 * mu * (r[mask_r_x] - x[mask_r_x])
    # r > y
    mask_r_y = r > y
    grad[2*n:3*n][mask_r_y] += 2 * mu * (r[mask_r_y] - y[mask_r_y])
    grad[n:2*n][mask_r_y] -= 2 * mu * (r[mask_r_y] - y[mask_r_y])
    
    # Gradient of overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy) + 1e-10
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                # d/dx_i of overlap^2 = 2*overlap * d(overlap)/dx_i = 2*overlap * (dx/dist)
                coeff = 2 * mu * overlap / dist
                grad[i] += coeff * dx
                grad[n+j] += coeff * dy
                # d/dx_j = -coeff * dx
                grad[j] -= coeff * dx
                grad[n+i] -= coeff * dy
                # d/dr_i = 2*overlap
                grad[2*n+i] += 2 * mu * overlap
                grad[2*n+j] += 2 * mu * overlap
    
    return grad

def _enforce_validity(circles, n):
    """Clamp circles to be valid (inside square, non-negative radius)."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Clamp radii to be positive
    r = np.maximum(r, 1e-6)
    
    # Clamp positions so circles stay inside [0,1]^2
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    return np.column_stack([x, y, r])

def _score_valid(circles, n):
    """Check if all circles are valid and return sum of radii if so."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check non-negative radius
    if np.any(r < 0):
        return 0.0
    
    # Check circles inside unit square
    if np.any(x < r) or np.any(y < r) or np.any(x > 1 - r) or np.any(y > 1 - r):
        return 0.0
    
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0
    
    return np.sum(r)

def construct():
    n = N
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_score = 0.0
    best_circles = None
    
    # Fallback: grid solution
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    fallback = np.array(cells[:n])
    best_circles = fallback.copy()
    best_score = _score_valid(best_circles, n)
    
    rng = np.random.default_rng(42)
    
    # Try multiple initializations and optimization runs
    n_attempts = 0
    while time.time() < deadline:
        n_attempts += 1
        
        # Choose initialization strategy
        if n_attempts % 3 == 1:
            init = _grid_initial(n, rng)
        else:
            init = _random_initial(n, rng)
        
        # Start with moderate penalty, increase if needed
        mu = 100.0
        
        for mu_iter in range(5):
            if time.time() > deadline:
                break
            
            # Optimize with current mu
            try:
                result = minimize(
                    _penalty_objective,
                    init.flatten(),
                    jac=_penalty_grad,
                    args=(n, mu),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                break
            
            if not result.success and result.fun > 1e6:
                break
            
            # Extract and validate
            params = result.x
            circles = params.reshape(n, 3)
            circles = _enforce_validity(circles, n)
            
            score = _score_valid(circles, n)
            
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
            
            # Check if constraints are satisfied (no overlap, inside square)
            x = circles[:, 0]
            y = circles[:, 1]
            r = circles[:, 2]
            
            # Check if we need to increase penalty
            max_violation = 0.0
            for i in range(n):
                max_violation = max(max_violation, max(0, r[i] - x[i], r[i] - y[i], x[i] - (1-r[i]), y[i] - (1-r[i])))
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_violation = max(max_violation, r[i] + r[j] - dist)
            
            if max_violation < 1e-4:
                break  # Constraints satisfied, stop increasing mu
            
            mu *= 3
        
        # If we found a good solution, try to polish it
        if best_score > 0 and time.time() < deadline - 5:
            # Local refinement with high penalty
            mu_high = 1000.0
            try:
                result = minimize(
                    _penalty_objective,
                    best_circles.flatten(),
                    jac=_penalty_grad,
                    args=(n, mu_high),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                params = result.x
                circles = params.reshape(n, 3)
                circles = _enforce_validity(circles, n)
                score = _score_valid(circles, n)
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
            except Exception:
                pass
        
        # Break if we've done enough attempts
        if n_attempts >= 8:
            break
    
    # Final validation
    final_score = _score_valid(best_circles, n)
    if final_score <= 0:
        return fallback
    
    return best_circles
# EVOLVE-BLOCK-END
