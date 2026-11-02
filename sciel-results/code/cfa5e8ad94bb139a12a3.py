"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26
PENALTY_SCALE = 100.0
SMOOTH_EPS = 1e-6

def _smooth_penalty_boundary(x, y, r):
    """Smooth penalty for circle being outside [0,1]^2."""
    # Use soft version: for each boundary, penalty = max(0, violation)^2
    # Smoothed: use log-sum-exp style or just clip with small epsilon
    violations = np.array([
        r - x,
        r - y, 
        r - (1 - x),
        r - (1 - y)
    ])
    # Smooth max(0, v)^2 using v * sigmoid(v/eps) approximation
    # Simpler: just use np.maximum with small smoothing
    pos_violations = np.maximum(violations, 0)
    return np.sum(pos_violations ** 2)

def _smooth_pairwise_penalty(x, y, r):
    """Smooth penalty for overlapping circles."""
    n = N
    total_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx * dx + dy * dy
            d = np.sqrt(d_sq + 1e-12)  # Avoid division by zero
            min_d = r[i] + r[j]
            overlap = min_d - d
            if overlap > 0:
                total_pen += overlap * overlap
    return total_pen

def _objective_and_grad(params):
    """Objective and gradient for optimization.
    
    Returns (objective_value, gradient).
    Objective = penalty - sum(radii), so minimizing maximizes radii.
    """
    n = N
    x = params[:n].copy()
    y = params[n:2*n].copy()
    r = params[2*n:].copy()
    
    # Objective value
    obj = 0.0
    
    # Boundary penalties
    for i in range(n):
        # x-boundary: violation = r[i] - x[i] if x[i] < r[i]
        viol_x_left = max(0, r[i] - x[i])
        viol_x_right = max(0, r[i] - (1 - x[i]))
        viol_y_bottom = max(0, r[i] - y[i])
        viol_y_top = max(0, r[i] - (1 - y[i]))
        
        obj += PENALTY_SCALE * (viol_x_left**2 + viol_x_right**2 + viol_y_bottom**2 + viol_y_top**2)
    
    # Pairwise overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_d = r[i] + r[j]
            overlap = min_d - d
            if overlap > 0:
                obj += PENALTY_SCALE * overlap * overlap
    
    # Negative sum of radii (we want to maximize sum(r))
    obj -= np.sum(r)
    
    # Gradient
    grad = np.zeros_like(params)
    
    # Gradient of boundary penalties w.r.t. x, y, r
    for i in range(n):
        # d/dx of PENALTY_SCALE * (max(0, r-x)^2 + max(0, r-(1-x))^2)
        if x[i] < r[i]:
            grad[i] += PENALTY_SCALE * 2 * (r[i] - x[i]) * (-1)  # d/dx of (r-x)^2
        if x[i] > 1 - r[i]:
            grad[i] += PENALTY_SCALE * 2 * (r[i] - (1 - x[i])) * (1)  # d/dx of (r-1+x)^2
        
        # d/dy
        idx_y = n + i
        if y[i] < r[i]:
            grad[idx_y] += PENALTY_SCALE * 2 * (r[i] - y[i]) * (-1)
        if y[i] > 1 - r[i]:
            grad[idx_y] += PENALTY_SCALE * 2 * (r[i] - (1 - y[i])) * (1)
        
        # d/dr of boundary penalties
        idx_r = 2*n + i
        if x[i] < r[i]:
            grad[idx_r] += PENALTY_SCALE * 2 * (r[i] - x[i]) * (1)
        if x[i] > 1 - r[i]:
            grad[idx_r] += PENALTY_SCALE * 2 * (r[i] - (1 - x[i])) * (1)
        if y[i] < r[i]:
            grad[idx_r] += PENALTY_SCALE * 2 * (r[i] - y[i]) * (1)
        if y[i] > 1 - r[i]:
            grad[idx_r] += PENALTY_SCALE * 2 * (r[i] - (1 - y[i])) * (1)
        
        # d/dr of -sum(r)
        grad[idx_r] += -1
    
    # Gradient of pairwise penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_d = r[i] + r[j]
            overlap = min_d - d
            
            if overlap > 0:
                # d/dx_i of PENALTY_SCALE * overlap^2
                # = PENALTY_SCALE * 2 * overlap * d(overlap)/dx_i
                # d(overlap)/dx_i = -dx/d (since overlap = r_i + r_j - d, and d depends on x_i)
                # Actually: overlap = r_i + r_j - d, so d(overlap)/dx_i = -d(d)/dx_i = -dx_i/d
                # where dx_i = x_i - x_j, so d(d)/dx_i = dx_i/d
                factor = PENALTY_SCALE * 2 * overlap
                grad[i] += factor * (-dx / d)       # d/dx_i
                grad[n + i] += factor * (-dy / d)   # d/dy_i
                grad[i] += factor * (-dx / d) * 0   # already counted
                grad[2*n + i] += factor * (1)       # d/dr_i = 1
                grad[2*n + j] += factor * (1)       # d/dr_j = 1
                
                # For circle j:
                grad[j] += factor * (dx / d)        # d/dx_j = -d(d)/dx_j = dx/d
                grad[n + j] += factor * (dy / d)    # d/dy_j = dy/d
    
    return obj, grad

def _compute_score(params):
    """Compute the actual score (sum of radii) if feasible, else -inf."""
    n = N
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    
    # Check boundaries
    for i in range(n):
        if x[i] < -1e-9 or x[i] > 1 + 1e-9 or y[i] < -1e-9 or y[i] > 1 + 1e-9:
            return -np.inf
        if r[i] < 0:
            return -np.inf
        if x[i] < r[i] - 1e-9 or x[i] > 1 - r[i] + 1e-9:
            return -np.inf
        if y[i] < r[i] - 1e-9 or y[i] > 1 - r[i] + 1e-9:
            return -np.inf
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - 1e-9:
                return -np.inf
    
    return np.sum(r)

def _grid_initialization(n_cols, n_rows):
    """Initialize circles in a grid pattern."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            x[count] = (col + 0.5) / n_cols
            y[count] = (row + 0.5) / n_rows
            r[count] = 0.5 / max(n_cols, n_rows) - 1e-4
            count += 1
    
    # Fill remaining with small circles in gaps
    if count < N:
        # Place remaining circles in a spiral or ring pattern
        for i in range(count, N):
            angle = 2 * np.pi * (i - count) / max(1, N - count)
            x[i] = 0.5 + 0.15 * np.cos(angle)
            y[i] = 0.5 + 0.15 * np.sin(angle)
            r[i] = 0.04
    
    return np.concatenate([x, y, r])

def _random_initialization():
    """Generate a random initialization."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.03, 0.15, N)
    return np.concatenate([x, y, r])

def _refine(params, deadline, max_iter=300, ftol=1e-10):
    """Run L-BFGS-B optimization with gradient."""
    bounds = [(0, 1)] * (2 * N) + [(0.01, 0.5)] * N
    
    def obj_wrapper(p):
        val, _ = _objective_and_grad(p)
        return val
    
    def grad_wrapper(p):
        _, g = _objective_and_grad(p)
        return g
    
    try:
        result = minimize(
            obj_wrapper,
            params,
            jac=grad_wrapper,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8}
        )
        return result.x
    except Exception:
        return params

def _polish(params, deadline):
    """Final high-precision polish pass."""
    if time.time() > deadline - 5:
        return params
    
    # Run with tighter tolerance and more iterations
    try:
        result = minimize(
            lambda p: _objective_and_grad(p)[0],
            params,
            jac=lambda p: _objective_and_grad(p)[1],
            method='L-BFGS-B',
            bounds=[(0, 1)] * (2 * N) + [(0.01, 0.5)] * N,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        return result.x
    except Exception:
        return params

def _enforce_validity(params):
    """Clip parameters to ensure validity."""
    n = N
    x = params[:n].copy()
    y = params[n:2*n].copy()
    r = params[2*n:].copy()
    
    # Ensure radii are positive and reasonable
    r = np.clip(r, 1e-6, 0.5)
    
    # Ensure circles are inside the square
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    return np.concatenate([x, y, r])

def construct():
    """Construct an arrangement of 26 circles in the unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_params = None
    best_score = -np.inf
    
    bounds = [(0, 1)] * (2 * N) + [(0.01, 0.5)] * N
    
    # Strategy 1: Multiple grid configurations
    grid_configs = [
        (5, 6),  # 5 cols, 6 rows = 30 slots, use 26
        (6, 5),  # 6 cols, 5 rows = 30 slots, use 26
        (4, 7),  # 4 cols, 7 rows = 28 slots, use 26
        (7, 4),  # 7 cols, 4 rows = 28 slots, use 26
    ]
    
    for n_cols, n_rows in grid_configs:
        if time.time() > deadline - 40:
            break
        
        params0 = _grid_initialization(n_cols, n_rows)
        params1 = _refine(params0, deadline, max_iter=200, ftol=1e-10)
        score = _compute_score(params1)
        
        if score > best_score:
            best_score = score
            best_params = params1.copy()
    
    # Strategy 2: Random restarts
    n_random_restarts = 15
    for _ in range(n_random_restarts):
        if time.time() > deadline - 20:
            break
        
        params0 = _random_initialization()
        params1 = _refine(params0, deadline, max_iter=150, ftol=1e-10)
        score = _compute_score(params1)
        
        if score > best_score:
            best_score = score
            best_params = params1.copy()
    
    # Strategy 3: If we have a good solution, try to improve it further
    if best_params is not None and best_score > 0:
        if time.time() < deadline - 15:
            # Try perturbations of the best solution
            for trial in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Add small random perturbation
                noise = np.random.normal(0, 0.01, 3 * N)
                params_perturbed = best_params + noise
                params_perturbed = _enforce_validity(params_perturbed)
                
                params1 = _refine(params_perturbed, deadline, max_iter=100, ftol=1e-10)
                score = _compute_score(params1)
                
                if score > best_score:
                    best_score = score
                    best_params = params1.copy()
        
        # Final polish
        if time.time() < deadline - 5:
            best_params = _polish(best_params, deadline)
            final_score = _compute_score(best_params)
            if final_score > best_score:
                best_score = final_score
    
    # Return the best valid solution
    if best_params is not None and best_score > 0:
        best_params = _enforce_validity(best_params)
        n = N
        x = best_params[:n]
        y = best_params[n:2*n]
        r = best_params[2*n:]
        return np.column_stack([x, y, r])
    
    # Fallback: return a valid grid arrangement
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    count = 0
    for row in range(side):
        for col in range(side):
            if count >= N:
                break
            x_val = (col + 0.5) / side
            y_val = (row + 0.5) / side
            r_val = 0.5 / side - 1e-3
            cells.append([x_val, y_val, r_val])
            count += 1
    
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
