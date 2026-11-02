"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def construct():
    """
    Pack N circles in the unit square maximizing sum of radii.
    Uses a differentiable optimization approach with soft contact constraints.
    """
    # Initial guess: grid placement
    side = int(np.ceil(np.sqrt(N)))
    # We'll use a grid of side x side, taking the first N positions
    # Better initial guess: slightly perturbed grid
    grid_positions = []
    for i in range(side):
        for j in range(side):
            grid_positions.append([(i + 0.5) / side, (j + 0.5) / side])
    grid_positions = grid_positions[:N]
    
    # Initial radii: uniform small radius
    r_init = 0.5 / side - 0.01
    x0 = np.array([p[0] for p in grid_positions] + [p[1] for p in grid_positions] + [r_init] * N)
    
    n = N
    # Variables: x[0..n-1], y[n..2n-1], r[2n..3n-1]
    # Objective: maximize sum(r) => minimize -sum(r)
    
    def objective(v):
        x = v[:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        return -np.sum(r)
    
    def constraints_func(v):
        x = v[:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        
        # Constraints:
        # 1. Circles inside unit square: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
        # 2. Non-overlapping: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        
        cons = []
        
        # Boundary constraints (linear, easy)
        for i in range(n):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i], 'jac': lambda v, i=i: _jac_x_min_r(v, i)})
            # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2*n+i], 'jac': lambda v, i=i: _jac_1_x_r(v, i)})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i], 'jac': lambda v, i=i: _jac_y_min_r(v, i)})
            # 1 - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[n+i] - v[2*n+i], 'jac': lambda v, i=i: _jac_1_y_r(v, i)})
        
        # Pairwise non-overlap constraints using soft contact
        # For efficiency, we'll use a simplified approach with only nearby pairs
        # But for correctness, we need all pairs. Let's use a penalty-based approach instead.
        
        return cons
    
    def _jac_x_min_r(v, i):
        jac = np.zeros(3*n)
        jac[i] = 1.0
        jac[2*n+i] = -1.0
        return jac
    
    def _jac_1_x_r(v, i):
        jac = np.zeros(3*n)
        jac[i] = -1.0
        jac[2*n+i] = -1.0
        return jac
    
    def _jac_y_min_r(v, i):
        jac = np.zeros(3*n)
        jac[n+i] = 1.0
        jac[2*n+i] = -1.0
        return jac
    
    def _jac_1_y_r(v, i):
        jac = np.zeros(3*n)
        jac[n+i] = -1.0
        jac[2*n+i] = -1.0
        return jac
    
    # Use SLSQP with boundary constraints only, and handle overlap via penalty in objective
    # This is more robust than adding 496 pairwise constraints
    
    def objective_with_penalty(v):
        x = v[:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += overlap * overlap
        
        # Penalty for being outside boundary
        for i in range(n):
            if x[i] < r[i]:
                penalty += (r[i] - x[i]) ** 2
            if x[i] > 1 - r[i]:
                penalty += (x[i] - (1 - r[i])) ** 2
            if y[i] < r[i]:
                penalty += (r[i] - y[i]) ** 2
            if y[i] > 1 - r[i]:
                penalty += (y[i] - (1 - r[i])) ** 2
        
        return -np.sum(r) + 100.0 * penalty
    
    def jac_objective_with_penalty(v):
        x = v[:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        
        jac = np.zeros(3*n)
        
        # Gradient of -sum(r)
        jac[2*n:3*n] = -1.0
        
        # Gradient of penalty terms
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # d(overlap^2)/dx_i = 2*overlap * d(overlap)/dx_i
                    # d(overlap)/dx_i = d(min_dist - dist)/dx_i = -d(dist)/dx_i = -dx/dist
                    coeff = 2 * overlap * (-dx / dist)
                    jac[i] += 100.0 * coeff
                    coeff_j = 2 * overlap * (dx / dist)
                    jac[j] += 100.0 * coeff_j
                    
                    coeff_y = 2 * overlap * (-dy / dist)
                    jac[n+i] += 100.0 * coeff_y
                    coeff_yj = 2 * overlap * (dy / dist)
                    jac[n+j] += 100.0 * coeff_yj
                    
                    # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
                    jac[2*n+i] += 100.0 * 2 * overlap
                    jac[2*n+j] += 100.0 * 2 * overlap
        
        # Boundary penalties
        for i in range(n):
            if x[i] < r[i]:
                diff = r[i] - x[i]
                jac[i] += 100.0 * 2 * diff * (-1)
                jac[2*n+i] += 100.0 * 2 * diff
            if x[i] > 1 - r[i]:
                diff = x[i] - (1 - r[i])
                jac[i] += 100.0 * 2 * diff
                jac[2*n+i] += 100.0 * 2 * diff * (-1)
            if y[i] < r[i]:
                diff = r[i] - y[i]
                jac[n+i] += 100.0 * 2 * diff * (-1)
                jac[2*n+i] += 100.0 * 2 * diff
            if y[i] > 1 - r[i]:
                diff = y[i] - (1 - r[i])
                jac[n+i] += 100.0 * 2 * diff
                jac[2*n+i] += 100.0 * 2 * diff * (-1)
        
        return jac
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x_i
    for i in range(n):
        bounds.append((0.0, 1.0))  # y_i
    for i in range(n):
        bounds.append((1e-4, 0.5))  # r_i
    
    # Run optimization
    result = minimize(
        objective_with_penalty,
        x0,
        jac=jac_objective_with_penalty,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
    )
    
    v_opt = result.x
    x = v_opt[:n]
    y = v_opt[n:2*n]
    r = v_opt[2*n:3*n]
    
    # Clip to ensure validity
    r = np.clip(r, 1e-4, 0.5)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    # Final validation and fix any remaining overlaps with a simple relaxation
    for iteration in range(100):
        fixed = False
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Push apart
                    overlap = min_dist - dist
                    if dist > 1e-10:
                        dx_unit = dx / dist
                        dy_unit = dy / dist
                        push = overlap / 2.0
                        x[i] += push * dx_unit
                        y[i] += push * dy_unit
                        x[j] -= push * dx_unit
                        y[j] -= push * dy_unit
                    else:
                        x[i] += push
                    fixed = True
        
        # Re-clip
        r = np.clip(r, 1e-4, 0.5)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        if not fixed:
            break
    
    return np.column_stack([x, y, r])
# EVOLVE-BLOCK-END
