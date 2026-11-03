"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Uses a multi-start local optimization approach:
    1. Generate multiple random initial configurations.
    2. For each, run gradient-based optimization to push circles apart and maximize radii.
    3. Return the best configuration found.
    """
    best_value = -np.inf
    best_result = None
    rng = np.random.default_rng(42)
    
    def objective_and_grad(params):
        """
        params: flat array of shape (3*N,) = [x_1, y_1, r_1, x_2, y_2, r_2, ...]
        We want to maximize sum(r_i) subject to:
          - r_i > 0
          - x_i - r_i >= 0, x_i + r_i <= 1
          - y_i - r_i >= 0, y_i + r_i <= 1
          - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
        
        We minimize a penalty function:
          - sum of overlaps (pairwise)
          - sum of boundary violations
          - negative sum of radii (to maximize)
        """
        n = len(params) // 3
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        penalty = 0.0
        grad = np.zeros_like(params)
        
        # Boundary penalties: circles must be inside [0,1]^2
        # Left: x_i - r_i >= 0 => violation if x_i - r_i < 0
        # Right: x_i + r_i <= 1 => violation if x_i + r_i > 1
        # Bottom: y_i - r_i >= 0
        # Top: y_i + r_i <= 1
        
        # We use a smooth penalty: max(0, violation)^2
        for i in range(n):
            # Left boundary
            v = r[i] - x[i]  # violation if positive
            if v > 0:
                penalty += v * v
                grad[3*i] += -2 * v      # d/dx_i
                grad[3*i+2] += 2 * v     # d/dr_i
            
            # Right boundary
            v = x[i] + r[i] - 1.0
            if v > 0:
                penalty += v * v
                grad[3*i] += 2 * v
                grad[3*i+2] += 2 * v
            
            # Bottom boundary
            v = r[i] - y[i]
            if v > 0:
                penalty += v * v
                grad[3*i+1] += -2 * v
                grad[3*i+2] += 2 * v
            
            # Top boundary
            v = y[i] + r[i] - 1.0
            if v > 0:
                penalty += v * v
                grad[3*i+1] += 2 * v
                grad[3*i+2] += 2 * v
        
        # Pairwise overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-8:
                    # Overlap amount
                    v = min_dist - dist
                    penalty += v * v
                    # Gradient with respect to positions and radii
                    # d(dist)/dx_i = -dx/dist, d(dist)/dy_i = -dy/dist
                    # d(min_dist)/dr_i = 1
                    # d(v)/dx_i = -d(dist)/dx_i = dx/dist
                    # d(v)/dy_i = dy/dist
                    # d(v)/dr_i = 1 - (-dx/dist * 0) ... wait
                    # v = r_i + r_j - dist
                    # d(v)/dx_i = -d(dist)/dx_i = dx/dist
                    # d(v)/dy_i = dy/dist
                    # d(v)/dr_i = 1
                    # d(v)/dr_j = 1
                    # d(v)/dx_j = -dx/dist
                    # d(v)/dy_j = -dy/dist
                    
                    inv_dist = 1.0 / dist
                    grad[3*i] += 2 * v * dx * inv_dist
                    grad[3*i+1] += 2 * v * dy * inv_dist
                    grad[3*i+2] += 2 * v
                    
                    grad[3*j] += -2 * v * dx * inv_dist
                    grad[3*j+1] += -2 * v * dy * inv_dist
                    grad[3*j+2] += 2 * v
                elif dist <= 1e-8:
                    # Circles at same center, push apart
                    penalty += min_dist * min_dist
                    # Arbitrary push
                    grad[3*i] += 2 * min_dist * 0.1
                    grad[3*j] += -2 * min_dist * 0.1
        
        # Objective: minimize penalty - sum(r_i)
        # We want to maximize sum(r_i), so subtract it
        penalty -= np.sum(r)
        
        # Gradient of -sum(r_i)
        for i in range(n):
            grad[3*i+2] -= 1.0
        
        return penalty, grad
    
    def run_optimization(params0, max_iter=200):
        """Run L-BFGS-B optimization from initial params."""
        n = len(params0) // 3
        
        # Define bounds: x, y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.001, 0.5))  # r
        
        # Use L-BFGS-B with gradient
        result = minimize(
            lambda p: objective_and_grad(p)[0],
            params0,
            jac=lambda p: objective_and_grad(p)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        # Check if result is feasible and compute value
        params = result.x
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Check feasibility
        feasible = True
        # Boundary check
        for i in range(n):
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > 1.0 + 1e-6:
                feasible = False
                break
            if y[i] - r[i] < -1e-6 or y[i] + r[i] > 1.0 + 1e-6:
                feasible = False
                break
        if feasible:
            # Pairwise check
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if dist < r[i] + r[j] - 1e-6:
                        feasible = False
                        break
                if not feasible:
                    break
        
        if feasible:
            value = np.sum(r)
        else:
            # Even if not perfectly feasible, use the sum of radii if penalty is low
            val, _ = objective_and_grad(params)
            # If penalty is small, accept
            if val < 1e-3:
                value = np.sum(r)
            else:
                value = -np.inf
        
        return value, params
    
    # Multiple random restarts
    n_restarts = 20
    for restart in range(n_restarts):
        # Generate random initial configuration
        # Start with a grid-like perturbation or pure random
        if restart == 0:
            # Grid initialization
            side = int(np.ceil(np.sqrt(N)))
            params0 = np.zeros(3*N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    r0 = 0.5 / side * 0.9
                    params0[3*idx] = cx
                    params0[3*idx+1] = cy
                    params0[3*idx+2] = r0
                    idx += 1
        else:
            # Random initialization with small radii
            params0 = np.zeros(3*N)
            for i in range(N):
                params0[3*i] = rng.uniform(0.1, 0.9)
                params0[3*i+1] = rng.uniform(0.1, 0.9)
                params0[3*i+2] = rng.uniform(0.01, 0.05)
        
        value, params = run_optimization(params0, max_iter=300)
        
        if value > best_value:
            best_value = value
            best_result = params
    
    # If no feasible solution found, fall back to grid
    if best_result is None or best_value < 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Format result
    x = best_result[0::3]
    y = best_result[1::3]
    r = best_result[2::3]
    
    # Clamp to ensure feasibility
    for i in range(N):
        r[i] = max(r[i], 0.001)
        x[i] = np.clip(x[i], r[i], 1.0 - r[i])
        y[i] = np.clip(y[i], r[i], 1.0 - r[i])
    
    result = np.column_stack([x, y, r])
    return result
# EVOLVE-BLOCK-END
