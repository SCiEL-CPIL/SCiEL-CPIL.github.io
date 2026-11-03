"""
Strategy: Successive Convex Relaxation (SCR) for Circle Packing
This approach formulates the packing problem as a non-convex Quadratic Program (QCP).
We maximize the sum of radii subject to non-overlap constraints.
The non-convex constraints (distance between centers >= sum of radii) are linearized
using first-order Taylor approximations around the current solution.
This transforms the problem into a sequence of convex Quadratic Programs (QPs).
Each iteration solves the convex relaxation and updates the linearization points.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _solve_qp_subproblem(pos, radii, w, h):
    """
    Solves the convex QP subproblem for one SCR iteration.
    Variables: x = [dx_0, dy_0, ..., dx_{N-1}, dy_{N-1}, r_0, ..., r_{N-1}]
    Objective: maximize sum(r_i) => minimize -sum(r_i)
    
    Constraints:
    1. Linearized non-overlap: 
       For each pair (i, j), let d_ij = p_i - p_j.
       Constraint: d_ij^T (p_i' - p_j') >= ||d_ij|| - (r_i + r_j)
       Wait, the standard linearization of ||x|| >= c is x^T y / ||y|| >= c if y != 0.
       Here, we want ||p_i - p_j|| >= r_i + r_j.
       Let y = p_i - p_j (current positions).
       Linearization: (p_i - p_j)^T (p_i' - p_j') / ||y|| >= r_i + r_j.
       This is linear in positions but involves radii on the RHS.
       Actually, we can treat radii as variables too.
       The constraint is: (p_i - p_j)^T (p_i' - p_j') / ||y|| - r_i - r_j >= 0.
       This is linear in x (positions and radii).
       
    2. Wall constraints:
       r_i <= p_ix (left)
       w - r_i >= p_ix (right)
       r_i <= p_iy (bottom)
       h - r_i >= p_iy (top)
       These are linear.
       
    3. Bounds:
       r_i >= 0
       0 <= p_ix <= w
       0 <= p_iy <= h
       
    Since the objective is linear and constraints are linear, this is actually a Linear Program (LP)
    if we fix the linearization coefficients.
    However, the linearization coefficients depend on the current positions.
    
    We will use scipy.optimize.minimize with method='SLSQP' or 'trust-constr' to handle the linear constraints.
    Given the small size (42 variables), SLSQP should be fast and robust.
    """
    n_vars = 2 * N + N  # 3N variables
    
    def objective(x):
        r = x[2*N:3*N]
        return -np.sum(r)
        
    def constraint_func(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:3*N]
        
        cons = []
        
        # Non-overlap linearized constraints
        for i in range(N):
            for j in range(i+1, N):
                # Current difference vector
                dy = pos[i] - pos[j]
                norm_y = np.linalg.norm(dy)
                if norm_y < 1e-8:
                    # Degenerate case: push apart along arbitrary axis
                    dy = np.array([1.0, 0.0])
                    norm_y = 1.0
                else:
                    dy = dy / norm_y # Unit vector direction
                    
                # Linearized constraint: dy^T (p_i - p_j) >= r_i + r_j
                # dy^T p_i - dy^T p_j - r_i - r_j >= 0
                # In scipy minimize, constraints are dict with 'fun' returning array where fun(x) >= 0
                # Let's compute the value
                val = np.dot(dy, p[i] - p[j]) - r[i] - r[j]
                cons.append(val)
                
        # Wall constraints
        for i in range(N):
            # Left: p_ix - r_i >= 0
            cons.append(p[i, 0] - r[i])
            # Right: w - p_ix - r_i >= 0
            cons.append(w - p[i, 0] - r[i])
            # Bottom: p_iy - r_i >= 0
            cons.append(p[i, 1] - r[i])
            # Top: h - p_iy - r_i >= 0
            cons.append(h - p[i, 1] - r[i])
            
        return cons

    constraints = [{'type': 'ineq', 'fun': constraint_func}]
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w)) # x_i
        bounds.append((0, h)) # y_i
        bounds.append((0, 1.0)) # r_i (upper bound 1.0 is safe for perimeter 4)
        
    # Initial guess for the subproblem is the current solution
    x0 = np.concatenate([pos.flatten(), radii])
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'maxiter': 500, 'ftol': 1e-12})
    
    if res.success:
        p_new = res.x[:2*N].reshape(N, 2)
        r_new = res.x[2*N:3*N]
        return p_new, r_new
    else:
        # If optimization fails, return current state or slightly shrunk
        return pos, radii

def _is_valid(pos, radii, w, h, tol=1e-5):
    for i in range(N):
        r = radii[i]
        if pos[i, 0] - r < -tol or pos[i, 0] + r > w + tol:
            return False
        if pos[i, 1] - r < -tol or pos[i, 1] + r > h + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5, 1.33, 0.75, 1.25, 1.1, 0.9]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 / (1.0 + aspect)
        h = 2.0 - w
        
        # Initial grid placement
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) * (w / side)
            pos[i, 1] = (row + 0.5) * (h / side)
            
        # Start with small radii
        min_dim = min(w, h)
        base_r = 0.5 * min_dim / side * 0.5
        radii = np.full(N, base_r)
        
        # SCR Loop
        # Iterate until no improvement or max iterations
        for _ in range(30):
            if time.time() > deadline:
                break
                
            p_new, r_new = _solve_qp_subproblem(pos, radii, w, h)
            
            # Check if we actually improved the objective (sum of radii)
            if np.sum(r_new) <= np.sum(radii) + 1e-8:
                break
                
            # Check validity of the new solution
            if _is_valid(p_new, r_new, w, h, tol=1e-5):
                pos = p_new
                radii = r_new
            else:
                # If invalid, it might be due to numerical errors in the solver or tight constraints.
                # Try to repair by shrinking radii slightly
                scale = 0.99
                while not _is_valid(pos, radii * scale, w, h, tol=1e-5) and scale > 0.5:
                    scale -= 0.01
                if _is_valid(pos, radii * scale, w, h, tol=1e-5):
                    radii = radii * scale
                else:
                    break
        
        # Final validation and update best
        if _is_valid(pos, radii, w, h, tol=1e-6):
            value = np.sum(radii)
            if value > best_value:
                best_value = value
                best_result = np.column_stack([pos, radii])
        else:
            # Attempt repair for the best found so far in this aspect ratio
            scale = 0.99
            while not _is_valid(pos, radii * scale, w, h, tol=1e-6) and scale > 0.5:
                scale -= 0.01
            if _is_valid(pos, radii * scale, w, h, tol=1e-6):
                value = np.sum(radii * scale)
                if value > best_value:
                    best_value = value
                    best_result = np.column_stack([pos, radii * scale])

    if best_result is None:
        # Fallback: simple valid grid
        w, h = 1.0, 1.0
        side = 5
        pos = np.zeros((N, 2))
        radii = np.full(N, 0.08)
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
        best_result = np.column_stack([pos, radii])
        
    return best_result
