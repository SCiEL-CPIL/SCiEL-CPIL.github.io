"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def objective_and_grad(params, mu):
        """Objective: -sum(r) + penalty for violations. Returns (value, gradient)."""
        n = N
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        
        # Objective: maximize sum of radii -> minimize -sum(r)
        obj_val = -np.sum(r)
        obj_grad = np.zeros(3*n)
        obj_grad[2*n:3*n] = -1.0
        
        # Penalty terms
        pen_val = 0.0
        pen_grad = np.zeros(3*n)
        
        # Circle-circle constraints: dist(i,j) >= r_i + r_j
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy) + 1e-10
                constraint = d - r[i] - r[j]
                if constraint < 0:
                    pen_val += mu * constraint * constraint
                    # Gradient of mu * constraint^2
                    grad_c_i_x = dx / d
                    grad_c_i_y = dy / d
                    grad_c_j_x = -dx / d
                    grad_c_j_y = -dy / d
                    grad_c_i_r = -1.0
                    grad_c_j_r = -1.0
                    factor = 2.0 * mu * constraint
                    pen_grad[i] += factor * grad_c_i_x
                    pen_grad[n+i] += factor * grad_c_i_y
                    pen_grad[2*n+i] += factor * grad_c_i_r
                    pen_grad[j] += factor * grad_c_j_x
                    pen_grad[n+j] += factor * grad_c_j_y
                    pen_grad[2*n+j] += factor * grad_c_j_r
        
        # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        for i in range(n):
            constraints = [r[i] - x[i], r[i] - (1.0 - x[i]), r[i] - y[i], r[i] - (1.0 - y[i])]
            grads = [
                (0.0, 0.0, 1.0, -1.0),      # r - x
                (0.0, 0.0, 1.0, 1.0),       # r - (1-x)
                (0.0, 0.0, 1.0, -1.0),      # r - y
                (0.0, 0.0, 1.0, 1.0)        # r - (1-y)
            ]
            for c, g in zip(constraints, grads):
                if c < 0:
                    pen_val += mu * c * c
                    factor = 2.0 * mu * c
                    pen_grad[i] += factor * g[0]
                    pen_grad[n+i] += factor * g[1]
                    pen_grad[2*n+i] += factor * g[2]
                    # Note: the 4th component doesn't map to a variable directly in this formulation
                    # Actually, let me reconsider the boundary constraints
        
        # Let me redo boundary constraints properly
        # x_i >= r_i  -> r_i - x_i <= 0
        # x_i <= 1 - r_i  -> x_i + r_i - 1 <= 0  -> -(1 - x_i - r_i) <= 0
        # y_i >= r_i  -> r_i - y_i <= 0
        # y_i <= 1 - r_i  -> y_i + r_i - 1 <= 0
        
        # Reset penalty for boundaries
        pen_val_boundary = 0.0
        pen_grad_boundary = np.zeros(3*n)
        
        for i in range(n):
            # Constraint 1: r_i - x_i <= 0
            c1 = r[i] - x[i]
            if c1 < 0:
                pen_val_boundary += mu * c1 * c1
                factor = 2.0 * mu * c1
                pen_grad_boundary[i] += factor * (-1.0)  # d/dc of (r-x) wrt x = -1
                pen_grad_boundary[2*n+i] += factor * 1.0   # d/dc of (r-x) wrt r = 1
            
            # Constraint 2: x_i + r_i - 1 <= 0
            c2 = x[i] + r[i] - 1.0
            if c2 < 0:
                pen_val_boundary += mu * c2 * c2
                factor = 2.0 * mu * c2
                pen_grad_boundary[i] += factor * 1.0   # d/dc of (x+r-1) wrt x = 1
                pen_grad_boundary[2*n+i] += factor * 1.0   # d/dc of (x+r-1) wrt r = 1
            
            # Constraint 3: r_i - y_i <= 0
            c3 = r[i] - y[i]
            if c3 < 0:
                pen_val_boundary += mu * c3 * c3
                factor = 2.0 * mu * c3
                pen_grad_boundary[n+i] += factor * (-1.0)  # d/dc of (r-y) wrt y = -1
                pen_grad_boundary[2*n+i] += factor * 1.0   # d/dc of (r-y) wrt r = 1
            
            # Constraint 4: y_i + r_i - 1 <= 0
            c4 = y[i] + r[i] - 1.0
            if c4 < 0:
                pen_val_boundary += mu * c4 * c4
                factor = 2.0 * mu * c4
                pen_grad_boundary[n+i] += factor * 1.0   # d/dc of (y+r-1) wrt y = 1
                pen_grad_boundary[2*n+i] += factor * 1.0   # d/dc of (y+r-1) wrt r = 1
        
        total_val = obj_val + pen_val + pen_val_boundary
        total_grad = obj_grad + pen_grad + pen_grad_boundary
        return total_val, total_grad
    
    def check_validity(params):
        """Check if the solution is valid (no overlaps, all in bounds)."""
        n = N
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        
        # Check radii are positive
        if np.any(r <= 0):
            return False
        
        # Check boundaries
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            return False
        
        # Check circle-circle non-overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-9:
                    return False
        
        return True
    
    def get_sum_radii(params):
        return np.sum(params[2*N:3*N])
    
    # Initial guesses: use a grid-based initialization with some randomness
    best_params = None
    best_score = -np.inf
    
    # Start with a reasonable grid initialization
    side = int(np.ceil(np.sqrt(N)))
    base_radius = 0.5 / side * 0.9  # Slightly smaller to allow movement
    
    # Try multiple initializations
    num_starts = 5
    rng = np.random.RandomState(42)
    
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
        
        # Create initial positions
        if start_idx == 0:
            # Regular grid
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) >= N:
                        break
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    positions.append((cx, cy, base_radius))
            x_init = np.array([p[0] for p in positions[:N]])
            y_init = np.array([p[1] for p in positions[:N]])
            r_init = np.array([p[2] for p in positions[:N]])
        else:
            # Perturbed grid
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) >= N:
                        break
                    cx = (i + 0.5) / side + rng.uniform(-0.1, 0.1)
                    cy = (j + 0.5) / side + rng.uniform(-0.1, 0.1)
                    cx = np.clip(cx, 0.05, 0.95)
                    cy = np.clip(cy, 0.05, 0.95)
                    positions.append((cx, cy, base_radius * rng.uniform(0.8, 1.2)))
            x_init = np.array([p[0] for p in positions[:N]])
            y_init = np.array([p[1] for p in positions[:N]])
            r_init = np.array([p[2] for p in positions[:N]])
        
        params_init = np.concatenate([x_init, y_init, r_init])
        
        # Use increasing penalty parameter
        mu = 1.0
        params = params_init.copy()
        
        for mu_iter in range(10):
            if time.time() > deadline:
                break
            
            result = minimize(
                lambda p: objective_and_grad(p, mu)[0],
                params,
                jac=lambda p: objective_and_grad(p, mu)[1],
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            
            params = result.x
            mu *= 5.0  # Increase penalty
        
        # Check validity and score
        if check_validity(params):
            score = get_sum_radii(params)
            if score > best_score:
                best_score = score
                best_params = params.copy()
    
    # If no valid solution found, return a safe grid
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        return np.array(cells)
    
    # Extract and return
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    # Clamp to ensure validity
    for i in range(N):
        r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        r[i] = max(r[i], 1e-6)
    
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
