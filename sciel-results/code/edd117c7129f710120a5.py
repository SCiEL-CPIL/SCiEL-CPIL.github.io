"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def construct():
    """
    Pack 26 circles in the unit square maximizing sum of radii.
    Uses a multi-start local optimization approach with random initialization
    and gradient-based refinement, followed by a simple relaxation to ensure
    validity.
    """
    best_value = 0.0
    best_circles = None
    
    n = N
    # Number of random restarts to stay within time limit
    n_restarts = 8
    
    for restart in range(n_restarts):
        # Random initial positions and radii
        np.random.seed(restart * 12345 + 7)
        
        # Initialize with random positions in [0.1, 0.9] and random radii
        # We'll optimize both positions and radii
        x = np.random.uniform(0.1, 0.9, n)
        y = np.random.uniform(0.1, 0.9, n)
        r = np.random.uniform(0.02, 0.12, n)
        
        # Ensure initial feasibility (roughly)
        # We'll let the optimizer handle it, but start from a somewhat feasible state
        
        def objective(params):
            """
            Minimize negative sum of radii, with penalties for constraint violations.
            params: [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
            """
            x = params[:n]
            y = params[n:2*n]
            r = params[2*n:]
            
            # Penalty for circles going outside the unit square
            penalty = 0.0
            for i in range(n):
                if x[i] - r[i] < 0:
                    penalty += (x[i] - r[i]) ** 2
                if x[i] + r[i] > 1:
                    penalty += (x[i] + r[i] - 1) ** 2
                if y[i] - r[i] < 0:
                    penalty += (y[i] - r[i]) ** 2
                if y[i] + r[i] > 1:
                    penalty += (y[i] + r[i] - 1) ** 2
            
            # Penalty for overlapping circles
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            # We want to maximize sum of radii, so minimize negative sum
            value = -np.sum(r)
            return value + 10.0 * penalty
        
        # Initial parameters
        params0 = np.concatenate([x, y, r])
        
        # Optimize
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-10})
        except Exception:
            continue
        
        if result.success or result.fun < best_value if best_circles is None else result.fun < best_value:
            # Check if the solution is valid
            x_opt = result.x[:n]
            y_opt = result.x[n:2*n]
            r_opt = result.x[2*n:]
            
            # Verify validity
            valid = True
            for i in range(n):
                if x_opt[i] - r_opt[i] < -1e-6 or x_opt[i] + r_opt[i] > 1 + 1e-6:
                    valid = False
                    break
                if y_opt[i] - r_opt[i] < -1e-6 or y_opt[i] + r_opt[i] > 1 + 1e-6:
                    valid = False
                    break
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dx = x_opt[i] - x_opt[j]
                        dy = y_opt[i] - y_opt[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r_opt[i] + r_opt[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Clip to ensure exact feasibility
                r_opt = np.maximum(r_opt, 1e-6)
                x_opt = np.clip(x_opt, r_opt, 1 - r_opt)
                y_opt = np.clip(y_opt, r_opt, 1 - r_opt)
                
                # Re-verify after clipping
                valid2 = True
                for i in range(n):
                    if x_opt[i] - r_opt[i] < -1e-9 or x_opt[i] + r_opt[i] > 1 + 1e-9:
                        valid2 = False
                        break
                    if y_opt[i] - r_opt[i] < -1e-9 or y_opt[i] + r_opt[i] > 1 + 1e-9:
                        valid2 = False
                        break
                if valid2:
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = x_opt[i] - x_opt[j]
                            dy = y_opt[i] - y_opt[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < r_opt[i] + r_opt[j] - 1e-9:
                                valid2 = False
                                break
                        if not valid2:
                            break
                
                if valid2:
                    value = np.sum(r_opt)
                    if value > best_value:
                        best_value = value
                        best_circles = np.column_stack([x_opt, y_opt, r_opt])
    
    # If no valid solution found, fall back to a simple valid configuration
    if best_circles is None:
        # Simple grid arrangement
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    cr = 0.5 / side - 1e-4
                    cells.append([cx, cy, cr])
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
