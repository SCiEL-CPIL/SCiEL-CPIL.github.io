"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 32

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def push_apart(vars, n_iter=100):
        """Push apart overlapping circles iteratively."""
        x = vars[0::3].copy()
        y = vars[1::3].copy()
        r = vars[2::3].copy()
        
        for _ in range(n_iter):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d = np.sqrt(dx*dx + dy*dy)
                    r_sum = r[i] + r[j]
                    
                    if d < r_sum - 1e-9:
                        if d < 1e-9:
                            dx, dy = 1e-9, 0.0
                            d = 1e-9
                        overlap = r_sum - d
                        # Push apart proportionally to radii
                        total_r = r[i] + r[j]
                        wi = r[i] / total_r
                        wj = r[j] / total_r
                        push = overlap * 0.5
                        x[i] -= push * wi * dx / d
                        y[i] -= push * wi * dy / d
                        x[j] += push * wj * dx / d
                        y[j] += push * wj * dy / d
                        moved = True
            
            # Clip to boundaries
            for i in range(N):
                x[i] = np.clip(x[i], r[i], 1.0 - r[i])
                y[i] = np.clip(y[i], r[i], 1.0 - r[i])
            
            if not moved:
                break
        
        vars[0::3] = x
        vars[1::3] = y
        return vars
    
    def objective(vars):
        """Smooth objective: -sum(r) + penalty terms using soft min."""
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # Ensure positive radii with a floor
        r_safe = np.maximum(r, 1e-6)
        
        # Objective: minimize -sum(r)
        obj = -np.sum(r_safe)
        
        # Smooth overlap penalty using soft min
        beta = 50.0  # Sharpness of soft min
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy + 1e-12)
                r_sum = r_safe[i] + r_safe[j]
                # Soft min of (d - r_sum): penalize when d < r_sum
                diff = d - r_sum
                # Use softplus-like penalty: max(0, -diff) approximated smoothly
                penalty = np.logaddexp(0, -beta * diff) / beta
                obj += penalty
        
        # Boundary penalties (smooth)
        for i in range(N):
            # Left
            obj += np.logaddexp(0, -beta * (x[i] - r_safe[i])) / beta
            # Right
            obj += np.logaddexp(0, -beta * (1.0 - x[i] - r_safe[i])) / beta
            # Bottom
            obj += np.logaddexp(0, -beta * (y[i] - r_safe[i])) / beta
            # Top
            obj += np.logaddexp(0, -beta * (1.0 - y[i] - r_safe[i])) / beta
        
        return obj
    
    def objective_grad(vars):
        """Gradient of the smooth objective."""
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        r_safe = np.maximum(r, 1e-6)
        
        grad = np.zeros(3 * N)
        
        # Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        beta = 50.0
        
        # Pairwise overlap gradients
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy + 1e-12
                d = np.sqrt(d_sq)
                r_sum = r_safe[i] + r_safe[j]
                diff = d - r_sum
                
                # Soft min gradient: -sigmoid(-beta * diff)
                sig = 1.0 / (1.0 + np.exp(beta * diff))
                coeff = -sig * beta / d  # d/d(d-r_sum) of logaddexp(0, -beta*(d-r_sum))/beta
                
                # d(diff)/d(x_i) = dx/d, etc.
                grad[3*i + 0] += coeff * dx
                grad[3*i + 1] += coeff * dy
                grad[3*i + 2] += coeff * (-1.0)
                
                grad[3*j + 0] += coeff * (-dx)
                grad[3*j + 1] += coeff * (-dy)
                grad[3*j + 2] += coeff * (-1.0)
        
        # Boundary gradients
        for i in range(N):
            # Left: logaddexp(0, -beta*(x_i - r_i))
            sig = 1.0 / (1.0 + np.exp(beta * (x[i] - r_safe[i])))
            grad[3*i + 0] += -sig * beta
            grad[3*i + 2] += sig * beta
            
            # Right: logaddexp(0, -beta*(1 - x_i - r_i))
            sig = 1.0 / (1.0 + np.exp(beta * (1.0 - x[i] - r_safe[i])))
            grad[3*i + 0] += sig * beta
            grad[3*i + 2] += sig * beta
            
            # Bottom
            sig = 1.0 / (1.0 + np.exp(beta * (y[i] - r_safe[i])))
            grad[3*i + 1] += -sig * beta
            grad[3*i + 2] += sig * beta
            
            # Top
            sig = 1.0 / (1.0 + np.exp(beta * (1.0 - y[i] - r_safe[i])))
            grad[3*i + 1] += sig * beta
            grad[3*i + 2] += sig * beta
        
        return grad
    
    def is_valid(vars, tol=1e-5):
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        for i in range(N):
            if x[i] < r[i] - tol or x[i] + r[i] > 1.0 + tol:
                return False
            if y[i] < r[i] - tol or y[i] + r[i] > 1.0 + tol:
                return False
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True
    
    def generate_initial_config(seed):
        """Generate an initial configuration with given seed."""
        rng = np.random.RandomState(seed)
        
        # Grid-based initialization with some randomness
        side = 6  # 6x6 grid gives 36 positions, pick 32
        positions = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side + rng.uniform(-0.1, 0.1)
                y = (j + 0.5) / side + rng.uniform(-0.1, 0.1)
                positions.append([x, y])
        
        positions = np.array(positions)
        # Shuffle and take N
        idx = rng.permutation(len(positions))[:N]
        init_points = positions[idx]
        
        # Initial radii
        init_radii = np.full(N, 0.04)
        
        vars = np.zeros(3 * N)
        vars[0::3] = init_points[:, 0]
        vars[1::3] = init_points[:, 1]
        vars[2::3] = init_radii
        
        # Push apart to get a feasible starting point
        vars = push_apart(vars, n_iter=50)
        
        return vars
    
    best_val = 0.0
    best_vars = None
    
    # Try multiple restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        seed = 42 + restart
        x0 = generate_initial_config(seed)
        
        # Bounds for L-BFGS-B
        bounds = []
        for i in range(N):
            bounds.append((1e-4, 1.0))  # x
            bounds.append((1e-4, 1.0))  # y
            bounds.append((0.01, 0.5))  # r
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                jac=objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            if result.success or result.fun < 1e-3:
                x_opt = result.x
                r_opt = x_opt[2::3]
                current_sum = np.sum(r_opt)
                
                if is_valid(x_opt):
                    if current_sum > best_val:
                        best_val = current_sum
                        best_vars = x_opt.copy()
        except Exception:
            continue
    
    # Final polish: if we have a valid solution, do a high-precision local optimization
    if best_vars is not None and time.time() < deadline - 10:
        x_polish = best_vars.copy()
        try:
            bounds = []
            for i in range(N):
                bounds.append((1e-4, 1.0))
                bounds.append((1e-4, 1.0))
                bounds.append((0.01, 0.5))
            
            result = minimize(
                objective,
                x_polish,
                jac=objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
            )
            
            if result.success or result.fun < 1e-6:
                r_polish = result.x[2::3]
                current_sum = np.sum(r_polish)
                
                if is_valid(result.x) and current_sum > best_val:
                    best_val = current_sum
                    best_vars = result.x.copy()
        except Exception:
            pass
    
    # Fallback if no valid solution found
    if best_vars is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Return array (x, y, radius)
    result = np.column_stack((best_vars[0::3], best_vars[1::3], best_vars[2::3]))
    return result
# EVOLVE-BLOCK-END
