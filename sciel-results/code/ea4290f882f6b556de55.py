"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configuration: grid layout
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    
    # Start with a slightly smaller grid to have room
    x0 = np.array([(c[0], c[1]) for c in cells[:N]])
    r0 = np.full(N, 0.5 / side - 0.02)
    
    best_x = x0.copy()
    best_r = r0.copy()
    best_score = np.sum(best_r)
    
    def objective(params):
        """Minimize negative sum of radii subject to constraints."""
        x = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        
        # Boundary constraints: circles must be inside [0,1]^2
        # x[i] - r[i] >= 0, x[i] + r[i] <= 1, y[i] - r[i] >= 0, y[i] + r[i] <= 1
        boundary_violation = 0
        for i in range(N):
            boundary_violation += max(0, r[i] - x[i, 0])
            boundary_violation += max(0, r[i] - (1 - x[i, 0]))
            boundary_violation += max(0, r[i] - x[i, 1])
            boundary_violation += max(0, r[i] - (1 - x[i, 1]))
        
        # Distance constraints: ||x_i - x_j|| >= r_i + r_j
        dist_violation = 0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt(np.sum((x[i] - x[j])**2))
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    dist_violation += (min_dist - dist)
        
        # Soft penalty for violations
        penalty = 1000 * (boundary_violation + dist_violation)
        
        return -np.sum(r) + penalty
    
    def objective_grad(params):
        """Gradient of the objective function."""
        x = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        
        grad = np.zeros_like(params)
        
        # Gradient of -sum(r)
        grad[2*N:] = -1.0
        
        # Gradient of boundary violations
        for i in range(N):
            # x[i,0] - r[i] >= 0 => violation = max(0, r[i] - x[i,0])
            if r[i] - x[i, 0] > 0:
                grad[2*i] += 1000 * (-1)      # d/dx[i,0]
                grad[2*N+i] += 1000 * (1)     # d/dr[i]
            # 1 - x[i,0] - r[i] >= 0 => violation = max(0, r[i] - (1-x[i,0]))
            if r[i] - (1 - x[i, 0]) > 0:
                grad[2*i] += 1000 * (1)       # d/dx[i,0]
                grad[2*N+i] += 1000 * (1)     # d/dr[i]
            # x[i,1] - r[i] >= 0
            if r[i] - x[i, 1] > 0:
                grad[2*i+1] += 1000 * (-1)    # d/dy[i]
                grad[2*N+i] += 1000 * (1)     # d/dr[i]
            # 1 - x[i,1] - r[i] >= 0
            if r[i] - (1 - x[i, 1]) > 0:
                grad[2*i+1] += 1000 * (1)     # d/dy[i]
                grad[2*N+i] += 1000 * (1)     # d/dr[i]
        
        # Gradient of distance violations
        for i in range(N):
            for j in range(i+1, N):
                diff = x[i] - x[j]
                dist = np.sqrt(np.sum(diff**2))
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-10:
                    # violation = min_dist - dist
                    # d/dx[i] = -diff/dist
                    # d/dx[j] = +diff/dist
                    # d/dr[i] = 1
                    # d/dr[j] = 1
                    factor = 1000 / dist
                    grad[2*i] += -factor * diff[0]
                    grad[2*i+1] += -factor * diff[1]
                    grad[2*j] += factor * diff[0]
                    grad[2*j+1] += factor * diff[1]
                    grad[2*N+i] += 1000
                    grad[2*N+j] += 1000
        
        return grad
    
    # Use L-BFGS-B with bounds
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
        bounds.append((0.01, 0.49))  # r
    
    params0 = np.concatenate([x0.flatten(), r0])
    
    # Multiple restarts with different initial conditions
    restarts = [
        ("grid", x0, r0),
    ]
    
    # Add some random restarts
    np.random.seed(42)
    for _ in range(5):
        x_rand = np.random.uniform(0.1, 0.9, (N, 2))
        r_rand = np.full(N, 0.15)
        restarts.append(("random", x_rand, r_rand))
    
    for name, x_init, r_init in restarts:
        if time.time() > deadline - 10:
            break
        
        params_init = np.concatenate([x_init.flatten(), r_init])
        
        try:
            result = minimize(
                objective,
                params_init,
                jac=objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score * -1:
                x_opt = result.x[:2*N].reshape(N, 2)
                r_opt = result.x[2*N:]
                
                # Verify constraints
                valid = True
                for i in range(N):
                    if x_opt[i, 0] - r_opt[i] < -1e-6 or x_opt[i, 0] + r_opt[i] > 1 + 1e-6:
                        valid = False
                        break
                    if x_opt[i, 1] - r_opt[i] < -1e-6 or x_opt[i, 1] + r_opt[i] > 1 + 1e-6:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt(np.sum((x_opt[i] - x_opt[j])**2))
                            if dist < r_opt[i] + r_opt[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt.copy()
                        best_r = r_opt.copy()
        except Exception as e:
            pass
    
    # Final polish: try to increase radii slightly
    if time.time() < deadline - 5:
        # Try small perturbations to see if we can improve
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            params = np.concatenate([best_x.flatten(), best_r])
            try:
                result = minimize(
                    objective,
                    params,
                    jac=objective_grad,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                x_opt = result.x[:2*N].reshape(N, 2)
                r_opt = result.x[2*N:]
                
                valid = True
                for i in range(N):
                    if x_opt[i, 0] - r_opt[i] < -1e-6 or x_opt[i, 0] + r_opt[i] > 1 + 1e-6:
                        valid = False
                        break
                    if x_opt[i, 1] - r_opt[i] < -1e-6 or x_opt[i, 1] + r_opt[i] > 1 + 1e-6:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt(np.sum((x_opt[i] - x_opt[j])**2))
                            if dist < r_opt[i] + r_opt[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt.copy()
                        best_r = r_opt.copy()
            except:
                pass
    
    # Build final result
    result = np.zeros((N, 3))
    result[:, 0] = best_x[:, 0]
    result[:, 1] = best_x[:, 1]
    result[:, 2] = best_r
    
    # Ensure all circles are strictly inside
    for i in range(N):
        r = result[i, 2]
        result[i, 0] = np.clip(result[i, 0], r + 1e-7, 1 - r - 1e-7)
        result[i, 1] = np.clip(result[i, 1], r + 1e-7, 1 - r - 1e-7)
    
    return result
# EVOLVE-BLOCK-END
