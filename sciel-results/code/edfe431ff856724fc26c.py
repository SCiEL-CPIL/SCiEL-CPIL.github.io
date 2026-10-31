import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 26
DEADLINE = 110.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    def get_state(x, y, r):
        return np.concatenate([x, y, r])

    def unpack_state(z):
        x = z[0:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        return x, y, r

    def objective(z, penalty_weight=100.0):
        x, y, r = unpack_state(z)
        
        # Base objective: maximize sum of radii -> minimize -sum
        obj = -np.sum(r)
        
        # Boundary violations
        # r <= x, r <= y, r <= 1-x, r <= 1-y
        viol_x = np.maximum(0, r - x)
        viol_y = np.maximum(0, r - y)
        viol_1x = np.maximum(0, r - (1 - x))
        viol_1y = np.maximum(0, r - (1 - y))
        
        boundary_violation = np.sum(viol_x**2) + np.sum(viol_y**2) + np.sum(viol_1x**2) + np.sum(viol_1y**2)
        
        # Pairwise overlap violations
        # dist(i,j) >= r_i + r_j
        # dist^2 = (x_i-x_j)^2 + (y_i-y_j)^2
        # We need dist >= r_i + r_j
        # Equivalent to dist^2 >= (r_i+r_j)^2
        # Or dist - (r_i+r_j) >= 0
        # Penalty on max(0, r_i+r_j - dist)^2
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist)**2
        
        return obj + penalty_weight * (boundary_violation + overlap_penalty)

    def gradient(z, penalty_weight=100.0):
        x, y, r = unpack_state(z)
        grad = np.zeros(3*N)
        
        # Gradient of -sum(r)
        grad[2*N:3*N] = -1.0
        
        # Boundary gradients
        # d/dx_i (r_i - x_i)^2 = -2(r_i - x_i) if r_i > x_i else 0
        viol_x = np.maximum(0, r - x)
        grad[0:N] -= 2 * viol_x
        grad[2*N:3*N] += 2 * viol_x
        
        viol_y = np.maximum(0, r - y)
        grad[N:2*N] -= 2 * viol_y
        grad[2*N:3*N] += 2 * viol_y
        
        viol_1x = np.maximum(0, r - (1 - x))
        # d/dx_i (r_i - (1-x_i))^2 = 2(r_i - (1-x_i))
        grad[0:N] += 2 * viol_1x
        grad[2*N:3*N] -= 2 * viol_1x
        
        viol_1y = np.maximum(0, r - (1 - y))
        grad[N:2*N] += 2 * viol_1y
        grad[2*N:3*N] -= 2 * viol_1y
        
        # Overlap gradients
        # P = (r_i + r_j - dist)^2
        # dP/dx_i = 2(r_i + r_j - dist) * (-(x_i - x_j) / dist)
        # dP/dy_i = 2(r_i + r_j - dist) * (-(y_i - y_j) / dist)
        # dP/dr_i = 2(r_i + r_j - dist) * 1
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    diff = min_dist - dist
                    factor = 2 * diff
                    
                    # Gradients for x, y of i
                    grad[0:N][i] += factor * (-dx / dist)
                    grad[N:2*N][i] += factor * (-dy / dist)
                    
                    # Gradients for x, y of j
                    grad[0:N][j] += factor * (dx / dist)
                    grad[N:2*N][j] += factor * (dy / dist)
                    
                    # Gradients for r
                    grad[2*N:3*N][i] += factor
                    grad[2*N:3*N][j] += factor
        
        return penalty_weight * grad

    def is_valid(x, y, r, tol=1e-5):
        # Check boundaries
        if np.any(r > x + tol) or np.any(r > y + tol) or np.any(r > 1-x + tol) or np.any(r > 1-y + tol):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def sum_radii(x, y, r):
        return np.sum(r)

    best_score = 0.0
    best_state = None
    
    # 1. Hexagonal Lattice Initialization
    # Side length approx 1/sqrt(26) * factor
    # For hex packing, area per circle is approx pi * r^2 / (sqrt(3)/2) ... 
    # Let's just place them in a hex grid and let optimizer adjust.
    side = int(np.ceil(np.sqrt(N)))
    # Create a grid of potential spots
    x_hex = np.zeros(N)
    y_hex = np.zeros(N)
    r_hex = np.zeros(N)
    
    idx = 0
    for row in range(side):
        for col in range(side):
            if idx >= N: break
            x_hex[idx] = (col + 0.5 + (row % 2) * 0.5) / side
            y_hex[idx] = (row + 0.5) / side
            # Initial radius estimate
            r_hex[idx] = 0.5 / side * 0.9
            idx += 1
            
    # 2. Random Initialization
    rng = np.random.default_rng(42)
    x_rand = rng.uniform(0.1, 0.9, N)
    y_rand = rng.uniform(0.1, 0.9, N)
    r_rand = rng.uniform(0.05, 0.1, N)
    
    # 3. Grid Initialization (perturbed)
    x_grid = np.zeros(N)
    y_grid = np.zeros(N)
    r_grid = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N: break
            x_grid[idx] = (i + 0.5) / side + rng.uniform(-0.1, 0.1)
            y_grid[idx] = (j + 0.5) / side + rng.uniform(-0.1, 0.1)
            r_grid[idx] = 0.5 / side * 0.9
            idx += 1
    
    initializations = [
        (x_hex, y_hex, r_hex),
        (x_rand, y_rand, r_rand),
        (x_grid, y_grid, r_grid)
    ]
    
    # Add more random restarts if time permits
    max_restarts = 5
    for k in range(max_restarts):
        x_r = rng.uniform(0.1, 0.9, N)
        y_r = rng.uniform(0.1, 0.9, N)
        r_r = rng.uniform(0.05, 0.1, N)
        initializations.append((x_r, y_r, r_r))
        
    for x_init, y_init, r_init in initializations:
        if time.time() - start_time > DEADLINE:
            break
            
        z0 = get_state(x_init, y_init, r_init)
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0, 0.5)) # r
            
        try:
            res = minimize(
                objective, 
                z0, 
                method='L-BFGS-B', 
                jac=gradient, 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if res.success:
                x_opt, y_opt, r_opt = unpack_state(res.x)
                
                # Verify validity
                if is_valid(x_opt, y_opt, r_opt):
                    score = sum_radii(x_opt, y_opt, r_opt)
                    if score > best_score:
                        best_score = score
                        best_state = res.x
                else:
                    # If invalid, try to repair by shrinking radii slightly
                    # or just skip if violation is large.
                    # Let's check how bad it is.
                    pass
                    
        except Exception as e:
            continue

    if best_state is None:
        # Fallback to a valid trivial solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x_final, y_final, r_final = unpack_state(best_state)
    
    # Final cleanup: ensure strict validity
    # Sometimes optimizer leaves tiny violations.
    # We can iteratively reduce radii if there are violations.
    for _ in range(10):
        valid = True
        # Check boundaries
        for i in range(N):
            min_dist_b = min(x_final[i], y_final[i], 1-x_final[i], 1-y_final[i])
            if r_final[i] > min_dist_b:
                r_final[i] = min_dist_b
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x_final[i] - x_final[j]
                dy = y_final[i] - y_final[j]
                dist = np.sqrt(dx*dx + dy*dy)
                req_dist = r_final[i] + r_final[j]
                if dist < req_dist:
                    # Reduce both radii proportionally or just scale down
                    # Simple fix: reduce the larger one or both
                    scale = dist / req_dist * 0.99
                    r_final[i] *= scale
                    r_final[j] *= scale
                    valid = False
        
        if valid:
            break
            
    result = np.column_stack([x_final, y_final, r_final])
    return result

if __name__ == "__main__":
    res = construct()
    print(np.sum(res[:, 2]))
