import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    best_value = -np.inf
    best_result = None

    # Helper to check validity and calculate score
    def evaluate(x, y, r, W, H):
        # Check perimeter
        if abs(2 * (W + H) - 4) > 1e-5:
            return -np.inf, False
        
        # Check bounds and overlaps
        for i in range(N):
            if r[i] <= 0:
                return -np.inf, False
            if x[i] < r[i] - 1e-5 or x[i] > W - r[i] + 1e-5:
                return -np.inf, False
            if y[i] < r[i] - 1e-5 or y[i] > H - r[i] + 1e-5:
                return -np.inf, False
            for j in range(i + 1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - 1e-5:
                    return -np.inf, False
        
        return np.sum(r), True

    # Objective function for L-BFGS-B
    def objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        # Soft penalties
        penalty = 0.0
        
        # Perimeter constraint
        perim = 2 * (W + H)
        penalty += (perim - 4)**2 * 100.0
        
        # Boundary constraints
        for i in range(N):
            # x_i >= r_i
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2 * 10.0
            # x_i + r_i <= W
            if x[i] + r[i] > W:
                penalty += (x[i] + r[i] - W)**2 * 10.0
            # y_i >= r_i
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2 * 10.0
            # y_i + r_i <= H
            if y[i] + r[i] > H:
                penalty += (y[i] + r[i] - H)**2 * 10.0
        
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2:
                    # Penalty for overlap
                    penalty += (rad_sum**2 - dist_sq) * 10.0
        
        return -np.sum(r) + penalty

    # Initialize multiple starts
    starts = []
    
    # Start 1: Grid
    side = int(np.ceil(np.sqrt(N)))
    x_grid = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
    y_grid = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:N])
    r_grid = np.full(N, 0.5 / side - 1e-3)
    W_grid = 1.0
    H_grid = 1.0
    starts.append(np.concatenate([x_grid, y_grid, r_grid, [W_grid, H_grid]]))
    
    # Start 2: Hexagonal-like packing
    x_hex = np.zeros(N)
    y_hex = np.zeros(N)
    r_hex = np.full(N, 0.15)
    W_hex = 1.0
    H_hex = 1.0
    count = 0
    for row in range(10):
        for col in range(10):
            if count >= N:
                break
            x_hex[count] = 0.1 + col * 0.2 + (0.1 if row % 2 else 0)
            y_hex[count] = 0.1 + row * 0.17
            count += 1
            if count >= N:
                break
    max_x = np.max(x_hex)
    max_y = np.max(y_hex)
    scale = min(1.0 / max_x, 1.0 / max_y)
    x_hex *= scale
    y_hex *= scale
    r_hex *= scale * 0.9
    starts.append(np.concatenate([x_hex, y_hex, r_hex, [W_hex, H_hex]]))

    # Start 3: Random (seeded for reproducibility)
    np.random.seed(42)
    x_rand = np.random.rand(N) * 0.8 + 0.1
    y_rand = np.random.rand(N) * 0.8 + 0.1
    r_rand = np.full(N, 0.1)
    W_rand = 1.0
    H_rand = 1.0
    starts.append(np.concatenate([x_rand, y_rand, r_rand, [W_rand, H_rand]]))

    bounds = []
    for i in range(N):
        bounds.append((0, 2)) # x
    for i in range(N):
        bounds.append((0, 2)) # y
    for i in range(N):
        bounds.append((0, 1)) # r
    bounds.append((0.1, 1.9)) # W
    bounds.append((0.1, 1.9)) # H

    for start in starts:
        res = minimize(objective, start, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or res.fun < best_value:
            x_final = res.x[:N]
            y_final = res.x[N:2*N]
            r_final = res.x[2*N:3*N]
            W_final = res.x[3*N]
            H_final = res.x[3*N+1]
            
            # Post-processing: Fix floating point errors
            perim = 2 * (W_final + H_final)
            if perim != 4:
                scale = 4.0 / perim
                x_final *= scale
                y_final *= scale
                r_final *= scale
                W_final *= scale
                H_final *= scale

            val, valid = evaluate(x_final, y_final, r_final, W_final, H_final)
            if valid and val > best_value:
                best_value = val
                best_result = np.column_stack((x_final, y_final, r_final))
                
    if best_result is None:
        # Fallback to grid
        return np.column_stack((x_grid, y_grid, r_grid))
        
    return best_result
