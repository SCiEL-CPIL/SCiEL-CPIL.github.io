import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin (limit 120s, evaluator 150s)
    
    def energy(v):
        # Unpack
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        # 1. Objective: Maximize sum of radii -> Minimize -sum(r)
        obj = -np.sum(r)
        
        # 2. Overlap Penalty
        overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                rad_sum = r[i] + r[j]
                if dist < rad_sum:
                    overlap += (rad_sum - dist)**2
        
        # 3. Wall Penalty
        wall = 0.0
        for i in range(N):
            # Left
            if x[i] < r[i]: wall += (r[i] - x[i])**2
            # Right
            if x[i] > W - r[i]: wall += (x[i] - (W - r[i]))**2
            # Bottom
            if y[i] < r[i]: wall += (r[i] - y[i])**2
            # Top
            if y[i] > H - r[i]: wall += (y[i] - (H - r[i]))**2
            
        # 4. Perimeter Penalty
        perim = 0.0
        if W + H > 2.0:
            perim = (W + H - 2.0)**2
            
        # 5. Small penalty for very small W or H to avoid degenerate solutions
        size_pen = 0.0
        if W < 0.2: size_pen += (0.2 - W)**2
        if H < 0.2: size_pen += (0.2 - H)**2

        # Weights
        w_overlap = 1000.0
        w_wall = 1000.0
        w_perim = 1000.0
        w_size = 100.0
        
        return obj + w_overlap * overlap + w_wall * wall + w_perim * perim + w_size * size_pen

    def is_valid(v, tol=1e-4):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        if W + H > 2.0 + tol: return False
        if W < 0.1 or H < 0.1: return False
        
        for i in range(N):
            if r[i] < 1e-6: return False
            if x[i] - r[i] < -tol: return False
            if x[i] + r[i] - W > tol: return False
            if y[i] - r[i] < -tol: return False
            if y[i] + r[i] - H > tol: return False
            
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - tol:
                    return False
        return True

    best_value = -np.inf
    best_v = None
    
    # Initial guesses
    def get_initial_guess(seed):
        np.random.seed(seed)
        # Random rectangle aspect ratio
        W = np.random.uniform(0.8, 1.2)
        H = 2.0 - W
        
        # Grid-like initialization with noise
        cols = 5
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.05)
        
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N: break
                x[count] = (i + 0.5) * (W / cols) + np.random.uniform(-0.02, 0.02)
                y[count] = (j + 0.5) * (H / rows) + np.random.uniform(-0.02, 0.02)
                r[count] = 0.05 + np.random.uniform(-0.01, 0.01)
                count += 1
        
        # Ensure initial radii are positive
        r = np.maximum(r, 0.01)
        
        return np.concatenate([x, y, r, [W, H]])

    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        v0 = get_initial_guess(i)
        
        # Bounds for optimizer (loose, penalties handle strict bounds)
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
        for i in range(N):
            bounds.append((0.0, 2.0)) # y
        for i in range(N):
            bounds.append((1e-4, 1.0)) # r
        bounds.append((0.1, 2.0)) # W
        bounds.append((0.1, 2.0)) # H
        
        try:
            # Use L-BFGS-B for speed and bound handling
            res = minimize(
                energy, 
                v0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if is_valid(res.x):
                val = np.sum(res.x[2*N:3*N])
                if val > best_value:
                    best_value = val
                    best_v = res.x
        except Exception:
            pass

    # If no valid solution found, return a trivial valid one
    if best_v is None:
        # Small circles in a square
        W = 1.0
        H = 1.0
        side = 5
        x = np.linspace(0.2, 0.8, N)
        y = np.linspace(0.2, 0.8, N)
        r = np.full(N, 0.01)
        
        # Ensure they don't overlap in this fallback
        # Spacing is 0.6/20 = 0.03. Radius 0.01. Diameter 0.02. Safe.
        best_v = np.concatenate([x, y, r, [W, H]])
        
    x = best_v[0:N]
    y = best_v[N:2*N]
    r = best_v[2*N:3*N]
    
    result = np.column_stack((x, y, r))
    return result
