"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _validity(arr):
    """Check if arrangement is valid: circles inside [0,1]^2 and disjoint interiors."""
    n = len(arr)
    if n != N:
        return False
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    # Check bounds
    if np.any(x < -1e-9) or np.any(y < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(r < -1e-9):
        return False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Initial seed: grid of equal circles
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    best_arr = np.array(cells[:N])
    best_val = np.sum(best_arr[:, 2])
    
    # Try multiple random initializations and optimize
    rng = np.random.RandomState(42)
    num_starts = 20
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization: scatter circles
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = np.full(N, 0.03)
        
        arr = np.column_stack([x, y, r])
        
        # Optimize using scipy
        def objective(params):
            """Minimize negative sum of radii (to maximize sum of radii)."""
            p = params.reshape(3, N)
            xi, yi, ri = p[0], p[1], p[2]
            # Penalty for being outside [0,1]^2
            pen = 0.0
            for i in range(N):
                if xi[i] < 0: pen += (100 * (xi[i]))**2
                if xi[i] > 1: pen += (100 * (xi[i] - 1))**2
                if yi[i] < 0: pen += (100 * (yi[i]))**2
                if yi[i] > 1: pen += (100 * (yi[i] - 1))**2
                if ri[i] < 0: pen += (100 * ri[i])**2
            # Penalty for overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xi[i] - xi[j]
                    dy = yi[i] - yi[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    overlap = ri[i] + ri[j] - dist
                    if overlap > 0:
                        pen += (100 * overlap)**2
            return -np.sum(ri) + pen
        
        # Initial parameters
        x0 = arr.flatten()
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
            bounds.append((0, 0.5))  # radius
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < -best_val + 1e-6:
                new_arr = res.x.reshape(3, N).T
                new_val = np.sum(new_arr[:, 2])
                
                # Validate
                if _validity(new_arr):
                    if new_val > best_val:
                        best_val = new_val
                        best_arr = new_arr
        except Exception:
            continue
    
    # Final polish: try to improve the best solution found
    if time.time() < deadline - 10.0:
        def objective2(params):
            p = params.reshape(3, N)
            xi, yi, ri = p[0], p[1], p[2]
            pen = 0.0
            for i in range(N):
                if xi[i] < 0: pen += (1000 * (xi[i]))**2
                if xi[i] > 1: pen += (1000 * (xi[i] - 1))**2
                if yi[i] < 0: pen += (1000 * (yi[i]))**2
                if yi[i] > 1: pen += (1000 * (yi[i] - 1))**2
                if ri[i] < 0: pen += (1000 * ri[i])**2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xi[i] - xi[j]
                    dy = yi[i] - yi[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    overlap = ri[i] + ri[j] - dist
                    if overlap > 0:
                        pen += (1000 * overlap)**2
            return -np.sum(ri) + pen
        
        x0 = best_arr.flatten()
        bounds2 = []
        for i in range(N):
            bounds2.append((0, 1))
            bounds2.append((0, 1))
            bounds2.append((0, 0.5))
        
        try:
            res2 = minimize(objective2, x0, method='L-BFGS-B', bounds=bounds2,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res2.success or res2.fun < -best_val + 1e-8:
                new_arr2 = res2.x.reshape(3, N).T
                new_val2 = np.sum(new_arr2[:, 2])
                if _validity(new_arr2) and new_val2 > best_val:
                    best_val = new_val2
                    best_arr = new_arr2
        except Exception:
            pass
    
    return best_arr
# EVOLVE-BLOCK-END
