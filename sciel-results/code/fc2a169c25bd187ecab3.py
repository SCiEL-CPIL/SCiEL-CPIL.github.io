import time
import math
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 140.0

def _is_valid(circles, w, tol=1e-6):
    if circles is None:
        return False
    circles = np.asarray(circles)
    if circles.shape[0] != N:
        return False
    if not np.all(np.isfinite(circles)):
        return False
    r = circles[:, 2]
    if np.any(r < -tol):
        return False
    h = 2.0 - w
    x = circles[:, 0]
    y = circles[:, 1]
    for i in range(N):
        if abs(x[i]) + r[i] > w/2 + tol:
            return False
        if abs(y[i]) + r[i] > h/2 + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - tol:
                return False
    return True

def _score(circles, w, tol=1e-6):
    if not _is_valid(circles, w, tol):
        return -1e9
    return float(np.sum(circles[:, 2]))

def _objective(v, mu):
    w = v[0]
    # Clamp w to valid range [0.5, 1.5]
    w = np.clip(w, 0.5, 1.5)
    h = 2.0 - w
    x = v[1:N+1]
    y = v[N+1:2*N+1]
    r = v[2*N+1:3*N+1]
    
    # Soft positive constraint for r
    r = np.maximum(r, 1e-8)
    
    pen = 0.0
    # Boundary penalties
    for i in range(N):
        pen += max(0.0, abs(x[i]) + r[i] - w/2)**2
        pen += max(0.0, abs(y[i]) + r[i] - h/2)**2
        
    # Overlap penalties
    for i in range(N):
        for j in range(i+1, N):
            d = math.hypot(x[i]-x[j], y[i]-y[j])
            req = r[i] + r[j]
            if d < req:
                pen += (req - d)**2
                
    return mu * pen - np.sum(r)

def _initial_state(w_init, seed):
    np.random.seed(seed)
    h_init = 2.0 - w_init
    
    # Create a grid of points
    # Approximate number of circles per side
    cols = 5
    rows = 5
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.full(N, 0.1) # Initial small radius
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= N: break
            # Hexagonal offset for even rows
            offset = 0.5 * (w_init / cols) if i % 2 == 1 else 0.0
            x[idx] = -w_init/2 + (j + 0.5) * (w_init / cols) + offset
            y[idx] = -h_init/2 + (i + 0.5) * (h_init / rows)
            idx += 1
            
    # Add some randomness
    x += 0.05 * (np.random.rand(N) - 0.5)
    y += 0.05 * (np.random.rand(N) - 0.5)
    
    v = np.concatenate(([w_init], x, y, r))
    return v

def construct():
    best_circles = None
    best_w = 1.0
    best_score = -1e9
    
    # Try several aspect ratios and seeds
    w_candidates = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for w_init in w_candidates:
        if time.time() > DEADLINE - 10:
            break
            
        for seed in [1, 2, 3]:
            if time.time() > DEADLINE - 5:
                break
                
            v0 = _initial_state(w_init, seed)
            
            # Optimization
            # Use L-BFGS-B
            # mu controls the trade-off between feasibility and radius size
            # Start with a moderate mu, then increase?
            # Or just use a high mu to ensure feasibility.
            
            # Let's try a sequence of mu values
            mus = [10.0, 100.0, 1000.0]
            current_v = v0
            
            for mu in mus:
                if time.time() > DEADLINE - 5:
                    break
                    
                res = minimize(_objective, current_v, args=(mu,), method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12})
                if res.success or res.fun < _objective(current_v, mu):
                    current_v = res.x
                    
            # Extract results
            w_final = np.clip(current_v[0], 0.5, 1.5)
            x_final = current_v[1:N+1]
            y_final = current_v[N+1:2*N+1]
            r_final = current_v[2*N+1:3*N+1]
            
            circles = np.column_stack([x_final, y_final, r_final])
            
            # Check validity and score
            if _is_valid(circles, w_final):
                s = _score(circles, w_final)
                if s > best_score:
                    best_score = s
                    best_circles = circles.copy()
                    best_w = w_final
                    
    if best_circles is None:
        # Fallback: simple grid
        side = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.1)
        for idx in range(N):
            i = idx % side
            j = idx // side
            x[idx] = (i - 2) * 0.2
            y[idx] = (j - 2) * 0.2
        best_circles = np.column_stack([x, y, r])
        best_w = 1.0
        
    return best_circles
